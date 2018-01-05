from __future__ import print_function, absolute_import
from __future__ import division
import os
import csv
import copy
import json
import logging
import subprocess
import socket
from itertools import chain
from functools import partial

import numpy as np
import h5py

# Don't import pandas here; import it locally as needed
#import pandas as pd


from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.io_util.brick import Grid, Brick, generate_bricks_from_volume_source, realign_bricks_to_new_grid, pad_brick_data_from_volume_source
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid, retrieve_node_service
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.dvid.metadata import create_labelarray, is_datainstance
from DVIDSparkServices.util import Timer, runlength_encode, choose_pyramid_depth, nonconsecutive_bincount, cpus_per_worker, num_worker_nodes, persist_and_execute, NumpyConvertingEncoder
from DVIDSparkServices.auto_retry import auto_retry

from DVIDSparkServices.io_util.volume_service.dvid_volume_service import DvidSegmentationVolumeSchema

logger = logging.getLogger(__name__)

class DVIDCheckSegmentation(Workflow):
    Schema = \
    {
        "$schema": "http://json-schema.org/schema#",
        "title": "Service to load raw and label data into DVID",
        "type": "object",
        "additionalProperties": False,
        "required": ["input", "output"],
        "properties": {
            "input": DvidSegmentationVolumeSchema,
            "output": {
                "description": "Location of debug output file",
                "type": "string"
            },
            "options" : Workflow.OptionsSchema
        }
    }

    @classmethod
    def schema(cls):
        return DVIDCheckSegmentation.Schema

    # name of application for DVID queries
    APPNAME = "dvidchecksegmentation"

    def __init__(self, config_filename):
        super(DVIDCheckSegmentation, self).__init__( config_filename,
                                                DVIDCheckSegmentation.schema(),
                                                "Check DVID Segmentation" )

    def _sanitize_config(self):
        """
        Tidy up some config values.
        """
        input_config = self.config_data["input"]
        
        assert input_config["service-type"] != "SKIP",\
            "Not allowed to skip the input!"

        for cfg in [input_config]:
            # Prepend 'http://' to the server if necessary.
            if "server" in cfg and not cfg["server"].startswith('http'):
                cfg["server"] = 'http://' + cfg["server"]

        
    def execute(self):
        self._sanitize_config()

        # hard coding block size that is 64
        # (TODO: make dynamic)
        BLKSIZE = 64
        
        input_config = self.config_data["input"]
        options = self.config_data["options"]

        input_bb_zyx = np.array(input_config["geometry"]["bounding-box"])[:,::-1]
        input_bricks, bounding_box, _input_grid = self._partition_input()


        persist_and_execute(input_bricks, f"Reading entire volume", logger)

        # find the blocks for each body 
        def extractBodyBlockIds(brick):
            """Determined blocks that intersect each body.

            FlatMap: brick -> (bodyid, [blockids])
            """
            vol = brick.volume
            offsetzyx = brick.physical_box[0]

            zsz, ysz, xsz = vol.shape
            assert zsz == BLKSIZE
            assert ysz == BLKSIZE
            assert xsz % BLKSIZE == 0

            zid = offsetzyx[0] // BLKSIZE
            yid = offsetzyx[1] // BLKSIZE
            xid = offsetzyx[2] // BLKSIZE

            bodymappings = {}

            for blockspot in range(0, xsz, BLKSIZE):
                bodyids = np.unique(vol[:,:,blockspot:(blockspot+BLKSIZE)])
                
                for bodyid in bodyids:
                    if bodyid == 0:
                        # ignore background bodies
                        continue
                    if bodyid not in bodymappings:
                        bodymappings[bodyid] = []
                    bodymappings[bodyid].append((zid, yid, xid))
                xid += 1

            res = []
            for bodyid, mappings in bodymappings.items():
                res.append((bodyid, mappings))
            return res

        allbodies = input_bricks.flatMap(extractBodyBlockIds)
        del input_bricks

        # combine body information across RDD 
        def combineBodyInfo(part1, part2):
            part1.extend(part2)
            return part1
        allbodies = allbodies.reduceByKey(combineBodyInfo)
        allbodies.persist()
        
        # get global list
        globalbodylist = allbodies.map(lambda x: x[0]).collect()
        globalbodylist.sort()
        
        # group sorted bodies
        BODYLIMIT = 1000
        def findorder(bodyblocks):
            body, blocks = bodyblocks
            index = globalbodylist.index(body) // BODYLIMIT
            return (index, [(body, blocks)])
        allbodies_index = allbodies.map(findorder)

        def orderbodies(b1, b2):
            b1.extend(b2)
            return b1
        allbodies_sorted = allbodies_index.reduceByKey(orderbodies)
    
        # TODO extract indices in separate step to measure fetch time
        # fetch indices for provided block and produce list of [body, bad ids]
        server = input_config["dvid"]["server"]
        uuid = input_config["dvid"]["uuid"]
        resource_server = self.resource_server
        resource_port = self.resource_port
        labelname = input_config["dvid"]["segmentation-name"]
        appname = self.APPNAME

        def findindexerrors(bodies):
            index, bodylist = bodies
            bodymappings = {}
            rangequery = []
            for (body, bids) in bodylist:
                bodymappings[body] = bids
                rangequery.append(body)
            
            # call block index DVID API
            from libdvid import ConnectionMethod
            rangequery.sort()
            b1 = rangequery[0]
            b2 = rangequery[-1]
    
            ns = retrieve_node_service(server, uuid, resource_server, resource_port, appname)

            addr = str(labelname + "/sparsevols-coarse/" + str(b1) + "/" + str(b2))
            res = ns.custom_request(addr, None, ConnectionMethod.GET)
        
            bodyblockrle = np.fromstring(res, dtype=np.int32)
            currindex = 0
            
            bodymappingsdvid = {}
            while currindex < len(bodyblockrle):
                #  retrieve bodies
                hb = bodyblockrle[currindex]
                lb = bodyblockrle[currindex+1]
                currbody = hb | lb << 32 
                currindex += 2
                
                # retrieve runlengths
                numspans = bodyblockrle[currindex] 
                currindex += 1
                blockarray = []
                for index in range(numspans):
                    dimx = bodyblockrle[currindex] 
                    currindex += 1
                    dimy = bodyblockrle[currindex] 
                    currindex += 1
                    dimz = bodyblockrle[currindex] 
                    currindex += 1
                    runx = bodyblockrle[currindex] 
                    currindex += 1

                    # create body mappings
                    for xblock in range(dimx, dimx+runx):
                        blockarray.append((dimz, dimy, xblock))
                bodymappingsdvid[currbody] = blockarray

            allerrors = []
            # find differences
            for body, blocklist in bodymappings.items():
                if body not in bodymappingsdvid:
                    allerrors.append([True, body, blocklist])
                    continue

                # false negatives
                bset = set(blocklist)
                bsetdvid = set(bodymappingsdvid[body])
                errors = list(bset - bsetdvid)
                if len(errors) > 0:
                    allerrors.append([True, body, errors])
                
                # false positives
                errors2 = list(bsetdvid - bset)
                if len(errors2) > 0:
                    allerrors.append([False, body, errors2])
            return allerrors

        badindices = allbodies_sorted.flatMap(findindexerrors)

        # report errors
        allerrors = badindices.collect()
        
        # TODO provide link locations for bad bodies
        #self._log_neuroglancer_links()

        errorjson = []
        for bodyerror in allerrors:
            errorjson.append(bodyerror)

        fout = open(self.config_data["output"], 'w')
        fout.write(json.dumps(errorjson, indent=2, cls=NumpyConvertingEncoder))

        logger.info(f"DONE analyzing segmentation.")

    def _partition_input(self):
        """
        Map the input segmentation
        volume from DVID into an RDD of (volumePartition, data),
        using the config's bounding-box setting for the full volume region,
        using the input 'message-block-shape' as the partition size.

        Returns: (RDD, bounding_box_zyx, partition_shape_zyx)
            where:
                - RDD is (volumePartition, data)
                - bounding box is tuple (start_zyx, stop_zyx)
                - partition_shape_zyx is a tuple
            
        """
        input_config = self.config_data["input"]
        options = self.config_data["options"]

        # repartition to be z=blksize, y=blksize, x=runlength
        brick_shape_zyx = input_config["message-block-shape"][::-1]
        input_grid = Grid(brick_shape_zyx, (0,0,0))
        
        input_bb_zyx = np.array(input_config["bounding-box"])[:,::-1]

        # Aim for 2 GB RDD partitions
        GB = 2**30
        target_partition_size_voxels = 2 * GB // np.uint64().nbytes

        sparkdvid_input_context = sparkdvid(self.sc, input_config["server"], input_config["uuid"], self)
        bricks = sparkdvid_input_context.parallelize_bounding_box( input_config["segmentation-name"], input_bb_zyx, input_grid, target_partition_size_voxels )
        return bricks, input_bb_zyx, input_grid
    
    def _log_neuroglancer_links(self):
        """
        Write a link to the log file for viewing the segmentation data after it is ingested.
        We assume that the output server is hosting neuroglancer at http://<server>:<port>/neuroglancer/
        """
        for index, output_config in enumerate(self.config_data["outputs"]):
            server = output_config["server"] # Note: Begins with http://
            uuid = output_config["uuid"]
            instance = output_config["segmentation-name"]
            
            output_box_xyz = np.array(output_config["bounding-box"])
            output_center_xyz = (output_box_xyz[0] + output_box_xyz[1]) / 2
            
            link_prefix = f"{server}/neuroglancer/#!"
            link_json = \
            {
                "layers": {
                    "segmentation": {
                        "type": "segmentation",
                        "source": f"dvid://{server}/{uuid}/{instance}"
                    }
                },
                "navigation": {
                    "pose": {
                        "position": {
                            "voxelSize": [8,8,8],
                            "voxelCoordinates": output_center_xyz.tolist()
                        }
                    },
                    "zoomFactor": 8
                }
            }
            logger.info(f"Neuroglancer link to output {index}: {link_prefix}{json.dumps(link_json)}")


