import copy
import json
import logging

import requests
import numpy as np

from DVIDSparkServices.io_util.partitionSchema import volumePartition, VolumeOffset, PartitionDims, partitionSchema
from DVIDSparkServices.sparkdvid import sparkdvid
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.dvid.metadata import create_labelarray
from libdvid.util.roi_utils import copy_roi, RoiInfo
from DVIDSparkServices.reconutils.downsample import downsample_3Dlabels
from DVIDSparkServices.util import Timer, runlength_encode
#from DVIDSparkServices.dvid.local_server import ensure_dicedstore_is_running

class CopySegmentation(Workflow):
    
    DataInfoSchema = \
    {
        "type": "object",
        "default": {},
        "required": ["input", "output", "roi"],
        "additionalProperties": False,
        "properties": {
            #
            # INPUT
            #
            "input": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "required": ["server", "uuid", "segmentation-name"],
                "properties": {
                    "server": {
                        # Note: "node-local" is not supported yet.  Start the DVID server yourself and use 127.0.0.1 
                        "description": "location of DVID server to READ.  Either IP:PORT or the special word 'node-local'.",
                        "type": "string",
                    },
                    "database-location": {
                        "description": "If 'server' is 'node-local', then this is the location of the database they'll use.",
                        "type": "string"
                    },
                    "uuid": {
                        "description": "version node for READING segmentation",
                        "type": "string"
                    },
                    "segmentation-name": {
                        "description": "The labels instance to READ from. Instance may be either googlevoxels or labelblk.",
                        "type": "string",
                        "minLength": 1
                    }
                }
            },
    
            #
            # OUTPUT
            #
            "output": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "required": ["server", "uuid", "segmentation-name"],
                "properties": {
                    "server": {
                        "description": "location of DVID server to WRITE",
                        "type": "string",
                    },
                    "uuid": {
                        "description": "version node for WRITING segmentation",
                        "type": "string"
                    },
                    "segmentation-name": {
                        "description": "The labels instance to WRITE to.  If necessary, will be created (as labelblk).",
                        "type": "string",
                        "minLength": 1
                    },
                    "block-size": {
                        "description": "The DVID blocksize for new segmentation instances. Ignored if the output segmentation instance already exists.",
                        "type": "integer",
                        "default": 64
                    }
                }
            },

            #
            # ROI
            #
            "roi": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "required": ["name"],
                "properties": {
                    "name": {
                        "description": "region of interest to copy",
                        "type": "string",
                        "minLength": 1
                    },
                    "partition-method": {
                        "description": "Strategy to divide the ROI into substacks for processing.",
                        "type": "string",
                        "minLength": 1,
                        "enum": ["ask-dvid", "grid-aligned", "grid-aligned-32", "grid-aligned-64", "grid-aligned-128", "grid-aligned-256", "grid-aligned-512"],
                        "default": "ask-dvid"
                    },
                    "partition-filter": {
                        "description": "Optionally remove substacks from the compute set based on some criteria",
                        "type": "string",
                        "minLength": 1,
                        "enum": ["all", "interior-only"],
                        "default": "all"
                    }
                }
            }
        }
    }

    OptionsSchema = copy.copy(Workflow.OptionsSchema)
    OptionsSchema["properties"].update(
    {
        "chunk-size": {
            "description": "Size of block to download in each thread",
            "type": "integer",
            "default": 512
        },
        "pyramid-depth": {
            "description": "Number of pyramid levels to generate (0 means choose automatically, -1 means no pyramid)",
            "type": "integer",
            "default": 0 # automatic by default
        },
        "blockwritelimit": {
           "description": "Maximum number of blocks written per task request (0=no limit)",
           "type": "integer",
           "default": 100
        }
    })

    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to load raw and label data into DVID",
      "type": "object",
      "properties": {
        "data-info": DataInfoSchema,
        "options" : OptionsSchema
      }
    }

    @staticmethod
    def dumpschema():
        return json.dumps(CopySegmentation.Schema)

    # name of application for DVID queries
    APPNAME = "copysegmentation"

    def __init__(self, config_filename):
        super(CopySegmentation, self).__init__( config_filename,
                                                CopySegmentation.dumpschema(),
                                                "Copy Segmentation" )

        input_config = self.config_data["data-info"]["input"]
        output_config = self.config_data["data-info"]["output"]
        roi_config = self.config_data["data-info"]["roi"]

        for cfg in (input_config, output_config):
            # Prepend 'http://' if necessary.
            if not cfg['server'].startswith('http'):
                cfg['server'] = 'http://' + cfg['server']

        # Convert from unicode for easier C++ calls
            cfg["server"] = str(cfg["server"])
            cfg["uuid"] = str(cfg["uuid"])
            cfg["segmentation-name"] = str(cfg["segmentation-name"])

        roi_config["name"] = str(roi_config["name"])

        # create spark dvid contexts
        self.sparkdvid_input_context = sparkdvid.sparkdvid(self.sc, input_config["server"], input_config["uuid"], self)
        self.sparkdvid_output_context = sparkdvid.sparkdvid(self.sc, output_config["server"], output_config["uuid"], self)


    def execute(self):
        input_config = self.config_data["data-info"]["input"]
        output_config = self.config_data["data-info"]["output"]
        roi_config = self.config_data["data-info"]["roi"]
        options = self.config_data["options"]

        #input_type = get_input_instance_type(input_config)
        create_labelarray( output_config["server"],
                           output_config["uuid"],
                           output_config["segmentation-name"],
                           3*(output_config["block-size"],) )

        # Copy the ROI from source to destination
        src_info = RoiInfo(input_config["server"], input_config["uuid"], roi_config["name"])
        dest_info = RoiInfo(output_config["server"], output_config["uuid"], roi_config["name"])
        copy_roi(src_info, dest_info)

        # (sv_id, sv)
        distsubvolumes = self.sparkdvid_input_context.parallelize_roi( roi_config["name"],
                                                                       options["chunk-size"],
                                                                       0,
                                                                       False, # Change to TRUE if stitching needed.
                                                                       roi_config["partition-method"],
                                                                       roi_config["partition-filter"] )


        # do not recompute ROI for each iteration
        distsubvolumes.persist()
        
        # (sv_id, data)
        seg_chunks = self.sparkdvid_input_context.map_labels64( distsubvolumes,
                                                                input_config['segmentation-name'],
                                                                0,
                                                                roi_config["name"] )


        # repartition to be z=blksize, y=blksize, x=runlength (x=0 is unlimited)
        partition_size = options["blockwritelimit"] * output_config["block-size"]
        partition_dims = PartitionDims(output_config["block-size"], output_config["block-size"], partition_size)
        schema = partitionSchema(partition_dims, padding=output_config["block-size"])
        
        # format segmentation chunks to be used with the partitionschema
        def combine_values( item ):
            (sv_id1, sv), (sv_id2, data) = item
            assert sv_id1 == sv_id2
            z = sv.box.z1
            y = sv.box.y1
            x = sv.box.x1
            return (volumePartition((z,y,x), VolumeOffset(z,y,x)), data)
        seg_chunks = distsubvolumes.zip(seg_chunks).map(combine_values)

        # RDD shuffling operation to get data into the correct partitions
        seg_chunks_partitioned = schema.partition_data(seg_chunks)
        
        # data must exist after writing to dvid for downsampling
        seg_chunks_partitioned.persist()

        # TODO: if labelarray already exists set pyramid depth from that

        # if no pyramid depth is specified, determine the max
        if options["pyramid-depth"] == 0:
            subvolumes = [sv for (_sid, sv) in distsubvolumes.collect()]
            sv_boxes = np.zeros( (len(subvolumes), 2, 3), np.int64 )
            sv_boxes[:,0] = [sv.box[:3] for sv in subvolumes]
            sv_boxes[:,1] = [sv.box[3:] for sv in subvolumes]
            
            global_start = sv_boxes.min(axis=0)
            global_stop  = sv_boxes.max(axis=0)
            global_shape = global_stop - global_start
            maxdim = global_shape.max()

            while maxdim > 512:
                options["pyramid-depth"] += 1
                maxdim /= 2

        # write level 0
        dataname = output_config["segmentation-name"]
        self._write_blocks(seg_chunks_partitioned, dataname, 0)

        # write pyramid levels for >=1 
        for level in range(1, options["pyramid-depth"] + 1):
            # downsample seg partition
            def downsample(part_vol):
                part, vol = part_vol
                vol = downsample_3Dlabels(vol, 1)[0]
                return (part, vol)
            downsampled_array = seg_chunks_partitioned.map(downsample)

            # prepare for repartition
            # (!!assume vol and offset will always be power of two because of padding)
            def repartition_down(part_volume):
                part, volume = part_volume
                downsampled_offset = np.array(part.get_offset()) / 2
                downsampled_reloffset = np.array(part.get_reloffset()) / 2
                offsetnew = VolumeOffset(*downsampled_offset)
                reloffsetnew = VolumeOffset(*downsampled_reloffset)
                partnew = volumePartition((offsetnew.z, offsetnew.y, offsetnew.x), offsetnew, reloffset=reloffsetnew)
                return partnew, volume
            downsampled_array = downsampled_array.map(repartition_down)
            
            # repartition downsampled data (unpersist previous level)
            schema = partitionSchema(partition_dims, padding=output_config["block-size"])
            seg_chunks_partitioned = schema.partition_data(downsampled_array)

            # persist for next level
            seg_chunks_partitioned.persist()
            
            # TODO: init levels when creating datatype or if already created, check how many levels are available
            # TEMPORARY HACK (NEED TO MODIFY LIBDVID)
            dataname = output_config["segmentation-name"] + "_" + str(level)
            create_labelarray( output_config["server"],
                               output_config["uuid"],
                               dataname,
                               3*(output_config["block-size"],) )

            #  write data new level
            self._write_blocks(seg_chunks_partitioned, dataname, level)

    def _write_blocks(self, partitions, dataname, level):
        """Writes partition to specified dvid.
        """
        output_config = self.config_data["data-info"]["output"]
        appname = self.APPNAME

        server = output_config["server"]
        uuid = output_config["uuid"]
        blksize = output_config["block-size"]
        
        resource_server = self.resource_server 
        resource_port = self.resource_port 
        
        # default delimiter
        delimiter = 0
    
        @self.collect_log(lambda (part, data): part.get_offset())
        def write_blocks(part_vol):
            logger = logging.getLogger(__name__)
            part, data = part_vol
            offset = part.get_offset()
            reloffset = part.get_reloffset()
            print "!!", offset, reloffset, level
            _, _, x_size = data.shape
            if x_size % blksize != 0:
                # check if padded
                raise ValueError("Data is not block aligned")

            shiftedoffset = (offset.z+reloffset.z, offset.y+reloffset.y, offset.x+reloffset.x)
            logger.info("Starting WRITE of partition at: {} size: {}".format(shiftedoffset, data.shape))
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port, appname)

            # Find all non-zero blocks (and record by block index)
            block_coords = []
            for block_index, block_x in enumerate(xrange(0, x_size, blksize)):
                if not (data[:, :, block_x:block_x+blksize] == delimiter).all():
                    block_coords.append( (0, 0, block_index) ) # (Don't care about Z,Y indexes, just X-index)

            # Find *runs* of non-zero blocks
            block_runs = runlength_encode(block_coords, True) # returns [[Z,Y,X1,X2], [Z,Y,X1,X2], ...]
            
            # Convert stop indexes from inclusive to exclusive
            block_runs[:,-1] += 1
            
            # Discard Z,Y indexes and convert from indexes to pixels
            ranges = blksize * block_runs[:, 2:4]
            
            # iterate through contiguous blocks and write to DVID
            for (data_x_start, data_x_end) in ranges:
                with Timer() as copy_timer:
                    datacrop = data[:,:,data_x_start:data_x_end].copy()
                logger.info("Copied {}:{} in {:.3f} seconds".format(data_x_start, data_x_end, copy_timer.seconds))

                data_offset_zyx = (shiftedoffset[0], shiftedoffset[1], shiftedoffset[2] + data_x_start)

                # TODO: modify labelblocks3D to take optional level information
                logger.info("STARTING Put: labels block {}".format(data_offset_zyx))
                throttle = (resource_server == "" and not server.startswith("http://127.0.0.1"))
                with Timer() as put_timer:
                    node_service.put_labelblocks3D( str(dataname), datacrop, data_offset_zyx, throttle )
                logger.info("Put block {} in {:.3f} seconds".format(data_offset_zyx, put_timer.seconds))

        partitions.foreach(write_blocks)
       
        
        
##
## FUNCTIONS BELOW THIS LINE ARE NOT USED (YET?)
##

#         def download_and_upload_chunk(subvolume):
#             seg_array = download_segmentation_chunk(input_config, options_config, subvolume)
#             upload_segmentation_chunk(output_config, options_config, subvolume, seg_array)
# 
#         distsubvolumes.values().map(download_and_upload_chunk).collect()

def download_segmentation_chunk( input_config, options_config, subvolume ):
    node_service = retrieve_node_service( input_config["server"], 
                                          input_config["uuid"],
                                          options_config["resource-server"],
                                          options_config["resource-port"],
                                          CopySegmentation.APPNAME )

    start_zyx = subvolume.box[:3]
    stop_zyx = subvolume.box[3:]
    shape_zyx = np.array(stop_zyx) - start_zyx
    
    # get_labels3D() happens to work for both labelblk AND googlevoxels.
    # (DVID/libdvid can't handle googlevoxels grayscale, but segmentation works.)
    return node_service.get_labels3D( input_config["segmentation-name"], shape_zyx, start_zyx )

def upload_segmentation_chunk( output_config, options_config, subvolume, seg_array ):
    node_service = retrieve_node_service( output_config["server"], 
                                          output_config["uuid"],
                                          options_config["resource-server"],
                                          options_config["resource-port"],
                                          CopySegmentation.APPNAME )

    start_zyx = subvolume.box[:3]
    stop_zyx = subvolume.box[3:]
    shape_zyx = np.array(stop_zyx) - start_zyx
    assert shape_zyx == seg_array.shape
    
    return node_service.put_labels3D( output_config["segmentation-name"], seg_array, start_zyx )


def get_input_instance_type(input_config):
    r = requests.get('{dvid-server}/api/node/{uuid}/{segmentation-name}/info'
                     .format(**input_config))
    r.raise_for_status()       

    info = r.json()
    typename = info["Base"]["TypeName"]
    assert typename in ("googlevoxels", "labelblk")
    return typename


if __name__ == "__main__":
    from DVIDSparkServices.json_util import validate_and_inject_defaults
    config = {
        "data-info": {
            "input": {
                "server": "127.0.0.1:8000",
                "uuid": "UUID1",
                "segmentation-name": "labels",
            },
            "output": {
                "server": "bergs-ws1:9000",
                "uuid": "UUID2",
                "segmentation-name": "labels",
            },
            "roi": {
                "name": "section-26"
            },
        },
        "options": {
            "corespertask": 1,
            "chunk-size": 512,
    
            "pyramid-depth": 0,
            "blockwritelimit": 0,
    
            "resource-port": 0,
            "resource-server": "",
    
            "log-collector-directory": "",
            "log-collector-port": 0,
    
            "debug": False,
        }
    }

    validate_and_inject_defaults(config, CopySegmentation.Schema)
    print json.dumps(config, indent=4, separators=(',', ': '))



