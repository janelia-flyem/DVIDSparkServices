"""Defines workflow to ingest 3D volumes and their denormalizations into DVID.

This file contains a top-level class that is callable via DVIDSparkServices
workflow interface.  It also contains a library for access as a standalone
library without requiring Apache Spark.
"""
import copy
import json
import logging

logger = logging.getLogger(__name__)

import numpy as np

from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

from DVIDSparkServices.io_util.partitionSchema import volumePartition, VolumeOffset, VolumeSize, PartitionDims, partitionSchema
from DVIDSparkServices.io_util.imagefileSrc import imagefileSrc
from DVIDSparkServices.io_util.dvidSrc import dvidSrc
from DVIDSparkServices.dvid.metadata import is_dvidversion, is_datainstance, dataInstance, set_sync, has_sync, get_blocksize, create_rawarray8, create_labelarray, Compression, extend_list_value, update_extents 
from DVIDSparkServices.reconutils.downsample import downsample_raw, downsample_3Dlabels
from DVIDSparkServices.util import Timer, runlength_encode, unicode_to_str

class Ingest3DVolumeDirect(object):
    """Called by Ingest3DVolume (see below for detailed documentation).

    This class implements logic for ingesting 3D volumes into DVID.
    Unlike the command line Spark workflow which uses 'Ingest3DVolume', this
    allows a user to specify data using a numpy array.
    """

    def __init__(self, data, dvidserver, uuid, dataname, options=None):
        """Init.

        Args:
            data (numpy array, RDD, or volumeSrc): Data to be loaded and downsample
            dvidserver (str): location of dvid server
            uuid (str): dvid version (must be open node)
            dataname (str): instance name to create
            options (json): Option configuration
        """
   
        # TODO finish implementation, numpy array or RDD is all in memory so
        # iterations are unnecessary.


class Ingest3DVolume(Workflow):
    """Ingest 3D volume and denormalizations into DVID.

    This class is called as a Spark workflow, in which, the
    JSON schema must be satisfied.  This is a wrapper that
    will call core pyramid generation that can be called from
    an RDD or numpy array. 

    The default options are set to the presumed most common use case.

    The ingestion supports the following typical workflows:

    * Load raw disk images into a DVID array
    * Load raw data into DVID array, compute/load octree
    * Load raw data into DVID array, compute/load octree,
    compute/load lossy octree
    * Load raw data into DVID array, compute/load 2D tiles 
    * Compute and load only octree into DVID without raw level 0
    * Call library with partitioned segmentation, compute/load
    segmentation and octree.

    Note:
        DVID padding currently will not work properly when doing
        multiscale pyramids.
        
        The following naming convention is followed.  If an output
        name of 'grayscale' is chosen.  Tiles will be written
        out as 'grayscale-tiles'.  Octree will be written as
        'grayscale-1', 'grayscale-2', etc.  Lossy JPEG compression
        will be indicated in the name, e.g.,'grayscalejpeg'
        and 'grayscale-tilesjpeg'.

        The number of parallel tasks is >=  2*blocksize.
        The user should specify more tasks (a multiple of 2*blocksize
        ideally) if running on a larger spark cluster.

        Performance can be tuned with the hidden parameters to
        set the resource-server and the corespertask.  For
        very large XY images, the user should potential
        limit the partition-width.

    TODO:
        * (important) When padding data with data already stored in DVID,
        padding should be disabled in the partitioner and dvidSRC should
        be modified to be able to pad (reusing the functionality in the partioner).
        Alternatively, one could always persist padded data from DVID to
        lower resolutions during pyramid creation.  This will lead to slightly
        non-optimal boundary effects as lossy data could be downsampled.
        * Separate ingestion class that can be called by other workflows
        * Allow image source and ROI from DVID
        * Enable DVID GET/PUTs that mimimize work on DVID (e.g.,
        compress data locally the same as on DVID)
        * Enable padding of tile data with pre-existing DVID data
        * Better ensure data is properly synced by maintaining an
        ingestion log in DVID and using sync metadata 
        * Support output (like tiles) to disk since tile write
        performance is generally poor for leveldb.
        * Create simple function to estimate memory usage
        * Launch the DVID resource manager in the driver program
    
    Current Limitations:
        * Only supports ingesting of 8bit raw data and 64 bit label data.
        * Assumes isotropic array chunk sizes
        * Downsampling (octree creation) is currently isotropic
        * Only supports XY tiles
        * Tiles not available for label arrays
    """

    # schema for ingesting grayscale
    DvidInfoSchema = \
    {
      "type": "object",
      "required": ["dvid-server", "uuid", "dataname"],
      "additionalProperties": False,
      "properties": {
        "dvid-server": {
          "description": "location of DVID server",
          "type": "string",
          "property": "dvid-server"
        },
        "uuid": {
          "description": "version node to store segmentation",
          "type": "string"
        },
        "dataname": {
          "description": "data instance to create (default uint8blk type) and prefix for denormalizations",
          "type": "string"
        }
      }
    }

    IngestionWorkflowOptionsSchema = copy.copy(Workflow.OptionsSchema)
    IngestionWorkflowOptionsSchema["required"] = ["minslice", "maxslice", "basename"]
    IngestionWorkflowOptionsSchema["additionalProperties"] = False
    IngestionWorkflowOptionsSchema["properties"].update(
    {
        #
        # REQUIRED
        #
        "minslice": {
            "description": "Minimum Z image slice",
            "type": "integer" 
        },
        "maxslice": { 
            "description": "Maximum Z image slice (inclusive)",
            "type": "integer" 
        },
        "basename": { 
            "description": "Path and name format for image files (images should be 8-bit grayscale)",
            "type": "string" 
        },
     
        #
        # OPTIONAL
        #
        "create-pyramid": {
            "description": "create a multi-scale octree",
            "type": "boolean",
            "default": False
        },
        "create-pyramid-jpeg": {
            "description": "create lossy multi-scale octree (uint8blk only)",
            "type": "boolean",
            "default": True
        },
        "create-tiles": {
            "description": "create 2D tiles (uint8blk only)",
            "type": "boolean",
            "default": False
        },
        "create-tiles-jpeg": {
            "description": "create lossy 2D tiles (uint8blk only)",
            "type": "boolean",
            "default": False
        },
        "blocksize": {
          "description": "Internal block size (default: 64x64x64)",
          "type": "integer",
          "default": 64
        },
        "tilesize": {
          "description": "Tile size (default: 1024x1024)",
          "type": "integer",
          "default": 1024
        },
        "offset": {
          "description": "Offset (x,y,z) for loading data (z will be added to the minslice specified)",
          "type": "array",
          "items": [
            { 
              "type": "integer",
              "default": 0
            },
            { 
              "type": "integer",
              "default": 0
            },
            { 
              "type": "integer",
              "default": 0
            }
          ]
        },
        "blockwritelimit": {
           "description": "Maximum number of blocks written per task request (0=no limit)",
           "type": "integer",
           "default": 100
        },
        "has-dvidmask": {
            "description": "Enables padding of data from DVID (unless instance does not exist)",
            "type": "boolean",
            "default": True
        },
        "disable-original": {
            "description": "Do not write original data to DVID",
            "type": "boolean",
            "default": False
        },
        "blankdelimiter": {
          "description": "Delimiting value for a blank data",
          "type": "integer",
          "default": 0 
        },
        "is-rawarray": {
            "description": "Treat data as uint8blk",
            "type": "boolean",
            "default": True
        },
        "pyramid-depth": {
            "description": "Number of pyramid levels to generate (0 means choose automatically)",
            "type": "integer",
            "default": 0
        },
        "num-tasks": {
            "description": "Number of tasks to use (min is 2*blocksize and should be multiple of this)",
            "type": "integer",
            "default": 128
        }
    })

    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to load raw and label data into DVID",
      "type": "object",
      "required": ["dvid-info", "options"],
      "additionalProperties": False,
      "properties": {
        "dvid-info": DvidInfoSchema,
        "options" : IngestionWorkflowOptionsSchema
      }
    }

    # name of application for DVID queries
    APPNAME = "ingest3dvolume"
       
    # constant extensions for denormalization naming
    JPEGPYRAMID_NAME = "jpeg"
    TILENAME = "-tiles"
    JPEGTILENAME = "-tilesjpeg"

    def __init__(self, config_filename):
        """Init.

        Calls default init and sets option variables
        """
        super(Ingest3DVolume, self).__init__(config_filename, Ingest3DVolume.dumpschema(), "Ingest 3D Volume")

        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]

        if not dvid_info["dvid-server"].startswith("http://"):
            dvid_info["dvid-server"] = "http://" + dvid_info["dvid-server"]

        if not options["is-rawarray"]:
            assert not options["create-tiles"], "Bad config: Can't create tiles for label data."
            assert not options["create-tiles-jpeg"], "Bad config: Can't create tiles for label data."
            assert not options["create-pyramid-jpeg"], "Bad config: Can't create tiles for label data."

        if not options["create-pyramid-jpeg"] and not options["create-pyramid"]:
            assert options["pyramid-depth"] == 0, \
                "Bad config: Pyramid depth specified, but no 'create-pyramid' setting given."

        self.mintasks = options["blocksize"] * 2
        if options["num-tasks"] > self.mintasks:
            # fetch data in multiples of mintasks
            self.mintasks = (options["num-tasks"]/self.mintasks) * self.mintasks

        self.partition_size = options["blockwritelimit"] * options["blocksize"]

    def _writeimagepyramid(self, tilepartition):
        """Write image tiles to DVID.

        Note:
            This function makes a series of small tile writes.  One
            should consider multi-threaded requests as in 'CreateTiles'
            in situations with poor server latency.
        """
        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]

        maxlevel = self.maxlevel
        tilesize = options["tilesize"]
        delimiter = options["blankdelimiter"]
        createtiles = options["create-tiles"]
        createtilesjpeg = options["create-tiles-jpeg"]
        server = dvid_info["dvid-server"]
        tilename = dvid_info["dataname"]+self.TILENAME
        tilenamejpeg = dvid_info["dataname"]+self.JPEGTILENAME
        uuid = dvid_info["uuid"]

        @self.collect_log(lambda (part, vol): part.get_offset())
        def writeimagepyramid(part_data):
            logger = logging.getLogger(__name__)
            part, vol = part_data
            offset = part.get_offset() 
            zslice = offset.z
            from PIL import Image
            from scipy import ndimage
            import StringIO
            import requests
            s = requests.Session()
    
            # pad data with delimiter if needed
            timslice = vol[0, :, :]
            shiftx = offset.x % tilesize
            shifty = offset.y % tilesize
            tysize, txsize = timslice.shape
            ysize = tysize + shifty
            xsize = txsize + shiftx
            imslice = np.zeros((ysize, xsize))
            imslice[:,:] = delimiter
            imslice[shifty:ysize, shiftx:xsize] = timslice
            curry = (offset.y - shifty)/2 
            currx = (offset.x - shiftx)/2

            imlevels = []
            tileoffsetyx = []
            imlevels.append(imslice)
            tileoffsetyx.append((offset.y/tilesize, offset.x/tilesize))  

            with Timer() as downsample_timer:
                # use generic downsample algorithm
                for level in range(1, maxlevel+1):
                    
                    tysize, txsize = imlevels[level-1].shape
    
                    shiftx = currx % tilesize
                    shifty = curry % tilesize
                    
                    ysize = tysize + shifty
                    xsize = txsize + shiftx
                    imslice = np.zeros((ysize, xsize))
                    imslice[:,:] = delimiter
                    timslice = ndimage.interpolation.zoom(imlevels[level-1], 0.5)
                    imslice[shifty:ysize, shiftx:xsize] = timslice
                    imlevels.append(imslice) 
                    tileoffsetyx.append((currx/tilesize, curry/tilesize))  
                    
                    curry = (curry - shifty)/2 
                    currx = (currx - shiftx)/2

            logger.info("Downsampled {} levels in {:.3f} seconds".format(maxlevel, downsample_timer.seconds))

            # write tile pyramid using custom requests
            for levelnum in range(0, len(imlevels)):
                levelslice = imlevels[levelnum]
                dim1, dim2 = levelslice.shape

                num1tiles = (dim1-1)/tilesize + 1
                num2tiles = (dim2-1)/tilesize + 1

                with Timer() as post_timer:
                    for iter1 in range(0, num1tiles):
                        for iter2 in range(0, num2tiles):
                            # extract tile
                            tileholder = np.zeros((tilesize, tilesize), np.uint8)
                            tileholder[:,:] = delimiter
                            min1 = iter1*tilesize
                            min2 = iter2*tilesize
                            tileslice = levelslice[min1:min1+tilesize, min2:min2+tilesize]
                            t1, t2 = tileslice.shape
                            tileholder[0:t1, 0:t2] = tileslice
    
                            starty, startx = tileoffsetyx[levelnum]
                            starty += iter1
                            startx += iter2
                            if createtiles:
                                buf = StringIO.StringIO() 
                                img = Image.frombuffer('L', (tilesize, tilesize), tileholder.tostring(), 'raw', 'L', 0, 1)
                                img.save(buf, format="png")
    
                                urlreq = server + "/api/node/" + uuid + "/" + tilename + "/tile/xy/" + str(levelnum) + "/" + str(startx) + "_" + str(starty) + "_" + str(zslice)
                                s.post(urlreq , data=buf.getvalue())
                                buf.close()
                            
                            if createtilesjpeg:
                                buf = StringIO.StringIO() 
                                img = Image.frombuffer('L', (tilesize, tilesize), tileholder.tostring(), 'raw', 'L', 0, 1)
                                img.save(buf, format="jpeg")
    
                                urlreq = server + "/api/node/" + uuid + "/" + tilenamejpeg + "/tile/xy/" + str(levelnum) + "/" + str(startx) + "_" + str(starty) + "_" + str(zslice)
                                s.post(urlreq , data=buf.getvalue())
                                buf.close()
                logger.info("Posted {} tiles (level={}) in {} seconds".format( num1tiles*num2tiles, levelnum, post_timer.seconds ) )

        tilepartition.foreach(writeimagepyramid)

    def _write_blocks(self, partitions, dataname, dataname_lossy):
        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]
        appname = self.APPNAME

        server = dvid_info["dvid-server"]
        uuid = dvid_info["uuid"]
        resource_server = self.resource_server
        resource_port = self.resource_port
        blksize = options["blocksize"]
        delimiter = options["blankdelimiter"]
        israw = options["is-rawarray"]

        @self.collect_log(lambda (part, data): part.get_offset())
        def write_blocks(part_vol):
            logger = logging.getLogger(__name__)
            part, data = part_vol
            offset = part.get_offset()
            _, _, x_size = data.shape
            if x_size % blksize != 0:
                # check if padded
                raise ValueError("Data is not block aligned")

            logger.info("Starting WRITE of partition at: {} size: {}".format(offset, data.shape))
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port, appname)

            # Find all non-zero blocks (and record by block index)
            block_coords = []
            for block_index, block_x in enumerate(range(0, x_size, blksize)):
                if not (data[:, :, block_x:block_x+blksize] == delimiter).all():
                    block_coords.append( (0, 0, block_index) ) # (Don't care about Z,Y indexes, just X-index)

            # Find *runs* of non-zero blocks
            block_runs = runlength_encode(block_coords, True) # returns [[Z,Y,X1,X2], [Z,Y,X1,X2], ...]
            
            # Convert stop indexes from inclusive to exclusive
            block_runs[:,-1] += 1
            
            # Discard Z,Y indexes and convert from indexes to pixels
            ranges = blksize * block_runs[:, 2:4]
            
            ranges[:] += offset.x
            
            # iterate through contiguous blocks and write to DVID
            # TODO: write compressed data directly into DVID
            for (offsetx, endx) in ranges:
                with Timer() as copy_timer:
                    datacrop = data[:,:,offsetx:endx].copy()
                logger.info("Copied {}:{} in {:.3f} seconds".format(offsetx, endx, copy_timer.seconds))

                data_offset_zyx = (offset.z, offset.y, offsetx)

                if dataname is not None:
                    with Timer() as put_timer:
                        if not israw: 
                            logger.info("STARTING Put: labels block {}".format())
                            if resource_server != "":
                                node_service.put_labels3D(dataname, datacrop, data_offset_zyx, compress=True, throttle=False)
                            else:
                                node_service.put_labels3D(dataname, datacrop, data_offset_zyx, compress=True)
                        else:
                            logger.info("STARTING Put: raw block {}".format(data_offset_zyx))
                            if resource_server != "":
                                node_service.put_gray3D(dataname, datacrop, data_offset_zyx, compress=False, throttle=False)
                            else:
                                node_service.put_gray3D(dataname, datacrop, data_offset_zyx, compress=False)
                    logger.info("Put block {} in {:.3f} seconds".format(data_offset_zyx, put_timer.seconds))

                if dataname_lossy is not None:
                    logger.info("STARTING Put: lossy block {}".format(data_offset_zyx))
                    with Timer() as put_lossy_timer:
                        if resource_server != "":
                            node_service.put_gray3D(dataname_lossy, datacrop, data_offset_zyx, compress=False, throttle=False)
                        else:
                            node_service.put_gray3D(dataname_lossy, datacrop, data_offset_zyx, compress=False)
                    logger.info("Put lossy block {} in {:.3f} seconds".format(data_offset_zyx, put_lossy_timer.seconds))

        partitions.foreach(write_blocks)


    def execute(self):
        """Execute spark workflow.
        """
        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]
        block_shape = 3*(options["blocksize"],)
        # ?? num parallel requests might be really small at high levels of pyramids

        # xdim is unbounded or very large
        partition_dims = PartitionDims(options["blocksize"], options["blocksize"], self.partition_size)
        partition_schema = partitionSchema( partition_dims,
                                            blank_delimiter=options["blankdelimiter"],
                                            padding=options["blocksize"],
                                            enablemask=options["has-dvidmask"])

        offset_zyx = np.array( options["offset"][::-1] )
        offset_zyx[0] += options["minslice"]
        imgreader = imagefileSrc( partition_schema,
                                  options["basename"],
                                  (options["minslice"], options["maxslice"]),
                                  VolumeOffset(*offset_zyx),
                                  self.sc )
       
        # !! hack: override iteration size that is set to partition size, TODO: add option
        # this just makes the downstream processing a little more convenient, and reduces
        # unnecessary DVID patching if that is enabled.
        # (must be a multiple of block size)
        imgreader.iteration_size = self.mintasks

        # get dims from image (hackage)
        from PIL import Image
        import requests
        img = Image.open(options["basename"] % options["minslice"]) 
        volume_shape = (1 + options["maxslice"] - options["minslice"], img.height, img.width)
        del img

        global_box_zyx = np.zeros((2,3), dtype=int)
        global_box_zyx[0] = options["offset"]
        global_box_zyx[0] += (options["minslice"], 0, 0)

        global_box_zyx[1] = global_box_zyx[0] + volume_shape

        if is_datainstance( dvid_info["dvid-server"], dvid_info["uuid"], dvid_info["dataname"] ):
            logger.info("'{dataname}' already exists, skipping creation".format(**dvid_info) )
        else:
            # create data instance and disable dvidmask
            # !! assume if data instance exists and mask is set that all pyramid
            # !! also exits, meaning the mask should be used. 
            options["has-dvidmask"] = False
            if options["disable-original"]:
                logger.info("Not creating '{dataname}' due to 'disable-original' config setting".format(**dvid_info) )
            else:
                if options["is-rawarray"]:
                    create_rawarray8( dvid_info["dvid-server"],
                                      dvid_info["uuid"],
                                      dvid_info["dataname"],
                                      block_shape )
                else:
                    create_labelarray( dvid_info["dvid-server"],
                                       dvid_info["uuid"],
                                       dvid_info["dataname"],
                                       block_shape )

        if not options["disable-original"]:
            update_extents( dvid_info["dvid-server"],
                            dvid_info["uuid"],
                            dvid_info["dataname"],
                            global_box_zyx )

            # Bottom level of pyramid is listed as neuroglancer-compatible
            extend_list_value(dvid_info["dvid-server"], dvid_info["uuid"], '.meta', 'neuroglancer', [dvid_info["dataname"]])

        # determine number of pyramid levels if not specified 
        if options["create-pyramid"] or options["create-pyramid-jpeg"]:
            if options["pyramid-depth"] == 0:
                zsize = options["maxslice"] - options["minslice"] + 1
                while zsize > 512:
                    options["pyramid-depth"] += 1
                    zsize /= 2

        # create pyramid data instances
        if options["create-pyramid-jpeg"]:
            dataname_jpeg = dvid_info["dataname"] + self.JPEGPYRAMID_NAME 
            if is_datainstance(dvid_info["dvid-server"], dvid_info["uuid"], dataname_jpeg):
                logger.info("'{}' already exists, skipping creation".format(dataname_jpeg) )
            else:
                create_rawarray8( dvid_info["dvid-server"],
                                  dvid_info["uuid"],
                                  dataname_jpeg,
                                  block_shape,
                                  Compression.JPEG )

            update_extents( dvid_info["dvid-server"],
                            dvid_info["uuid"],
                            dataname_jpeg,
                            global_box_zyx )

            # Bottom level of pyramid is listed as neuroglancer-compatible
            extend_list_value(dvid_info["dvid-server"], dvid_info["uuid"], '.meta', 'neuroglancer', [dataname_jpeg])

    
        if options["create-pyramid"]:
            for level in range(1, 1 + options["pyramid-depth"]):
                downsampled_box_zyx = global_box_zyx / (2**level)
                downname = dvid_info["dataname"] + "_%d" % level

                if is_datainstance(dvid_info["dvid-server"], dvid_info["uuid"], downname):
                    logger.info("'{}' already exists, skipping creation".format(downname) )
                else:
                    if options["is-rawarray"]:
                        create_rawarray8( dvid_info["dvid-server"],
                                          dvid_info["uuid"],
                                          downname,
                                          block_shape )
                    else:
                        create_labelarray( dvid_info["dvid-server"],
                                           dvid_info["uuid"],
                                           downname,
                                           block_shape )

                update_extents( dvid_info["dvid-server"],
                                dvid_info["uuid"],
                                downname,
                                downsampled_box_zyx )

                # Higher-levels of the pyramid should not appear in the DVID-lite console.
                extend_list_value(dvid_info["dvid-server"], dvid_info["uuid"], '.meta', 'restrictions', [downname])

        if options["create-pyramid-jpeg"]: 
            for level in range(1, 1 + options["pyramid-depth"]):
                downsampled_box_zyx = global_box_zyx / (2**level)
                downname = dvid_info["dataname"] + self.JPEGPYRAMID_NAME + "_%d" % level
                if is_datainstance(dvid_info["dvid-server"], dvid_info["uuid"], downname):
                    logger.info("'{}' already exists, skipping creation".format(downname) )
                else:
                    create_rawarray8( dvid_info["dvid-server"],
                                      dvid_info["uuid"],
                                      downname,
                                      block_shape,
                                      Compression.JPEG )

                update_extents( dvid_info["dvid-server"],
                                dvid_info["uuid"],
                                downname,
                                downsampled_box_zyx )

                # Higher-levels of the pyramid should not appear in the DVID-lite console.
                extend_list_value(dvid_info["dvid-server"], dvid_info["uuid"], '.meta', 'restrictions', [downname])
            
        # create tiles
        if options["create-tiles"] or options["create-tiles-jpeg"]:
            MinTileCoord = global_box_zyx[0][::-1] / options["tilesize"]
            MaxTileCoord = global_box_zyx[1][::-1] / options["tilesize"]
            
            # get max level by just finding max tile coord
            maxval = max(MaxTileCoord) - min(MinTileCoord) + 1
            import math
            self.maxlevel = int(math.log(maxval)/math.log(2))

            tilemeta = {}
            tilemeta["MinTileCoord"] = MinTileCoord.tolist()
            tilemeta["MaxTileCoord"] = MaxTileCoord.tolist()
            tilemeta["Levels"] = {}

            currres = 8.0 # just use as placeholder for now
            for level in range(0, self.maxlevel+1):
                tilemeta["Levels"][str(level)] = { "Resolution" : 3*[currres],
                                                   "TileSize": 3*[options["tilesize"]] }
                currres *= 2

            if options["create-tiles"]:
                requests.post("{dvid-server}/api/repo/{uuid}/instance".format(**dvid_info),
                              json={"typename": "imagetile",
                                    "dataname": dvid_info["dataname"]+self.TILENAME,
                                    "source": dvid_info["dataname"],
                                    "format": "png"})
                requests.post("{dvid-server}/api/repo/{uuid}/{dataname}{tilename}/metadata".format(tilename=self.TILENAME, **dvid_info), json=tilemeta)

            if options["create-tiles-jpeg"]:
                requests.post("{dvid-server}/api/repo/{uuid}/instance".format(**dvid_info),
                              json={ "typename": "imagetile",
                                     "dataname": dvid_info["dataname"]+self.JPEGTILENAME,
                                     "source": dvid_info["dataname"],
                                     "format": "jpg"} )
                requests.post("{dvid-server}/api/repo/{uuid}/{dataname_jpeg_tile}/metadata"
                              .format( dataname_jpeg_tile=dvid_info["dataname"]+self.JPEGTILENAME, **dvid_info ),
                              json=tilemeta)

        # TODO Validation: should verify syncs exist, should verify pyramid depth 

        # TODO: set syncs for pyramids, tiles if base datatype exists
        # syncs should be removed before ingestion and added afterward

        levels_cache = {}

        # iterate through each partition
        for arraypartition in imgreader:
            # DVID pad if necessary
            if options["has-dvidmask"]:
                dvidsrc = dvidSrc( dvid_info["dvid-server"],
                                   dvid_info["uuid"],
                                   dvid_info["dataname"],
                                   arraypartition,
                                   resource_server=self.resource_server,
                                   resource_port=self. resource_port)

                arraypartition = dvidsrc.extract_volume()

            # potentially need for future iterations
            arraypartition.persist()

            # check for final layer
            finallayer = imgreader.curr_slice > imgreader.end_slice

            if not options["disable-original"]:
                # for each statement to disk (write jpeg at same time)
                dataname = None
                datanamelossy = None
                if options["create-pyramid-jpeg"]:
                    datanamelossy = dvid_info["dataname"] + self.JPEGPYRAMID_NAME
                if options["create-pyramid"]:
                    dataname = dvid_info["dataname"]
                self._write_blocks(arraypartition, dataname, datanamelossy) 

            if options["create-tiles"] or options["create-tiles-jpeg"]:
                # repartition into tiles
                schema = partitionSchema(PartitionDims(1,0,0))
                tilepartition = schema.partition_data(arraypartition)
               
                # write unpadded tilesize (will pad with delimiter if needed)
                self._writeimagepyramid(tilepartition)

            if options["create-pyramid"] or options["create-pyramid-jpeg"]:
                if 0 not in levels_cache:
                    levels_cache[0] = []
                levels_cache[0].append(arraypartition) 
                curr_level = 1
                downsample_factor = 2

                # should be a multiple of Z blocks or the final fetch
                assert imgreader.curr_slice % options["blocksize"] == 0
                while ((((imgreader.curr_slice / options["blocksize"]) % downsample_factor) == 0) or finallayer) and curr_level <= options["pyramid-depth"]:
                    partlist = levels_cache[curr_level-1]
                    part = partlist[0]
                    # union all RDDs from the same level
                    for iter1 in range(1, len(partlist)):
                        part = part.union(partlist[iter1])
                    
                    # downsample map
                    israw = options["is-rawarray"]
                    def downsample(part_vol):
                        part, vol = part_vol
                        if not israw:
                            vol = downsample_3Dlabels(vol)[0]
                        else:
                            vol = downsample_raw(vol)[0]
                        return (part, vol)
                    downsampled_array = part.map(downsample)
        
                    # repart (vol and offset will always be power of two because of padding)
                    def repartition_down(part_volume):
                        part, volume = part_volume
                        downsampled_offset = np.array(part.get_offset()) / 2
                        offsetnew = VolumeOffset(*downsampled_offset)
                        partnew = volumePartition((offsetnew.z, offsetnew.y, offsetnew.x), offsetnew)
                        return partnew, volume

                    downsampled_array = downsampled_array.map(repartition_down)
                    
                    # repartition downsample data
                    partition_dims = PartitionDims(options["blocksize"], options["blocksize"], self.partition_size)
                    schema = partitionSchema( partition_dims,
                                              blank_delimiter=options["blankdelimiter"],
                                              padding=options["blocksize"],
                                              enablemask=options["has-dvidmask"] ) 
                    downsampled_array = schema.partition_data(downsampled_array)

                    # persist before padding if there are more levels
                    if curr_level < options["pyramid-depth"]:
                        downsampled_array.persist()
                        if curr_level not in levels_cache:
                            levels_cache[curr_level] = []
                        levels_cache[curr_level].append(downsampled_array)

                    # pad from DVID (move before persist will allow multi-ingest
                    # but will lead to slightly non-optimal downsampling boundary
                    # effects if using a lossy compression only.
                    if options["has-dvidmask"]:
                        padname = dvid_info["dataname"]
                        if options["create-pyramid-jpeg"]: # !! should pad with orig if computing
                            # pad with jpeg
                            padname += self.JPEGPYRAMID_NAME 
                        padname += "_%d" % curr_level
                        dvidsrc = dvidSrc( dvid_info["dvid-server"],
                                           dvid_info["uuid"],
                                           padname,
                                           downsampled_array,
                                           resource_server=self.resource_server,
                                           resource_port=self.resource_port )

                        downsampled_array = dvidsrc.extract_volume()

                    # write result
                    downname = None
                    downnamelossy = None
                    if options["create-pyramid"]:
                        downname = dvid_info["dataname"] + "_%d" % curr_level 
                    if options["create-pyramid-jpeg"]:
                        downnamelossy = dvid_info["dataname"] + self.JPEGPYRAMID_NAME + "_%d" % curr_level 
                    self._write_blocks(downsampled_array, downname, downnamelossy) 

                    # remove previous level
                    del levels_cache[curr_level-1]
                    curr_level += 1
                    downsample_factor *= 2

        
    @staticmethod
    def dumpschema():
        return json.dumps(Ingest3DVolume.Schema)
