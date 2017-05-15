"""Defines workflow to ingest 3D volumes and their denormalizations into DVID.

This file contains a top-level class that is callable via DVIDSparkServices
workflow interface.  It also contains a library for access as a standalone
library without requiring Apache Spark.
"""
import copy
import json

from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

from DVIDSparkServices.io_util.partitionSchema import volumePartition, VolumeOffset, VolumeSize, PartitionDims, partitionSchema
from DVIDSparkServices.io_util.imagefileSrc import imagefileSrc
from DVIDSparkServices.io_util.dvidSrc import dvidSrc
from DVIDSparkServices.dvid.metadata import is_dvidversion, is_datainstance, dataInstance, set_sync, has_sync, get_blocksize, create_rawarray8, create_labelarray, Compression 
from DVIDSparkServices.reconutils.downsample import downsample_raw, downsample_3Dlabels
from DVIDSparkServices.util import Timer
import numpy
import logging

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

        # primary input/output parameters
        self.minslice = self.config_data["options"]["minslice"]
        self.maxslice = self.config_data["options"]["maxslice"]
        self.basename = str(self.config_data["options"]["basename"]) 
        self.dvidserver = str(self.config_data["dvid-info"]["dvid-server"]) 
        self.uuid = str(self.config_data["dvid-info"]["uuid"])
        self.dataname = str(self.config_data["dvid-info"]["dataname"]) 

        # options
        self.createpyramid = self.config_data["options"]["create-pyramid"]
        self.createpyramidjpeg = self.config_data["options"]["create-pyramid-jpeg"]
        self.createtiles = self.config_data["options"]["create-tiles"]
        self.createtilesjpeg = self.config_data["options"]["create-tiles-jpeg"]
        self.blksize = self.config_data["options"]["blocksize"]
        self.tilesize = self.config_data["options"]["tilesize"]
        self.offset = self.config_data["options"]["offset"]
        self.partition_size = self.config_data["options"]["blockwritelimit"] * self.blksize
        self.use_dvidmask = self.config_data["options"]["has-dvidmask"]
        self.disable_original = self.config_data["options"]["disable-original"]
        self.delimiter = self.config_data["options"]["blankdelimiter"]
        self.israw = self.config_data["options"]["is-rawarray"]
        self.pyramid_depth = self.config_data["options"]["pyramid-depth"]
        self.num_tasks = self.config_data["options"]["num-tasks"]
        self.resource_server = str(self.resource_server)
        self.resource_port = self.resource_port

        # disable options if not raw
        if not self.israw:
            self.createtiles = False
            self.createtilesjpeg = False
            self.createpyramidjpeg = False

        if not self.createpyramidjpeg and not self.createpyramid:
            self.pyramid_depth = 0

        # create image source object
        self.mintasks = self.blksize * 2
        if self.num_tasks > self.mintasks:
            # fetch data in multiples of mintasks
            self.mintasks = (self.num_tasks/self.mintasks) * self.mintasks


    def _writeimagepyramid(self, tilepartition):
        """Write image tiles to DVID.

        Note:
            This function makes a series of small tile writes.  One
            should consider multi-threaded requests as in 'CreateTiles'
            in situations with poor server latency.
        """

        maxlevel = self.maxlevel
        tilesize = self.tilesize
        delimiter = self.delimiter
        createtiles = self.createtiles
        createtilesjpeg = self.createtilesjpeg
        server = self.dvidserver
        tilename = self.dataname+self.TILENAME
        tilenamejpeg = self.dataname+self.JPEGTILENAME
        uuid = self.uuid

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
            imslice = numpy.zeros((ysize, xsize))
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
                    imslice = numpy.zeros((ysize, xsize))
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
                            tileholder = numpy.zeros((tilesize, tilesize), numpy.uint8)
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
        appname = self.APPNAME
        server = self.dvidserver
        uuid = self.uuid
        resource_server = self.resource_server
        resource_port = self.resource_port
        blksize = self.blksize
        delimiter = self.delimiter
        israw = self.israw

        @self.collect_log(lambda (part, data): part.get_offset())
        def write_blocks(part_vol):
            logger = logging.getLogger(__name__)
            part, data = part_vol
            offset = part.get_offset()
            z,y,x = data.shape
            if x % blksize != 0:
                # check if padded
                raise ValueError("Data is not block aligned")

            logger.info("Starting WRITE of partition at: {} size: {}".format(offset, data.shape))
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port, appname)

            # find all contiguous ranges (do not write 0s)
            started = False
            ranges = []
            for blockiter in range(0, x, blksize):
                datablk = data[:,:,blockiter:blockiter+blksize]
                vals = numpy.unique(datablk)
                if len(vals) == 1 and vals[0] == delimiter:
                    if started:
                        started = False
                        ranges.append((startx, blockiter))
                else:
                    if not started:
                        startx = blockiter
                        started = True
            if started:
                ranges.append((startx, x))

            # iterate through contiguous blocks and write to DVID
            # TODO: write compressed data directly into DVID
            for (offsetx, endx) in ranges:
                with Timer() as copy_timer:
                    datacrop = data[:,:,offsetx:endx].copy()
                logger.info("Copied {}:{} in {:.3f} seconds".format(offsetx, endx, copy_timer.seconds))

                if dataname is not None:
                    with Timer() as put_timer:
                        if not israw: 
                            logger.info("STARTING Put: labels block {}".format((offset.z, offset.y, offsetx)))
                            if resource_server != "":
                                node_service.put_labels3D(dataname, datacrop, (offset.z, offset.y, offsetx), compress=True, throttle=False)
                            else:
                                node_service.put_labels3D(dataname, datacrop, (offset.z, offset.y, offsetx), compress=True)
                        else:
                            logger.info("STARTING Put: raw block {}".format((offset.z, offset.y, offsetx)))
                            if resource_server != "":
                                node_service.put_gray3D(dataname, datacrop, (offset.z, offset.y, offsetx), compress=False, throttle=False)
                            else:
                                node_service.put_gray3D(dataname, datacrop, (offset.z, offset.y, offsetx), compress=False)
                    logger.info("Put block {} in {:.3f} seconds".format((offset.z, offset.y, offsetx), put_timer.seconds))

                if dataname_lossy is not None:
                    logger.info("STARTING Put: lossy block {}".format((offset.z, offset.y, offsetx)))
                    with Timer() as put_lossy_timer:
                        if resource_server != "":
                            node_service.put_gray3D(dataname_lossy, datacrop, (offset.z, offset.y, offsetx), compress=False, throttle=False)
                        else:
                            node_service.put_gray3D(dataname_lossy, datacrop, (offset.z, offset.y, offsetx), compress=False)
                    logger.info("Put lossy block {} in {:.3f} seconds".format((offset.z, offset.y, offsetx), put_lossy_timer.seconds))

        partitions.foreach(write_blocks)


    def execute(self):
        """Execute spark workflow.
        """
        # ?? num parallel requests might be really small at high levels of pyramids

        # xdim is unbounded or very large
        schema = partitionSchema(PartitionDims(self.blksize, self.blksize, self.partition_size), blank_delimiter=self.delimiter, padding=self.blksize, enablemask=self.use_dvidmask) 
        imgreader = imagefileSrc(schema, self.basename, minmaxplane=(self.minslice, self.maxslice),
                offset=VolumeOffset(self.offset[2]+self.minslice, self.offset[1],
                    self.offset[0]), spark_context=self.sc)
       
        # !! hack: override iteration size that is set to partition size, TODO: add option
        # this just makes the downstream processing a little more convenient, and reduces
        # unnecessary DVID patching if that is enabled.
        # (must be a multiple of block size)
        imgreader.iteration_size = self.mintasks

        # no syncs necessary if base datatype is not needed and does not exist
        hassyncs = True
        if not is_datainstance(self.dvidserver, self.uuid, self.dataname):
            # create data instance and disable dvidmask
            # !! assume if data instance exists and mask is set that all pyramid
            # !! also exits, meaning the mask should be used. 
            self.use_dvidmask = False
            if not self.disable_original:
                if self.israw:
                    create_rawarray8(self.dvidserver, self.uuid, self.dataname,
                            blocksize=(self.blksize,self.blksize,self.blksize))
                else:
                    create_labelarray(self.dvidserver, self.uuid, self.dataname,
                            blocksize=(self.blksize,self.blksize,self.blksize))
            else:
                hassyncs = False

        # determine number of pyramid levels if not specified 
        if self.createpyramid or self.createpyramidjpeg:
            if self.pyramid_depth == 0:
                zsize = self.maxslice - self.minslice + 1
                while zsize > 512:
                    self.pyramid_depth += 1
                    zsize /= 2

        # create pyramid data instances
        if self.createpyramidjpeg:
            downname = self.dataname + self.JPEGPYRAMID_NAME 
            if not is_datainstance(self.dvidserver, self.uuid, downname):
                create_rawarray8(self.dvidserver, self.uuid, downname,
                        blocksize=(self.blksize,self.blksize,self.blksize),
                        compression=Compression.JPEG)
    
        for level in range(1, self.pyramid_depth+1):
            if self.createpyramid: 
                downname = self.dataname 
                downname += "_%d" % level
                if not is_datainstance(self.dvidserver, self.uuid, downname):
                    if self.israw:
                        create_rawarray8(self.dvidserver, self.uuid,
                                downname, blocksize=(self.blksize,
                                    self.blksize,self.blksize))
                    else:
                        create_labelarray(self.dvidserver, self.uuid,
                                self.downname, blocksize=(self.blksize,
                                    self.blksize,self.blksize))
            if self.createpyramidjpeg: 
                downname = self.dataname + self.JPEGPYRAMID_NAME 
                downname += "_%d" % level
                if not is_datainstance(self.dvidserver, self.uuid, downname):
                    create_rawarray8(self.dvidserver, self.uuid, downname,
                            blocksize=(self.blksize,self.blksize,self.blksize),
                            compression=Compression.JPEG)
            
        # create tiles
        if self.createtiles or self.createtilesjpeg:
            # get dims from image (hackage)
            from PIL import Image
            import requests
            img = Image.open(self.basename % self.minslice) 
            
            xmin, ymin, zmin = self.offset
            zmin += self.minslice
            
            xmax, ymax, zmax = img.width, img.height, self.maxslice-self.minslice+1
            xmax += xmin
            ymax += ymin
            zmax += zmin

            MinTileCoord = [xmin/self.tilesize, ymin/self.tilesize, zmin/self.tilesize]
            MaxTileCoord = [xmax/self.tilesize, ymax/self.tilesize, zmax/self.tilesize]
            
            # get max level by just finding max tile coord
            maxval = max(MaxTileCoord) - min(MinTileCoord) + 1
            import math
            self.maxlevel = int(math.log(maxval)/math.log(2))

            tilemeta = {}
            tilemeta["MinTileCoord"] = MinTileCoord
            tilemeta["MaxTileCoord"] = MaxTileCoord
            tilemeta["Levels"] = {}
            currres = 8.0 # just use as placeholder for now
            for level in range(0, self.maxlevel+1):
                tilemeta["Levels"][str(level)] = { "Resolution" : [currres, currres, currres],
                                                   "TileSize": [self.tilesize, self.tilesize, self.tilesize]}
                currres *= 2

            if not self.dvidserver.startswith("http://"):
                self.dvidserver = "http://" + self.dvidserver

            if self.createtiles:
                requests.post(self.dvidserver + "/api/repo/" + self.uuid + "/instance", json={"typename": "imagetile", "dataname": self.dataname+self.TILENAME, "source": self.dataname, "format": "png"})
                requests.post(self.dvidserver + "/api/node/" + self.uuid + "/" + self.dataname+self.TILENAME + "/metadata", json=tilemeta)
            if self.createtilesjpeg:
                requests.post(self.dvidserver + "/api/repo/" + self.uuid + "/instance", json={"typename": "imagetile", "dataname": self.dataname+self.JPEGTILENAME, "source": self.dataname, "format": "jpg"})
            
                requests.post(self.dvidserver + "/api/node/" + self.uuid + "/" + self.dataname+self.JPEGTILENAME + "/metadata", json=tilemeta)

        # TODO Validation: should verify syncs exist, should verify pyramid depth 

        # TODO: set syncs for pyramids, tiles if base datatype exists
        # syncs should be removed before ingestion and added afterward

        levels_cache = {}

        # iterate through each partition
        for arraypartition in imgreader:
            # DVID pad if necessary
            if self.use_dvidmask:
                dvidsrc = dvidSrc(self.dvidserver, self.uuid, self.dataname,
                        arraypartition, resource_server=self.resource_server,
                        resource_port=self.resource_port)
                arraypartition = dvidsrc.extract_volume()

            # potentially need for future iterations
            arraypartition.persist()

            # check for final layer
            finallayer = imgreader.curr_slice > imgreader.end_slice

            if not self.disable_original:
                # for each statement to disk (write jpeg at same time)
                dataname = None
                datanamelossy = None
                if self.createpyramidjpeg:
                    datanamelossy = self.dataname + self.JPEGPYRAMID_NAME
                if self.createpyramid:
                    dataname = self.dataname
                self._write_blocks(arraypartition, dataname, datanamelossy) 

            if self.createtiles or self.createtilesjpeg:
                # repartition into tiles
                schema = partitionSchema(PartitionDims(1,0,0))
                tilepartition = schema.partition_data(arraypartition)
               
                # write unpadded tilesize (will pad with delimiter if needed)
                self._writeimagepyramid(tilepartition)

            if self.createpyramid or self.createpyramidjpeg:
                if 0 not in levels_cache:
                    levels_cache[0] = []
                levels_cache[0].append(arraypartition) 
                curr_level = 1
                downsample_factor = 2

                # should be a multiple of Z blocks or the final fetch
                assert imgreader.curr_slice % self.blksize == 0
                while ((((imgreader.curr_slice / self.blksize) % downsample_factor) == 0) or finallayer) and curr_level <= self.pyramid_depth:
                    partlist = levels_cache[curr_level-1]
                    part = partlist[0]
                    # union all RDDs from the same level
                    for iter1 in range(1, len(partlist)):
                        part = part.union(partlist[iter1])
                    
                    # downsample map
                    israw = self.israw
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
                        offset = part.get_offset()
                        offsetnew = VolumeOffset(offset.z/2, offset.y/2, offset.x/2)
                        partnew = volumePartition((offsetnew.z, offsetnew.y, offsetnew.x), offsetnew)
                        return partnew, volume

                    downsampled_array = downsampled_array.map(repartition_down)
                    
                    # repartition downsample data
                    schema = partitionSchema(PartitionDims(self.blksize, self.blksize,
                        self.partition_size), blank_delimiter=self.delimiter, padding=self.blksize, enablemask=self.use_dvidmask) 
                    downsampled_array = schema.partition_data(downsampled_array)

                    # persist before padding if there are more levels
                    if curr_level < self.pyramid_depth:
                        downsampled_array.persist()
                        if curr_level not in levels_cache:
                            levels_cache[curr_level] = []
                        levels_cache[curr_level].append(downsampled_array)

                    # pad from DVID (move before persist will allow multi-ingest
                    # but will lead to slightly non-optimal downsampling boundary
                    # effects if using a lossy compression only.
                    if self.use_dvidmask:
                        padname = self.dataname
                        if self.createpyramidjpeg: # !! should pad with orig if computing
                            # pad with jpeg
                            padname += self.JPEGPYRAMID_NAME 
                        padname += "_%d" % curr_level
                        dvidsrc = dvidSrc(self.dvidserver, self.uuid, padname,
                            downsampled_array, resource_server=self.resource_server,
                            resource_port=self.resource_port)
                        downsampled_array = dvidsrc.extract_volume()

                    # write result
                    downname = None
                    downnamelossy = None
                    if self.createpyramid:
                        downname = self.dataname + "_%d" % curr_level 
                    if self.createpyramidjpeg:
                        downnamelossy = self.dataname + self.JPEGPYRAMID_NAME + "_%d" % curr_level 
                    self._write_blocks(downsampled_array, downname, downnamelossy) 

                    # remove previous level
                    del levels_cache[curr_level-1]
                    curr_level += 1
                    downsample_factor *= 2

        
    @staticmethod
    def dumpschema():
        return json.dumps(Ingest3DVolume.Schema)
