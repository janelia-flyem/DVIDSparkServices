from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

class CreatePyramid(DVIDWorkflow):
    # schema for ingesting creating image pyramid 
    Schema = """
{ "$schema": "http://json-schema.org/schema#",
  "title": "Tool to Create DVID block pyramid for DVID volume",
  "type": "object",
  "properties": {
    "dvid-info": {
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
        "source": {
          "description": "name of image volume source (only support uint8blk now)",
          "type": "string"
        }
      },
      "required" : ["dvid-server", "uuid", "source"]
    },
    "options": {
      "type": "object",
      "properties": {
        "blankdelimiter": {
          "description": "Delimiting value for a blank block",
          "type": "integer",
          "default": 0 
        },
        "xrunlimit": {
           "description": "Maximum number of blocks read per task request (0=no limit)",
           "type": "integer",
           "default": 0
        }
      }
    }
  }
}
    """
    
    # name of application for DVID queries
    APPNAME = "createpyramid"
    
    # calls the default initializer
    def __init__(self, config_filename):
        super(CreatePyramid, self).__init__(config_filename, self.Schema, "Create Pyramid")

    # creates tiles for dataset loaded as grayscale blocks
    def execute(self):
        server = str(self.config_data["dvid-info"]["dvid-server"])
        uuid = str(self.config_data["dvid-info"]["uuid"])
        source = str(self.config_data["dvid-info"]["source"])
        
        import requests
        # determine grayscale blk extants
        if not server.startswith("http://"):
            server = "http://" + server

        req = requests.get(server + "/api/node/" + uuid + "/" + source + "/info")
        sourcemeta = req.json()
        
        xmin, ymin, zmin = sourcemeta["Extended"]["MinIndex"] 
        xmax, ymax, zmax = sourcemeta["Extended"]["MaxIndex"] 
        
        # !! always assume isotropic block
        BLKSIZE = int(sourcemeta["Extended"]["BlockSize"][0])

        maxdim = max(xmax,ymax,zmax)
        # build pyramid until BLKSIZE * 4
        import math
        maxlevel = int(math.log(maxdim+1)/math.log(2)) - 1

        # assume 0,0,0 start for now
        xspan, yspan, zspan = xmax+1, ymax+1, zmax+1
        
        xrunlimit = self.config_data["options"]["xrunlimit"]
        xrunlimit = xrunlimit + (xrunlimit % 2) # should be even

        currsource = source

        # create source pyramid and append _level to name
        for level in range(1, maxlevel+1):
            node_service = retrieve_node_service(server, uuid, self.APPNAME)
            # !! limit to grayscale now
            prevsource = currsource
            currsource = source + ("_%d" % level)
            node_service.create_grayscale8(currsource, BLKSIZE)
            
           
            # set extents for new volume
            newsourceext = {}
            newsourceext["MinPoint"] = [0,0,0] # for now no offset
            newsourceext["MaxPoint"] = [((xspan-1)/2+1)*BLKSIZE-1,((yspan-1)/2+1)*BLKSIZE-1,((zspan-1)/2+1)*BLKSIZE-1]
            requests.post(server + "/api/node/" + uuid + "/" + currsource + "/extents", json=newsourceext)

            # determine number of requests
            maxxrun = xspan
            if xrunlimit > 0 and xrunlimit < xpsan:
                maxxrun = xrunlimit
            if maxxrun % 2:
                maxxrun += 1

            xsize = xspan / maxxrun
            if xspan % maxxrun:
                xsize += 1
            ysize = (yspan+1)/2
            zsize = (zspan+1)/2
            
            workqueue = []
            for ziter in range(0, zsize):
                for yiter in range(0, ysize):
                    for xiter in range(0, xsize):
                        workqueue.append((xiter,yiter,ziter))

            # parallelize jobs
            pieces = self.sc.parallelize(workqueue, len(workqueue))

            # grab data corresponding to xrun
            def retrievedata(coord):
                xiter, yiter, ziter = coord
                node_service = retrieve_node_service(server, uuid)

                shape_zyx = ( BLKSIZE*2, BLKSIZE*2, maxxrun*BLKSIZE )
                offset_zyx = (ziter*BLKSIZE*2, yiter*BLKSIZE*2, xiter*BLKSIZE*maxxrun)
                vol_zyx = node_service.get_gray3D( str(prevsource), shape_zyx, offset_zyx)

                return (coord, vol_zyx)

            volumedata = pieces.map(retrievedata)

            # downsample data
            def downsample(vdata):
                coords, data = vdata
                from scipy import ndimage
                data = ndimage.interpolation.zoom(data, 0.5)
                return (coords, data)

            downsampleddata = volumedata.map(downsample)

            appname = self.APPNAME
            delimiter = self.config_data["options"]["blankdelimiter"]
            
            # write results ?!
            def write2dvid(vdata):
                from libdvid import ConnectionMethod
                import numpy
                node_service = retrieve_node_service(server, uuid, appname) 
                
                coords, data = vdata 
                xiter, yiter, ziter = coords

                # set block indices
                zbindex = ziter
                ybindex = yiter

                zsize,ysize,xsize = data.shape
                xrun = xsize/BLKSIZE
                xbindex = xiter 

                # retrieve blocks
                blockbuffer = ""

                # skip blank blocks
                startblock = False
                xrun = 0

                for iterx in range(0, xsize, BLKSIZE):
                    block = data[:,:,iterx:iterx+BLKSIZE].copy()
                    vals = numpy.unique(block)
                    if len(vals) == 1 and vals[0] == delimiter:
                        # check if the block is blank
                        if startblock:
                            # if the previous block has data, push blocks in current queue
                            node_service.custom_request(str((currsource + "/blocks/%d_%d_%d/%d") % (xbindex, ybindex, zbindex, xrun)), blockbuffer, ConnectionMethod.POST) 
                            startblock = False
                            xrun = 0
                            blockbuffer = ""

                    else:
                        if startblock == False:
                            xbindex = xiter + iterx/BLKSIZE
                       
                        startblock = True
                        blockbuffer += block.tostring() #numpy.getbuffer(block)
                        xrun += 1


                # write-out leftover blocks
                if xrun > 0:
                    node_service.custom_request(str((currsource + "/blocks/%d_%d_%d/%d") % (xbindex, ybindex, zbindex, xrun)), blockbuffer, ConnectionMethod.POST) 


            downsampleddata.foreach(write2dvid)

            # adjust max coordinate for new level
            xspan = (xspan-1) / 2
            yspan = (yspan-1) / 2
            zspan = (zspan-1) / 2

    @staticmethod
    def dumpschema():
        return CreatePyramid.Schema



