from __future__ import division
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

class CreatePyramid(DVIDWorkflow):
    # schema for ingesting creating image pyramid 
    Schema = \
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
          "description": "name of image volume source (uint8blk or labelblk)",
          "type": "string"
        }
      },
      "required" : ["dvid-server", "uuid", "source"]
    },
    "options": {
      "type": "object",
      "properties": {
        "resource-server": { 
          "description": "0mq resource server",
          "type": "string",
          "default": ""
        },
        "resource-port": { 
          "description": "0mq resource port",
          "type": "integer",
          "default": 12000
        },
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

    @classmethod
    def schema(cls):
        return CreatePyramid.Schema
    
    # name of application for DVID queries
    APPNAME = "createpyramid"
    
    # calls the default initializer
    def __init__(self, config_filename):
        super(CreatePyramid, self).__init__(config_filename, self.schema(), "Create Pyramid")

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
       
        # xmin, ymin, zmin not being used explicitly yet
        #xmin, ymin, zmin = sourcemeta["Extended"]["MinIndex"] 
        xmin, ymin, zmin = 0, 0, 0 
        xmax, ymax, zmax = sourcemeta["Extended"]["MaxIndex"] 
       
        islabelblk = False
        datatype = sourcemeta["Extended"]["Values"][0]["Label"]
        if str(datatype) == "labelblk":
            islabelblk = True

        # !! always assume isotropic block
        BLKSIZE = int(sourcemeta["Extended"]["BlockSize"][0])

        maxdim = max(xmax,ymax,zmax)
        # build pyramid until BLKSIZE * 4
        import math
        maxlevel = int(math.log(maxdim+1) / math.log(2)) - 2

        # assume 0,0,0 start for now
        xspan, yspan, zspan = xmax+1, ymax+1, zmax+1
        
        xrunlimit = self.config_data["options"]["xrunlimit"]
        xrunlimit = xrunlimit + (xrunlimit % 2) # should be even

        currsource = source

        # create source pyramid and append _level to name
        for level in range(1, maxlevel+1):
            node_service = retrieve_node_service(server, uuid, self.resource_server, self.resource_port, self.APPNAME)
            # !! limit to grayscale now
            prevsource = currsource
            currsource = source + ("_%d" % level)
            
            # TODO: set voxel resolution to base dataset (not too important in current workflows)
            if islabelblk:
                node_service.create_labelblk(currsource, None, BLKSIZE)
            else:
                node_service.create_grayscale8(currsource, BLKSIZE)
                # set extents for new volume (only need to do for grayscale)
                newsourceext = {}
                newsourceext["MinPoint"] = [0,0,0] # for now no offset
                newsourceext["MaxPoint"] = [((xspan-1) // 2+1)*BLKSIZE-1,((yspan-1) // 2+1)*BLKSIZE-1,((zspan-1) // 2+1)*BLKSIZE-1]
                requests.post(server + "/api/node/" + uuid + "/" + currsource + "/extents", json=newsourceext)

            # determine number of requests
            maxxrun = xspan
            if xrunlimit > 0 and xrunlimit < xspan:
                maxxrun = xrunlimit
            if maxxrun % 2:
                maxxrun += 1

            xsize = xspan // maxxrun
            if xspan % maxxrun:
                xsize += 1
            ysize = (yspan+1) // 2
            zsize = (zspan+1) // 2
            resource_server = self.resource_server
            resource_port = self.resource_port

            for ziter2 in range(0, zsize, 2):
                workqueue = []
                for yiter in range(0, ysize):
                    for xiter in range(0, xsize):
                        for miniz in range(ziter2, ziter2+2):
                            workqueue.append((xiter,yiter,miniz))

                # parallelize jobs
                pieces = self.sc.parallelize(workqueue, len(workqueue))

                # grab data corresponding to xrun
                def retrievedata(coord):
                    xiter, yiter, ziter = coord
                    node_service = retrieve_node_service(server, uuid, resource_server, resource_port)

                    shape_zyx = ( BLKSIZE*2, BLKSIZE*2, maxxrun*BLKSIZE )
                    offset_zyx = (ziter*BLKSIZE*2, yiter*BLKSIZE*2, xiter*BLKSIZE*maxxrun)
                    vol_zyx = None
                    if islabelblk:
                        vol_zyx = node_service.get_labels3D( str(prevsource), shape_zyx, offset_zyx, throttle=False)
                    else:
                        vol_zyx = node_service.get_gray3D( str(prevsource), shape_zyx, offset_zyx, throttle=False)

                    return (coord, vol_zyx)

                volumedata = pieces.map(retrievedata)

                # downsample gray data
                def downsamplegray(vdata):
                    coords, data = vdata
                    from scipy import ndimage
                    data = ndimage.interpolation.zoom(data, 0.5)
                    return (coords, data)

                # downsample label data (TODO: make faster)
                def downsamplelabels(vdata):
                    coords, data = vdata
                    import numpy 
                    zmax, ymax, xmax = data.shape
                    data2 = numpy.zeros((zmax // 2, ymax // 2, xmax // 2)).astype(numpy.uint64)

                    for ziter in range(0,zmax,2):
                        for yiter in range(0, ymax,2):
                            for xiter in range(0,xmax,2):
                                v1 = data[ziter, yiter, xiter] 
                                v2 = data[ziter, yiter, xiter+1] 
                                v3 = data[ziter, yiter+1, xiter] 
                                v4 = data[ziter, yiter+1, xiter+1] 
                                v5 = data[ziter+1, yiter, xiter] 
                                v6 = data[ziter+1, yiter, xiter+1] 
                                v7 = data[ziter+1, yiter+1, xiter] 
                                v8 = data[ziter+1, yiter+1, xiter+1]

                                freqs = {}
                                freqs[v2] = 0
                                freqs[v3] = 0
                                freqs[v4] = 0
                                freqs[v5] = 0
                                freqs[v6] = 0
                                freqs[v7] = 0
                                freqs[v8] = 0
                                
                                freqs[v1] = 1
                                freqs[v2] += 1
                                freqs[v3] += 1
                                freqs[v4] += 1
                                freqs[v5] += 1
                                freqs[v6] += 1
                                freqs[v7] += 1
                                freqs[v8] += 1

                                maxval = 0
                                freqkey = 0
                                for key, val in freqs.items():
                                        if val > maxval:
                                                maxval = val
                                                freqkey = key
        
                                data2[ziter // 2, yiter // 2, xiter // 2] = freqkey
            
                    return (coords, data2)

                downsampleddata = None
                if islabelblk:
                    downsampleddata = volumedata.map(downsamplelabels)
                else:
                    downsampleddata = volumedata.map(downsamplegray)

                appname = self.APPNAME
                delimiter = self.config_data["options"]["blankdelimiter"]
                
                # write results ?!
                def write2dvid(vdata):
                    from libdvid import ConnectionMethod
                    import numpy
                    node_service = retrieve_node_service(server, uuid, resource_server, resource_port, appname) 
                    
                    coords, data = vdata 
                    xiter, yiter, ziter = coords

                    # set block indices
                    zbindex = ziter
                    ybindex = yiter

                    zsize,ysize,xsize = data.shape
                    #xrun = xsize/BLKSIZE
                    xbindex = xiter*maxxrun // 2

                    # retrieve blocks
                    blockbuffer = ""

                    # skip blank blocks
                    startblock = False
                    xrun = 0

                    if islabelblk: 
                        vals = numpy.unique(data)
                        # TODO: ignore blank blocks within an x line 
                        if not (len(vals) == 1 and vals[0] == 0):
                            if resource_server != "":
                                node_service.put_labels3D(currsource, data, (zbindex*BLKSIZE, ybindex*BLKSIZE, xbindex*BLKSIZE), compress=True, throttle=False)
                            else:
                                node_service.put_labels3D(currsource, data, (zbindex*BLKSIZE, ybindex*BLKSIZE, xbindex*BLKSIZE), compress=True)
                    else:
                        for iterx in range(0, xsize, BLKSIZE):
                            block = data[:,:,iterx:iterx+BLKSIZE]
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
                                    xbindex = xiter*maxxrun // 2 + iterx // BLKSIZE
                               
                                startblock = True
                                blockbuffer += block.tobytes()
                                xrun += 1


                        # write-out leftover blocks
                        if xrun > 0:
                            node_service.custom_request(str((currsource + "/blocks/%d_%d_%d/%d") % (xbindex, ybindex, zbindex, xrun)), blockbuffer, ConnectionMethod.POST) 


                downsampleddata.foreach(write2dvid)

            # adjust max coordinate for new level
            xspan = (xspan-1) // 2
            yspan = (yspan-1) // 2
            zspan = (zspan-1) // 2



