from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

class CreateTiles2(DVIDWorkflow):
    # schema for ingesting grayscale
    Schema = """
{ "$schema": "http://json-schema.org/schema#",
  "title": "Tool to Create DVID blocks from image slices",
  "type": "object",
  "properties": {
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
        "grayname": {
          "description": "name of grayscale datatype",
          "type": "string"
        },
        "tilename": {
          "description": "name of tile datatype",
          "type": "string"
        }
      },
      "required" : ["dvid-server", "uuid", "grayname", "tilename"]
    },
    "options": {
      "type": "object",
      "properties": {
        "format": {
          "description": "compression used for tiles",
          "type": "string",
          "enum": ["png", "jpg"],
          "default": "png"
        },
        "corespertask": {
          "description": "Number of cores for each task (use higher number for memory intensive tasks)",
          "type": "integer",
          "default": 1
        }
      }
    }
  },
  "required" : ["minslice", "maxslice", "basename"]
}
    """

    APPNAME = "createtiles2"

    # calls the default initializer
    def __init__(self, config_filename):
        super(CreateTiles2, self).__init__(config_filename, self.Schema, "Create Tiles2")

    # creates tiles for dataset loaded as grayscale blocks
    def execute(self):
        # tile size default
        TILESIZE = 512
        
        server = str(self.config_data["dvid-info"]["dvid-server"])
        uuid = str(self.config_data["dvid-info"]["uuid"])
        grayname = str(self.config_data["dvid-info"]["grayname"])
        tilename = str(self.config_data["dvid-info"]["tilename"])
        
        # determine grayscale blk extants
        if not server.startswith("http://"):
            server = "http://" + server

        
        xmin, ymin, zmin = 0, 0, 0 
        
        minslice = self.config_data["minslice"]
        maxslice = self.config_data["maxslice"]
        # map file to numpy array
        basename = self.config_data["basename"]
        
        # open image
        from PIL import Image
        import requests
        import numpy
        
        img = Image.open(basename % minslice) 
        xmax, ymax, zmax = img.width, img.height, maxslice

        # create tiles type and meta
        imformat = str(self.config_data["options"]["format"])
        requests.post(server + "/api/repo/" + uuid + "/instance", json={"typename": "imagetile", "dataname": tilename, "source": grayname, "format": imformat})

        MinTileCoord = [xmin/TILESIZE, ymin/TILESIZE, zmin/TILESIZE]
        MaxTileCoord = [xmax/TILESIZE, ymax/TILESIZE, zmax/TILESIZE]
        
        # get max level by just finding max tile coord
        maxval = max(MaxTileCoord) + 1
        import math
        maxlevel = int(math.log(maxval)/math.log(2))

        tilemeta = {}
        tilemeta["MinTileCoord"] = MinTileCoord
        tilemeta["MaxTileCoord"] = MaxTileCoord
        tilemeta["Levels"] = {}
        currres = 10.0 # just use as placeholder for now
        for level in range(0, maxlevel+1):
            tilemeta["Levels"][str(level)] = { "Resolution" : [currres, currres, currres], "TileSize": [TILESIZE, TILESIZE, TILESIZE]}
            currres *= 2
        
        requests.post(server + "/api/node/" + uuid + "/" + tilename + "/metadata", json=tilemeta)
       
        # make each image a separate task
        imgs = self.sparkdvid_context.sc.parallelize(range(minslice, maxslice+1), maxslice-minslice+1)

        def img2npy(slicenum):
            try:
                img = Image.open(basename % slicenum)
                return slicenum, numpy.array(img)
            except Exception, e:
                # could give empty image, but for now just fail
                raise
        npy_images = imgs.map(img2npy) 
    
        appname = self.APPNAME

        def writeimagepyramid(image):
            slicenum, imnpy = image 
            
            from PIL import Image
            from scipy import ndimage
            import StringIO
            
            from libdvid import ConnectionMethod
            node_service = retrieve_node_service(server, uuid, appname) 

            # actually perform tile load
            def loadTile(reqpair):
                urlreq, reqbuff = reqpair 
                node_service.custom_request(urlreq, reqbuff, ConnectionMethod.POST) 
                #requests.post(urlreq , data=reqbuff)
                

            work_queue = []
            # iterate slice by slice
                
            imlevels = []
            imlevels.append(imnpy)
            # use generic downsample algorithm
            for level in range(1, maxlevel+1):
                dim1, dim2 = imlevels[level-1].shape
                imlevels.append(ndimage.interpolation.zoom(imlevels[level-1], 0.5)) 

            # write pyramid for each slice using custom request
            for levelnum in range(0, len(imlevels)):
                levelslice = imlevels[levelnum]
                dim1, dim2 = levelslice.shape

                num1tiles = (dim1-1)/TILESIZE + 1
                num2tiles = (dim2-1)/TILESIZE + 1

                for iter1 in range(0, num1tiles):
                    for iter2 in range(0, num2tiles):
                        # extract tile
                        tileholder = numpy.zeros((TILESIZE, TILESIZE), numpy.uint8)
                        min1 = iter1*TILESIZE
                        min2 = iter2*TILESIZE
                        tileslice = levelslice[min1:min1+TILESIZE, min2:min2+TILESIZE]
                        t1, t2 = tileslice.shape
                        tileholder[0:t1, 0:t2] = tileslice

                        # write tileholder to dvid
                        buf = StringIO.StringIO() 
                        img = Image.frombuffer('L', (TILESIZE, TILESIZE), tileholder.tostring(), 'raw', 'L', 0, 1)
                        imformatpil = imformat
                        if imformat == "jpg":
                            imformatpil = "jpeg"
                        img.save(buf, format=imformatpil)

                        loadTile((tilename + "/tile/xy/" + str(levelnum) + "/" + str(iter2) + "_" + str(iter1) + "_" + str(slicenum), buf.getvalue()))
                        buf.close()

        npy_images.foreach(writeimagepyramid)

    @staticmethod
    def dumpschema():
        return CreateTiles2.Schema



