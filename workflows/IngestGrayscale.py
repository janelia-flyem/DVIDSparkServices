from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

class IngestGrayscale(Workflow):
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
    "output-dir": { 
      "description": "Directory where blocks will be written",
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
        }
      }
    }

  },
  "required" : ["minslice", "maxslice", "basename"]
}
    """
    # block size default
    BLKSIZE = 32
   
    # calls the default initializer
    def __init__(self, config_filename):
        super(IngestGrayscale, self).__init__(config_filename, self.Schema, "Ingest Grayscale")

    # generates cubes of block size
    # handles stacks that are not multiples of the block dim
    # assumes all images are the same size and the same offeset
    def execute(self):
        from PIL import Image
        import numpy
        import os
        import string
            
        minslice = self.config_data["minslice"]
        # map file to numpy array
        basename = self.config_data["basename"]
        
        # format should be gs://<bucket>/path
        gbucketname = ""
        gpath = ""
        if basename.startswith('gs://'):
            # parse google bucket names
            tempgs = basename.split('//')
            bucketpath = tempgs[1].split('/')
            gbucketname = bucketpath[0]
            gpath = string.join(bucketpath[1:], '/')


        # create metadata before workers start if using DVID
        if "output-dir" not in self.config_data or self.config_data["output-dir"] == "":
            # write to dvid
            server = self.config_data["dvid-info"]["dvid-server"]
            uuid = self.config_data["dvid-info"]["uuid"]
            grayname = self.config_data["dvid-info"]["grayname"]
            
            # create grayscale type
            node_service = retrieve_node_service(server, uuid)
            node_service.create_grayscale8(str(grayname))

        for slice in range(self.config_data["minslice"], self.config_data["maxslice"]+1, self.BLKSIZE):
            # parallelize images across many machines
            imgs = self.sc.parallelize(range(slice, slice+self.BLKSIZE), self.BLKSIZE)

            def img2npy(slicenum):
                try:
                    img = None
                    if gbucketname == "":
                        img = Image.open(basename % slicenum)
                    else:
                        from gcloud import storage
                        from io import BytesIO
                        client = storage.Client()
                        gbucket = client.get_bucket(gbucketname)
                        gblob = gbucket.get_blob(gpath % slicenum)
                        
                        # write to bytes which implements file interface
                        gblobfile = BytesIO()
                        gblob.download_to_file(gblobfile)
                        gblobfile.seek(0)
                        img = Image.open(gblobfile)
                    return slicenum, numpy.array(img)
                except Exception, e:
                    # just return a blank slice -- will be handled downstream
                    return slicenum, numpy.zeros((0,0), numpy.uint8)

            npy_images = imgs.map(img2npy) 
          
            # map numpy array into y lines of block height
            blocksize = self.BLKSIZE
            def npy2lines(arrpair):
                z, arr = arrpair
                ysize, xsize = arr.shape
                npylines = []
               
                for itery in range(0, ysize, blocksize):
                    line = numpy.zeros((blocksize, ((xsize-1)/blocksize + 1)*blocksize), numpy.uint8)
                    uppery = blocksize
                    if (itery + blocksize) > ysize:
                        uppery = ysize - itery

                    line[0:uppery, 0:xsize] = arr[itery:itery+blocksize, 0:xsize]

                    npylines.append((itery/blocksize, (z, line)))

                return npylines

            npy_lines = npy_images.flatMap(npy2lines)

            # reduce y lines into DVID blocks
            groupedlines = npy_lines.groupByKey()         

            # map y lines => (y, blocks)
            def lines2blocks(linespair):
                y, linesp = linespair

                xsize = None
                blockdata = None
                for z, line in linesp:
                    if xsize is None:
                        _, xsize = line.shape
                        blockdata = numpy.zeros((blocksize, blocksize, xsize), numpy.uint8)

                    blockdata[(z - minslice)%blocksize, :, :] = line
                return y, blockdata
            
            yblocks = groupedlines.map(lines2blocks)
        
            if "output-dir" in self.config_data and self.config_data["output-dir"] != "":
                # write blocks to disk for separte post-process -- write directly to DVID eventually?
                output_dir = self.config_data["output-dir"]
                def write2disk(yblocks):
                    zbindex = slice/blocksize 
                    ybindex, blocks = yblocks
                    
                    zsize,ysize,xsize = blocks.shape
                    
                    outdir = output_dir 
                    outdir += "/" + ("%05d" % zbindex) + ".z/"
                    filename = outdir + ("%05d" % ybindex) + "-" + str(xsize/blocksize) + ".blocks"

                    try: 
                        os.makedirs(outdir)
                    except Exception, e:
                        pass

                    # extract blocks from buffer and write to disk
                    fout = open(filename, 'w')
                    for iterx in range(0, xsize, blocksize):
                        block = blocks[:,:,iterx:iterx+blocksize].copy()
                        fout.write(numpy.getbuffer(block))
                    fout.close()

                yblocks.foreach(write2disk) 
            else:
                # write to dvid
                server = self.config_data["dvid-info"]["dvid-server"]
                uuid = self.config_data["dvid-info"]["uuid"]
                grayname = self.config_data["dvid-info"]["grayname"]

                def write2dvid(yblocks):
                    from libdvid import ConnectionMethod
                    import numpy
                    node_service = retrieve_node_service(server, uuid)
                    
                    # get block coordinates
                    zbindex = slice/blocksize 
                    ybindex, blocks = yblocks
                    zsize,ysize,xsize = blocks.shape
                    xrun = xsize/blocksize
                    xbindex = 0 # assume x starts at 0!!

                    # retrieve blocks
                    blockbuffer = ""
                    for iterx in range(0, xsize, blocksize):
                        block = blocks[:,:,iterx:iterx+blocksize].copy()
                        blockbuffer += block.tostring() #numpy.getbuffer(block)

                    node_service.custom_request(str((grayname + "/blocks/0_%d_%d/%d") % (ybindex, zbindex, xrun)), blockbuffer, ConnectionMethod.POST) 


                yblocks.foreach(write2dvid)

    @staticmethod
    def dumpschema():
        return IngestGrayscale.Schema



