from __future__ import print_function, absolute_import
from __future__ import division
from DVIDSparkServices.util import default_dvid_session
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

class IngestGrayscale(Workflow):
    # schema for ingesting grayscale
    Schema = \
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
    },
    "options" : {
      "type": "object",
      "properties": {
        "blankdelimiter": {
          "description": "Delimiting value for a blank block",
          "type": "integer",
          "default": 0 
        },
        "numblocklayers": {
          "description": "Determine the number of block layers that should be procesed in each job",
          "type": "integer",
          "default": 1
        },
        "blocksize": {
          "description": "Block size for uint8blk (default: 32x32x32)",
          "type": "integer",
          "default": 32
        },
        "blockwritelimit": {
           "description": "Maximum number of blocks written per task request (0=no limit)",
           "type": "integer",
           "default": 0
        },
        "offset": {
          "description": "Offset (x,y,z) for loading grayscale data (must be multiple of block size).  Only relevant when posting directly to DVID",
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
        "corespertask": {
          "description": "Number of cores for each task (use higher number for memory intensive tasks)",
          "type": "integer",
          "default": 1
        }
      },
      "additionalProperties": True,
      "default" : {}
    }
  },
  "required" : ["minslice", "maxslice", "basename"]
}

    @classmethod
    def schema(cls):
        return IngestGrayscale.Schema

    # name of application for DVID queries
    APPNAME = "blockingest"

    # calls the default initializer
    def __init__(self, config_filename):
        super(IngestGrayscale, self).__init__(config_filename, self.schema(), "Ingest Grayscale")
        
        # block size default
        self.BLKSIZE = self.config_data["options"]["blocksize"]
       
        # num blocks per write
        self.BLOCKLIMIT = self.config_data["options"]["blockwritelimit"]

    # generates cubes of block size
    # handles stacks that are not multiples of the block dim
    # assumes all images are the same size and the same offeset
    def execute(self):
        from PIL import Image
        import numpy
        import os
       
        iterslices = self.BLKSIZE * self.config_data["options"]["numblocklayers"]

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
            gpath = '/'.join(bucketpath[1:])


        server = None
     
        xoffset = yoffset = zoffset = 0

        if "offset" in self.config_data["options"]:
            xoffset = self.config_data["options"]["offset"][0] 
            yoffset = self.config_data["options"]["offset"][1] 
            zoffset = self.config_data["options"]["offset"][2] 

            if xoffset % self.BLKSIZE != 0:
                raise Exception("offset not block aligned")        
            if yoffset % self.BLKSIZE != 0:
                raise Exception("offset not block aligned")        
            if zoffset % self.BLKSIZE != 0:
                raise Exception("offset not block aligned")        

            xoffset /= self.BLKSIZE
            yoffset /= self.BLKSIZE
            zoffset /= self.BLKSIZE

        # this will start the Z block writing at the specified offse
        # (changes default behavior when loading nonzero starting image slice)
        zoffset -= (minslice // self.BLKSIZE)


        # create metadata before workers start if using DVID
        if "output-dir" not in self.config_data or self.config_data["output-dir"] == "":
            # write to dvid
            server = self.config_data["dvid-info"]["dvid-server"]
            uuid = self.config_data["dvid-info"]["uuid"]
            grayname = self.config_data["dvid-info"]["grayname"]
            resource_server = str(self.resource_server)
            resource_port = self.resource_port

            # create grayscale type
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port, self.APPNAME)
            node_service.create_grayscale8(str(grayname), self.BLKSIZE)

        for slice in range(self.config_data["minslice"], self.config_data["maxslice"]+1, iterslices):
            # parallelize images across many machines
            imgs = self.sc.parallelize(list(range(slice, slice+iterslices)), iterslices)

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
                except Exception as e:
                    # just return a blank slice -- will be handled downstream
                    return slicenum, numpy.zeros((0,0), numpy.uint8)

            npy_images = imgs.map(img2npy) 
          
            # map numpy array into y lines of block height
            blocksize = self.BLKSIZE
            blocklimit = self.BLOCKLIMIT 
            def npy2lines(arrpair):
                z, arr = arrpair
                ysize, xsize = arr.shape
                npylines = []
               
                for itery in range(0, ysize, blocksize):
                    line = numpy.zeros((blocksize, ((xsize-1) // blocksize + 1)*blocksize), numpy.uint8)
                    uppery = blocksize
                    if (itery + blocksize) > ysize:
                        uppery = ysize - itery

                    line[0:uppery, 0:xsize] = arr[itery:itery+blocksize, 0:xsize]

                    npylines.append((itery // blocksize, (z, line)))

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
                        blockdata = numpy.zeros((iterslices, blocksize, xsize), numpy.uint8)

                    blockdata[(z - minslice)%iterslices, :, :] = line
                return y, blockdata
            
            yblocks = groupedlines.map(lines2blocks)
       
            # map multilayer of blocks to an array of single layer blocks
            def multi2single(yblocks):
                ybindex, blocks = yblocks
                blockarr = []
                num_layers = iterslices // blocksize
                for layer in range(0,num_layers):
                    blockarr.append(((ybindex, layer), blocks[layer*blocksize:(layer*blocksize+blocksize),:,:]))

                return blockarr

            yblockssplit = yblocks.flatMap(multi2single)


            if "output-dir" in self.config_data and self.config_data["output-dir"] != "":
                # write blocks to disk for separte post-process -- write directly to DVID eventually?
                output_dir = self.config_data["output-dir"]
                def write2disk(yblocks):
                    zbindex = slice // blocksize 
                    (ybindex, layer), blocks = yblocks
                    zbindex += layer

                    zsize,ysize,xsize = blocks.shape
                    
                    outdir = output_dir 
                    outdir += "/" + ("%05d" % zbindex) + ".z/"
                    filename = outdir + ("%05d" % ybindex) + "-" + str(xsize // blocksize) + ".blocks"

                    try: 
                        os.makedirs(outdir)
                    except Exception as e:
                        pass

                    # extract blocks from buffer and write to disk
                    fout = open(filename, 'wb')
                    for iterx in range(0, xsize, blocksize):
                        block = blocks[:,:,iterx:iterx+blocksize].copy()
                        fout.write(block)
                    fout.close()

                yblockssplit.foreach(write2disk) 
            else:
                # write to dvid
                server = self.config_data["dvid-info"]["dvid-server"]
                uuid = self.config_data["dvid-info"]["uuid"]
                grayname = self.config_data["dvid-info"]["grayname"]
                appname = self.APPNAME
                delimiter = self.config_data["options"]["blankdelimiter"]
                
                def write2dvid(yblocks):
                    from libdvid import ConnectionMethod
                    import numpy
                    node_service = retrieve_node_service(server, uuid, resource_server, resource_port, appname) 
                    
                    # get block coordinates
                    zbindex = slice // blocksize 
                    (ybindex, layer), blocks = yblocks
                    zbindex += layer
                    zsize,ysize,xsize = blocks.shape
                    xrun = xsize // blocksize
                    xbindex = 0 # assume x starts at 0!!

                    # retrieve blocks
                    blockbuffer = ""

                    # skip blank blocks
                    startblock = False
                    xrun = 0
                    xbindex = 0

                    for iterx in range(0, xsize, blocksize):
                        block = blocks[:,:,iterx:iterx+blocksize].copy()
                        vals = numpy.unique(block)
                        if len(vals) == 1 and vals[0] == delimiter:
                            # check if the block is blank
                            if startblock:
                                # if the previous block has data, push blocks in current queue
                                node_service.custom_request(str((grayname + "/blocks/%d_%d_%d/%d") % (xbindex+xoffset, ybindex+yoffset, zbindex+zoffset, xrun)), blockbuffer, ConnectionMethod.POST) 
                                startblock = False
                                xrun = 0
                                blockbuffer = ""

                        else:
                            if startblock == False:
                                xbindex = iterx // blocksize
                            
                            startblock = True
                            blockbuffer += block.tobytes()
                            xrun += 1

                            if blocklimit > 0 and xrun >= blocklimit:
                                # if the previous block has data, push blocks in current queue
                                node_service.custom_request(str((grayname + "/blocks/%d_%d_%d/%d") % (xbindex+xoffset, ybindex+yoffset, zbindex+zoffset, xrun)), blockbuffer, ConnectionMethod.POST) 
                                startblock = False
                                xrun = 0
                                blockbuffer = ""

                    # write-out leftover blocks
                    if xrun > 0:
                        node_service.custom_request(str((grayname + "/blocks/%d_%d_%d/%d") % (xbindex+xoffset, ybindex+yoffset, zbindex+zoffset, xrun)), blockbuffer, ConnectionMethod.POST) 


                yblockssplit.foreach(write2dvid)
        
            self.workflow_entry_exit_printer.write_data("Ingested %d slices" % iterslices)
        
        # just fetch one image at driver to get dims
        width = height = 1
        try:
            img = None
            if gbucketname == "":
                img = Image.open(basename % minslice) 
                width, height = img.width, img.height
            else:
                from gcloud import storage
                from io import BytesIO
                client = storage.Client()
                gbucket = client.get_bucket(gbucketname)
                gblob = gbucket.get_blob(gpath % minslice)
                
                # write to bytes which implements file interface
                gblobfile = BytesIO()
                gblob.download_to_file(gblobfile)
                gblobfile.seek(0)
                img = Image.open(gblobfile)
                width, height = img.width, img.height
        except Exception as e:
            # just set size to 1 
            pass

        if "output-dir" not in self.config_data or self.config_data["output-dir"] == "":
            # update metadata
            grayext = {}
            grayext["MinPoint"] = [xoffset*self.BLKSIZE,yoffset*self.BLKSIZE,zoffset*self.BLKSIZE+minslice]
            grayext["MaxPoint"] = [xoffset*self.BLKSIZE + width-1, yoffset*self.BLKSIZE + height-1, zoffset*self.BLKSIZE+minslice + self.config_data["maxslice"]]
            if not server.startswith("http://"):
                server = "http://" + server
            session = default_dvid_session()
            session.post(server + "/api/node/" + uuid + "/" + grayname + "/extents", json=grayext)

