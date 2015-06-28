import Image
import numpy
import os

from DVIDSparkServices.reconutils.workflow import Workflow

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
    }
  },
  "required" : ["minslice", "maxslice", "basename", "output-dir"]
}
    """
    # block size default
    blocksize = 32
   
    # calls the default initializer
    def __init__(self, config_filename):
        super(IngestGrayscale, self).__init__(config_filename, self.Schema, "Ingest Grayscale")

    # generates cubes of block size
    # handles stacks that are not multiples of the block dim
    # assumes all images are the same size and the same offeset
    def execute(self):
        for slice in range(self.config_data["minslice"], self.config_data["maxslice"]+1, self.blocksize):
            # parallelize images across many machines
            imgs = self.sc.parallelize(range(slice, slice+self.blocksize), self.blocksize)

            minslice = self.config_data["minslice"]

            # map file to numpy array
            basename = self.config_data["basename"]
            def img2npy(slicenum):
                try:
                    img = Image.open(basename % slicenum)
                    return slicenum, numpy.array(img)
                except Exception, e:
                    # just return a blank slice -- will be handled downstream
                    return slicenum, numpy.zeros((0,0), numpy.uint8)

            npy_images = imgs.map(img2npy) 
          
            # map numpy array into y lines of block height
            blocksize = self.blocksize
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

                    npylines.append((itery, (z, line)))

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
          
            # write blocks to disk for separte post-process -- write directly to DVID eventually?
            output_dir = self.config_data["output-dir"]
            def write2disk(yblocks):
                zbindex = slice/blocksize 
                ypos, blocks = yblocks
                ybindex = ypos / blocksize
                
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
            
    @staticmethod
    def dumpschema():
        return IngestGrayscale.Schema



