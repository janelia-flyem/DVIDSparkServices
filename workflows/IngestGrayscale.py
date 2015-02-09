import json
import sys
import httplib
from pyspark import SparkContext

import Image
import numpy
import os


config_data = json.load(open(sys.argv[1]))

#sc = SparkContext("local", "Compute Graph")
#sc = SparkContext("local[32]", "Ingest Grayscale")
sc = SparkContext(None, "Ingest Grayscale")

# block size default
blocksize = 32

# !! handles stacks that are not multiples of 32
# !! assumes same size images; assumes 0,0 offset -- z absolute
for slice in range(config_data["minslice"], config_data["maxslice"]+1, blocksize):
    # parallelize images across many machines
    imgs = sc.parallelize(range(slice, slice+blocksize), blocksize)

    minslice = config_data["minslice"]

    # map file to numpy array
    def img2npy(slicenum):
        try:
            img = Image.open(config_data["basename"] % slicenum)
            return slicenum, numpy.array(img)
        except Exception, e:
            # just return a blank slice -- will be handled downstream
            return slicenum, numpy.zeros((0,0), numpy.uint8)

    npy_images = imgs.map(img2npy)  
  
    # map numpy array into y lines of block height
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

    # ?! repartition npy_lines ?? -- maybe for tiling at least ?
    npy_lines = npy_images.flatMap(npy2lines)

    # reduce y lines into DVID blocks -- perhaps this can be reduced to a smaller number of nodes ??
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
    def write2disk(yblocks):
        zbindex = slice/blocksize 
        ypos, blocks = yblocks
        ybindex = ypos / blocksize
        
        zsize,ysize,xsize = blocks.shape
        
        outdir = config_data["output-dir"]
        outdir += "/" + ("%05d" % zbindex) + ".z/"
        filename = outdir + ("%05d" % ybindex) + "-" + str(xsize/blocksize) + ".blocks"

        try: 
            os.makedirs(outdir)
        except Exception, e:
            pass

        # extract blocks from buffer and write to disk
        fout = open(filename, 'w')
        for iterx in range(0, xsize, blocksize):
            block = blocks[:,:,iterx:iterx+32].copy()
            fout.write(numpy.getbuffer(block))
        fout.close()

    yblocks.foreach(write2disk) 
    
    
