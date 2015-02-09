import json
import sys
#from pydvid.errors import DvidHttpError
import httplib

config_data = json.load(open(sys.argv[1]))

from pyspark import SparkContext

#sc = SparkContext("local", "Compute Graph")
#sc = SparkContext("local[32]", "Ingest Grayscale")
sc = SparkContext(None, "Ingest Grayscale")

# assume all slices are the same size

# block size default
blocksize = 32
jobfactor = 4

import Image
import numpy

# ?! assume multiples of 32 for now!! -- is this needed??

for slice in range(config_data["minslice"], config_data["maxslice"]+1, blocksize):
    # parallelize images across many machines
    imgs = sc.parallelize(range(slice, slice+blocksize), blocksize)

    minslice = config_data["minslice"]

    # map file to numpy array
    def img2npy(slicenum):
        return slicenum, numpy.array(Image.open(config_data["basename"] % slicenum))

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
  
    """
    # ?! dumb temp

    def stupid(blocks):
        return 42
    nums = yblocks.map(stupid)
    final_list = nums.collect()
    print len(final_list)

    # ?! end dumb temp
    """

    # sort blocks and collect (should just do a foreach when DVID can handle parallel) ?!
    sortedyblocks_rdd = yblocks.sortByKey()
    
    # inefficient collect
    yblock_sorted_list = sortedyblocks_rdd.collect()

    for yindex, blocks in yblock_sorted_list:
        # ?! pydvid write
        _, _, xs = blocks.shape
        print slice / blocksize, yindex / blocksize, "0-" + str(xs/blocksize - 1)
