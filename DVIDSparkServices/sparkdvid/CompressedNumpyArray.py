"""Defines class for compressing/decompressing numpy arrays.

Some numpy datasets are very large but sparse in label information.
For instance, segmentation label volumes are highly compressible
(for instance 1GB can be losslessly compressed <50 MB).  The cPickler
and default Spark do not compress serialized data.

The LZ4 compressor defined in CompressedSerializerLZ4 could be
used by itself but cPickle is very slow on large data.  It makes sense
to have a specific serialization/deserialization routine for
numpy arrays.  The cPickler works much faster over a much
smaller binary string.  The double LZ4 compression also
leads to additional compression and may only require a
small runtime fraction of the original compression.

Workflow: npy.array => CompressedNumpyArray => RDD (w/lz4 compression)

"""

import numpy
import lz4
import struct
import StringIO

class CompressedNumpyArray(object):
    """ Serialize/deserialize and compress/decompress numpy array.
    
    This should speed up cPickling for large arrays for
    compressible arrays and improve cluster memory performance.
    
    Note: The lz4 is limited to INT_MAX size.  Since labelvolumes
    can be much smaller in compressed space, this function
    supports arbitrarily large numpy arrays.  They are serialized
    as a list of LZ4 chunks where each chunk decompressed is 1GB.

    """
    lz4_chunk = 1000000000
   
    def __init__(self, numpy_array):
        """Serializes and compresses the numpy array with LZ4"""
        
        # write numpy to memory using StringIO
        memfile = StringIO.StringIO()
        numpy.save(memfile, numpy_array)
        memfile.seek(0)
        numpy_array_binary = memfile.read()
        memfile.close()

        self.serialized_data = []

        # write in chunks of 1 billion bytes to prevent overflow
        for index in range(0, len(numpy_array_binary), self.lz4_chunk):
            self.serialized_data.append(lz4.dumps(numpy_array_binary[index:index+self.lz4_chunk]))
        
    def deserialize(self):
        """Extract the numpy array"""
        
        index = 0
        deserialized_data = ""

        # retrieve lz4 chunks
        for chunk in self.serialized_data:
            deserialized_data += lz4.loads(chunk)
                
              
        # use stringio to use numpy import
        memfile = StringIO.StringIO()
        memfile.write(deserialized_data)
        memfile.seek(0)
        
        # memfile will close automatically
        return numpy.load(memfile)

