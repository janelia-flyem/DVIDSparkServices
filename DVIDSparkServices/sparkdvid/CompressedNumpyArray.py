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

class CompressedNumpyArray(object):
    """ Serialize/deserialize and compress/decompress numpy array.
    
    This should speed up cPickling for large arrays for
    compressible arrays and improve cluster memory performance.
    
    Note: The lz4 is limited to INT_MAX size.  Since labelvolumes
    can be much smaller in compressed space, this function
    supports arbitrarily large numpy arrays.  They are serialized
    as a list of LZ4 chunks where each chunk decompressed is 1GB.

    """
    MAX_LZ4_BUFFER_SIZE = 1000000000
   
    def __init__(self, numpy_array):
        """Serializes and compresses the numpy array with LZ4"""

        # This would be easy to fix but I'm too lazy right now.
        assert numpy_array.ndim > 1, \
            "This class doesn't support 1D arrays."

        self.serialized_subarrays = []
        if numpy_array.flags['F_CONTIGUOUS']:
            self.layout = 'F'
        else:
            self.layout = 'C'

        if self.layout == 'F':
            numpy_array = numpy_array.transpose()

        self.dtype = numpy_array.dtype
        self.shape = numpy_array.shape

        for subarray in numpy_array:
            self.serialized_subarrays.append( self.serialize_subarray(subarray) )

    @classmethod
    def serialize_subarray(cls, subarray):
        if not subarray.flags['C_CONTIGUOUS']:
            subarray = subarray.copy(order='C')

        # Buffers larger than 1 GB would overflow
        # We could fix this by slicing each slice into smaller pieces...
        assert subarray.nbytes <= cls.MAX_LZ4_BUFFER_SIZE, \
            "FIXME: This class doesn't support arrays whose slices are each > 1 GB"
        
        return lz4.dumps( numpy.getbuffer(subarray) )
        
    def deserialize(self):
        """Extract the numpy array"""
        numpy_array = numpy.ndarray( shape=self.shape, dtype=self.dtype )
        
        for subarray, serialized_subarray in zip(numpy_array, self.serialized_subarrays):
            buf = lz4.loads(serialized_subarray)
            subarray[:] = numpy.frombuffer(buf, self.dtype).reshape( subarray.shape )
         
        if self.layout == 'F':
            return numpy_array.transpose()
        else:
            return numpy_array
