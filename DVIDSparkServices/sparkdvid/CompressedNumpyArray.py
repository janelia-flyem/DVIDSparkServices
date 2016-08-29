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
import copy_reg
import numpy as np
import lz4

def activate_compressed_numpy_pickling():
    """
    Override the default pickle representation for numpy arrays.
    This affects all pickle behavior in the entire process.
    """
    copy_reg.pickle(np.ndarray, reduce_ndarray_compressed)
    
    # Handle subclasses, too.
    # Obviously, this only works for subclasses have been imported so far...

    import vigra # Load vigra subclass of np.ndarray

    for array_type in np.ndarray.__subclasses__():
        if array_type not in (np.ma.core.MaskedArray,):
            copy_reg.pickle(array_type, reduce_ndarray_compressed)

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
        
        self.serialized_subarrays = []
        if numpy_array.flags['F_CONTIGUOUS']:
            self.layout = 'F'
        else:
            self.layout = 'C'

        if self.layout == 'F':
            numpy_array = numpy_array.transpose()

        self.dtype = numpy_array.dtype
        self.shape = numpy_array.shape

        # For 1D or 0D arrays, serialize everything in one buffer.
        if numpy_array.ndim <= 1:
            self.serialized_subarrays.append( self.serialize_subarray(numpy_array) )
        else:
            # For ND arrays, serialize each slice independently, to ease RAM usage
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
        
        return lz4.dumps( np.getbuffer(subarray) )
        
    def deserialize(self):
        """Extract the numpy array"""
        numpy_array = np.ndarray( shape=self.shape, dtype=self.dtype )
        
        # See serialization of 1D and 0D arrays, above.
        if numpy_array.ndim <= 1:
            buf = lz4.loads(self.serialized_subarrays[0])
            numpy_array[:] = np.frombuffer(buf, self.dtype).reshape( numpy_array.shape )
        else:
            for subarray, serialized_subarray in zip(numpy_array, self.serialized_subarrays):
                buf = lz4.loads(serialized_subarray)
                subarray[:] = np.frombuffer(buf, self.dtype).reshape( subarray.shape )
         
        if self.layout == 'F':
            numpy_array = numpy_array.transpose()

        return numpy_array
    
def reduce_ndarray_compressed(a):
    """
    Custom 'reduce' function to override np.ndarray.__reduce__() during pickle saving.
    (See documentation for copy_reg module for details.)
    
    Here, we 'reduce' a numpy array to a CompressedNumpyArray.
    
    SUBCLASS SUPPORT: Subclasses of np.ndarray are also supported *IFF*
    they can be trivially reconstructed using ndarray.view() combined with
    direct assignment of the view's __dict__.
    """
    #print "PICKLING COMPRESSED NUMPY ARRAY! VIEW_TYPE={}, DTYPE={}, SHAPE={}".format(str(type(a)), str(a.dtype), a.shape)
    assert isinstance(a, np.ndarray)
    if type(a) == np.ndarray:
        view_type = None
    else:
        view_type = type(a)
    if hasattr(a, '__dict__'):
        view_dict = a.__dict__
    else:
        view_dict = None
    return reconstruct_ndarray_from_compressed, (CompressedNumpyArray(a), view_type, view_dict)

def reconstruct_ndarray_from_compressed(compressed_array, view_type, view_dict):
    """
    Used to reconstruct an array from it's pickled representation,
    as produced via reduce_ndarray_compressed(), above.
    """
    base = compressed_array.deserialize()
    #print "************UN-PICKLED COMPRESSED NUMPY ARRAY! VIEW_TYPE={}, DTYPE={}, SHAPE={}".format(str(view_type), str(base.dtype), base.shape)
    if view_type is None:
        return base
    
    view = base.view(view_type)
    
    if view_dict is not None:
        view.__dict__ = view_dict
    return view

