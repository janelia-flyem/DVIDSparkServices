
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

Workflow: npy.array => CompressedNumpyArray => RDD (w/ compression)


FIXME:  This whole file is starting to get really messy.
        We should fix it as follows:
        
        - Create multiple variants of CompressedNumpyArray,
         and convert the main CompressedNumpyArray class into a factory.
        
        - The view_type stuff in the pickle functions should be migrated
          to the CompressedNumpyArray itself.
          
        - We need to reformulate and consolidate the blockwise compression algorithms,
          and the code for extraction/injection of blocks into the whole array
          should not live in this file.
"""
import sys
if sys.version_info.major == 2:
    import copy_reg as copyreg
else:
    import copyreg

import numpy as np
import lz4
import logging
import warnings

from skimage.util import view_as_blocks
from DVIDSparkServices.util import box_to_slicing

# Special compression for labels (uint64)
from libdvid import encode_label_block, decode_label_block, encode_mask_array, decode_mask_array

def activate_compressed_numpy_pickling():
    """
    Override the default pickle representation for numpy arrays.
    This affects all pickle behavior in the entire process.
    """
    copyreg.pickle(np.ndarray, reduce_ndarray_compressed)
    
    # Handle subclasses, too.
    # Obviously, this only works for subclasses have been imported so far...

    import vigra # Load vigra subclass of np.ndarray

    for array_type in np.ndarray.__subclasses__():
        if array_type not in (np.ma.core.MaskedArray,):
            copyreg.pickle(array_type, reduce_ndarray_compressed)

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
        
        self.raw_buffer = None # only used if we can't compress
        self.compressed_label_blocks = None # only used for label arrays of suitable shape
        self.compressed_mask_array = None # only used for binary masks
        
        self.serialized_subarrays = []
        if numpy_array.flags['F_CONTIGUOUS']:
            self.layout = 'F'
        else:
            self.layout = 'C'

        if self.layout == 'F':
            numpy_array = numpy_array.transpose()

        self.dtype = numpy_array.dtype
        self.shape = numpy_array.shape

        # TODO: Also support compression of bool arrays via the special DVID binary compression
        if self.is_labels(numpy_array):
            self.compressed_label_blocks = serialize_uint64_blocks(numpy_array)
        elif self.dtype == np.bool and numpy_array.ndim == 3:
            # It turns out that encode_mask_array + lz4.compress is better than
            # lz4.compression alone (even multiple rounds of lz4 alone)
            self.compressed_mask_array = lz4.compress(encode_mask_array(numpy_array))
        else:

            if numpy_array.ndim <= 1:
                slice_bytes = numpy_array.nbytes
            else:
                slice_bytes = numpy_array[0].nbytes
            
            if slice_bytes > CompressedNumpyArray.MAX_LZ4_BUFFER_SIZE:
                warnings.warn("Array is too large to compress -- not compressing.")
                if not numpy_array.flags['C_CONTIGUOUS']:
                    numpy_array = numpy_array.copy(order='C')
                self.raw_buffer = bytearray(numpy_array)
            else:
                # For 1D or 0D arrays, serialize everything in one buffer.
                if numpy_array.ndim <= 1:
                    self.serialized_subarrays.append( self.serialize_subarray(numpy_array) )
                else:
                    # For ND arrays, serialize each slice independently, to ease RAM usage
                    for subarray in numpy_array:
                        self.serialized_subarrays.append( self.serialize_subarray(subarray) )

    @property
    def compressed_nbytes(self):
        if self.raw_buffer is not None:
            return len(self.raw_buffer)
        if self.compressed_label_blocks is not None:
            return len(self.compressed_label_blocks)
        if self.compressed_mask_array is not None:
            return len(self.compressed_mask_array)

        nbytes = 0
        for buf in self.serialized_subarrays:
            nbytes += len(buf)
        return nbytes

    @classmethod
    def serialize_subarray(cls, subarray):
        if not subarray.flags['C_CONTIGUOUS']:
            subarray = subarray.copy(order='C')

        # Buffers larger than 1 GB would overflow
        # We could fix this by slicing each slice into smaller pieces...
        assert subarray.nbytes <= cls.MAX_LZ4_BUFFER_SIZE, \
            "FIXME: This class doesn't support compression of arrays whose slices are each > 1 GB"
        
        return lz4.compress( subarray )

    def deserialize(self):
        """Extract the numpy array"""
        if self.raw_buffer is not None:
            # Compression was not used.
            numpy_array = np.frombuffer(self.raw_buffer, dtype=self.dtype).reshape(self.shape)
        elif self.compressed_label_blocks is not None:
            # label compression was used.
            numpy_array = deserialize_uint64_blocks(self.compressed_label_blocks, self.shape)
        elif self.compressed_mask_array is not None:
            numpy_array, _, _ = decode_mask_array(lz4.uncompress(self.compressed_mask_array), self.shape)
            numpy_array = np.asarray(numpy_array, order='C')
        else:
            numpy_array = np.ndarray( shape=self.shape, dtype=self.dtype )
            
            # See serialization of 1D and 0D arrays, above.
            if numpy_array.ndim <= 1:
                buf = lz4.uncompress(self.serialized_subarrays[0])
                numpy_array[:] = np.frombuffer(buf, self.dtype).reshape( numpy_array.shape )
            else:
                for subarray, serialized_subarray in zip(numpy_array, self.serialized_subarrays):
                    buf = lz4.uncompress(serialized_subarray)
                    subarray[:] = np.frombuffer(buf, self.dtype).reshape( subarray.shape )

        if self.layout == 'F':
            numpy_array = numpy_array.transpose()

        return numpy_array

    @classmethod
    def is_labels(cls, volume):
        return volume.dtype == np.uint64 and volume.ndim == 3

def serialize_uint64_blocks(volume):
    """
    Compress and serialize a volume of uint64.
    
    Preconditions:
      - volume.dtype == np.uint64
      - volume.ndim == 3
      
    NOTE: If volume.shape is NOT divisible by 64, the input will be copied and padded.
    
    Returns compressed_blocks, where the blocks are a flat list, in scan-order
    """
    assert volume.dtype == np.uint64
    assert volume.ndim == 3

    if (np.array(volume.shape) % 64).any():
        padding = 64 - ( np.array(volume.shape) % 64 )
        aligned_shape = volume.shape + padding
        aligned_volume = np.zeros( aligned_shape, dtype=np.uint64 )
        aligned_volume[box_to_slicing((0,0,0), volume.shape)] = volume
    else:
        aligned_volume = volume
    
    assert (np.array(aligned_volume.shape) % 64 == 0).all()
    
    block_view = view_as_blocks( aligned_volume, (64,64,64) )
    compressed_blocks = []
    for zi, yi, xi in np.ndindex(*block_view.shape[:3]):
        block = block_view[zi,yi,xi].copy('C')
        encoded_block = encode_label_block(block)

        # We compress AGAIN, with LZ4, because this seems to provide
        # an additional 2x size reduction for nearly no slowdown.
        compressed_block = lz4.compress( encoded_block )
        compressed_blocks.append( compressed_block )
        del block
    
    return compressed_blocks


def deserialize_uint64_blocks(compressed_blocks, shape):
    """
    Reconstitute a volume that was serialized with serialize_uint64_blocks(), above.
    
    NOTE: If the volume is not 64-px aligned, then the output will NOT be C-contiguous.
    """
    if (np.array(shape) % 64).any():
        padding = 64 - ( np.array(shape) % 64 )
        aligned_shape = shape + padding
    else:
        aligned_shape = shape

    aligned_volume = np.empty( aligned_shape, dtype=np.uint64 )
    block_view = view_as_blocks( aligned_volume, (64,64,64) )
    
    for bi, (zi, yi, xi) in enumerate(np.ndindex(*block_view.shape[:3])):
        compressed_block = compressed_blocks[bi]
        
        # (See note above regarding recompression with LZ4)
        encoded_block = lz4.decompress( compressed_block )
        block = decode_label_block( encoded_block )
        block_view[zi,yi,xi] = block
    
    if shape == tuple(aligned_shape):
        volume = aligned_volume
    else:
        # Trim
        volume = np.asarray(aligned_volume[box_to_slicing((0,0,0), shape)], order='C')

    return volume
        

def reduce_ndarray_compressed(a):
    """
    Custom 'reduce' function to override np.ndarray.__reduce__() during pickle saving.
    (See documentation for copy_reg module for details.)
    
    Here, we 'reduce' a numpy array to a CompressedNumpyArray.
    
    SUBCLASS SUPPORT: Subclasses of np.ndarray are also supported *IFF*
    they can be trivially reconstructed using ndarray.view() combined with
    direct assignment of the view's __dict__.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Pickling compressed numpy array: type={}, dtype={}, shape={}".format(str(type(a)), str(a.dtype), a.shape))
    assert isinstance(a, np.ndarray)
    if type(a) == np.ndarray:
        view_type = None
    else:
        view_type = type(a)
    if hasattr(a, '__dict__'):
        view_dict = a.__dict__
    else:
        view_dict = None
    
    if a.dtype == np.object or a.ndim != 3 or a.size == 0:
        # CompressedNumpyArray isn't designed for weird cases.
        # Use standard numpy pickle routine instead.
        return a.__reduce__()
    else:
        return reconstruct_ndarray_from_compressed, (CompressedNumpyArray(a), view_type, view_dict)

def reconstruct_ndarray_from_compressed(compressed_array, view_type, view_dict):
    """
    Used to reconstruct an array from it's pickled representation,
    as produced via reduce_ndarray_compressed(), above.
    """
    base = compressed_array.deserialize()
    logger = logging.getLogger(__name__)
    logger.debug("Unpickling compressed numpy array: type={}, dtype={}, shape={}".format(str(view_type), str(base.dtype), base.shape))
    if view_type is None:
        return base
    
    view = base.view(view_type)
    
    if view_dict is not None:
        view.__dict__ = view_dict
    return view

