import numpy as np
from numpy_allocation_tracking.decorators import assert_mem_usage_factor
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray

class TestCompressedNumpyArray(object):
    
    def test_c_contiguous(self):
        original = np.random.random((100,100,100)).astype(np.float32)
        assert original.flags['C_CONTIGUOUS']

        # RAM is allocated for the lz4 buffers, but there should be 0.0 numpy array allocations
        compress = assert_mem_usage_factor(0.0)(CompressedNumpyArray)
        compressed = compress( original )

        # No new numpy arrays needed for deserialization except for the resut itself.        
        uncompress = assert_mem_usage_factor(1.0, comparison_input_arg=original)(compressed.deserialize)        
        uncompressed = uncompress()

        assert uncompressed.flags['C_CONTIGUOUS']
        assert (uncompressed == original).all()

    def test_f_contiguous(self):
        original = np.random.random((100,100,100)).astype(np.float32)
        original = original.transpose()
        assert original.flags['F_CONTIGUOUS']
        
        # RAM is allocated for the lz4 buffers, but there should be 0.0 numpy array allocations
        compress = assert_mem_usage_factor(0.0)(CompressedNumpyArray)        
        compressed = compress( original )
        
        # No new numpy arrays needed for deserialization except for the resut itself.        
        uncompress = assert_mem_usage_factor(1.0, comparison_input_arg=original)(compressed.deserialize)        
        uncompressed = uncompress()

        assert uncompressed.flags['F_CONTIGUOUS']
        assert (uncompressed == original).all()

    def test_non_contiguous(self):
        original = np.random.random((100,100,100)).astype(np.float32)
        original = original.transpose(1,2,0)
        assert not original.flags.contiguous
        
        # Since this array isn't contiguous, we need *some* overhead as the data is copied.
        # But it should only be 1 slice's worth.
        compress = assert_mem_usage_factor(0.01)(CompressedNumpyArray)        
        compressed = compress( original )

        # But decompression still requires no numpy allocations beyond the result itself.        
        uncompress = assert_mem_usage_factor(1.0, comparison_input_arg=original)(compressed.deserialize)        
        uncompressed = uncompress()

        assert (uncompressed == original).all()

    def test_1d_array(self):
        """
        A previous version of CompressedNumpyArray didn't support 1D arrays.
        Now it is supported as a special case.
        """
        original = np.random.random((1000,)).astype(np.float32)

        # RAM is allocated for the lz4 buffers, but there should be 0.0 numpy array allocations
        compress = assert_mem_usage_factor(0.0)(CompressedNumpyArray)
        compressed = compress( original )

        # No new numpy arrays needed for deserialization except for the resut itself.
        uncompress = assert_mem_usage_factor(1.0, comparison_input_arg=original)(compressed.deserialize)
        uncompressed = uncompress()

        assert (uncompressed == original).all()

if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
