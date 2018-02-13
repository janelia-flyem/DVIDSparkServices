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

        # No new numpy arrays needed for deserialization except for the result itself.
        uncompress = assert_mem_usage_factor(1.0, comparison_input_arg=original)(compressed.deserialize)
        uncompressed = uncompress()

        assert (uncompressed == original).all()

    def test_huge_slices(self):
        """
        The current implementation of CompressedNumpyArray just doesn't compress huge arrays.
        Instead, it just captures them as-is.
        """
        orig_max_size = CompressedNumpyArray.MAX_LZ4_BUFFER_SIZE

        try:
            CompressedNumpyArray.MAX_LZ4_BUFFER_SIZE = 1000
            original = np.zeros((10+CompressedNumpyArray.MAX_LZ4_BUFFER_SIZE,1,1), dtype=np.uint8)
            
            serialized = CompressedNumpyArray( original )
            deserialized = serialized.deserialize()
            assert (original == deserialized).all()
        finally:
            CompressedNumpyArray.MAX_LZ4_BUFFER_SIZE = orig_max_size

    def test_uint64_blocks(self):
        """
        CompressedNumpyArray uses special compression for labels (uint64),
        as long as they are aligned to 64px sizes.
        """
        original = np.random.randint(1, 100, (128,128,128), np.uint64)
        assert original.flags['C_CONTIGUOUS']

        # RAM is allocated for the compressed buffers, but there should be only small numpy array allocations
        # (Each block must be copied once to a C-contiguous buffer when it is compressed.)
        factor = (64.**3 / 128.**3) * 1.01 # 1% fudge-factor
        compress = assert_mem_usage_factor(factor)(CompressedNumpyArray)
        compressed = compress( original )

        # No new numpy arrays needed for deserialization except for the result itself.
        # (Though there are some copies on the C++ side, not reflected here.)
        uncompress = assert_mem_usage_factor(1.0, comparison_input_arg=original)(compressed.deserialize)
        uncompressed = uncompress()

        assert uncompressed.flags['C_CONTIGUOUS']
        assert (uncompressed == original).all()

    def test_uint64_nonblocks(self):
        """
        CompressedNumpyArray uses special compression for labels (uint64).
        It handles non-aligned data in a somewhat clumsy way, so the RAM requirements are higher.
        """
        original = np.random.randint(1, 100, (100,100,100), np.uint64) # Not 64px block-aligned
        assert original.flags['C_CONTIGUOUS']

        # Copies are needed.
        compress = assert_mem_usage_factor(3.0)(CompressedNumpyArray)
        compressed = compress( original )

        uncompress = assert_mem_usage_factor(3.0, comparison_input_arg=original)(compressed.deserialize)        
        uncompressed = uncompress()

        assert (uncompressed == original).all()

    def test_boolean_nonblocks(self):
        """
        CompressedNumpyArray uses special compression for binary images (bool).
        """
        original = np.random.randint(0, 1, (100,100,100), np.bool)
        assert original.flags['C_CONTIGUOUS']

        # Some copying is required, especially for non-block aligned data.
        compress = assert_mem_usage_factor(3.0)(CompressedNumpyArray)
        compressed = compress( original )

        # Some copying is required, especially for non-block aligned data.
        uncompress = assert_mem_usage_factor(5.0, comparison_input_arg=original)(compressed.deserialize)
        uncompressed = uncompress()

        assert uncompressed.flags['C_CONTIGUOUS']
        assert (uncompressed == original).all()


if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
