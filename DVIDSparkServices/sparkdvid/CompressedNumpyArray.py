import numpy
import lz4
import struct
import StringIO

class CompressedNumpyArray(object):
    """
    Serialize/deserialize and compress/decompress numpy array.
    This should speed up cPickling for large arrays for
    compressible arrays and improve cluster memory performance.
    LZ4 can still be implemented after RDD serialization as LZ4
    is very fast and multiple passes will lead to better compression.
    """
    lz4_chunk = 1000000000
   
    # serializes and compresses numpy array with lz4
    def __init__(self, numpy_array):
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
        
    # extract numpy array
    def deserialize(self):
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

