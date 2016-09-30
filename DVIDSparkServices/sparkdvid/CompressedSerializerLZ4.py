"""Defines class for compressing/decompressing serialized RDD data.

There are numerous challenges with using python for large datasets
in Spark.  In Spark, RDDs frequently need to be serialized/deserialized
as data is shuffled or cached.  However, in python, all cached
data is serialized (unlike in Java).  Futhermore, the flexibility in
representing data often results in inefficiently stored data.

To help reduce the size of the shuffled data, we look to compression.
While limited compression is available in pyspark, the following
is a very light-weight compression, lz4, that appears to negligibly impact
runtime while leading to good compression (for instance sparse
numpy array volumes can be shrunk by over 10x in a fraction of the
time it takes to cPickle the object).  While extensive performance
tests have not been performed, it is not likely to be the bottleneck
compared to the pickler's performance.

"""

from pyspark.serializers import FramedSerializer, PickleSerializer 
import lz4

class CompressedSerializerLZ4(FramedSerializer):
    """ Compress/decompress already serialized data using fast lz4.

        Note: extensive performance testing is still necessary.
              It might be a candidate for inclusion within the
              pyspark distribution.
        
        Note: The lz4 library can't handle objects larger than INT_MAX size.
              Therefore, this serializer can't be used with arbitrarily large arrays.
              If your arrays are larger than 2GB, you may need to manually break them
              up in your workflow or pre-compress them.

    """

    def __init__(self, serializer=PickleSerializer()):
        FramedSerializer.__init__(self)
        assert isinstance(serializer, FramedSerializer), "serializer must be a FramedSerializer"
        self.serializer = serializer

    def dumps(self, obj):
        return lz4.dumps(self.serializer.dumps(obj))

    def loads(self, obj):
        return self.serializer.loads(lz4.loads(obj))

    def __repr__(self):
        return "CompressedSerializerLZ4(%s)" % self.serializer
