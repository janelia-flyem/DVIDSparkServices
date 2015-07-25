import lz4
from pyspark.serializers import FramedSerializer, PickleSerializer 

# create custom lz4 rdd compression -- this should greatly
# speed-up RDD shuffling involving large label volumes


# (try to add this to default spark -- not sure whether lz4
# compression flags in the documentation apply to PySpark)
class CompressedSerializerLZ4(FramedSerializer):
    """
    Compress the serialized data
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
