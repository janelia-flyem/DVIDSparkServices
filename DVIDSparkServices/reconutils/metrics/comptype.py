"""Defines the the type of data that is compared.
"""
class ComparisonType(object):
    def __init__(self, typename="voxels", instance_name="voxels", sparse=False):
        """Initialization.

        Args:
            typename (str): name of comparison type (e.g., voxels, synapse, etc)
            instance_name (str): unique name of comparison
            sparse (boolean): true means comparison over dense (comprehensive) labeling
        """
        self.typename = typename
        self.instance_name = instance_name
        self.sparse = sparse

    def get_type(self):
        return self.typename

    def get_name(self):
        return self.typename + ":" + self.instance_name

    def __str__(self):
        return self.get_name()

    def __eq__(self, other):
        return self.typename == other.typename and self.instance_name == other.instance_name and self.sparse == other.sparse

    def __ne__(self, other):
        return self.typename != other.typename or self.instance_name != other.instance_name or self.sparse != other.sparse


