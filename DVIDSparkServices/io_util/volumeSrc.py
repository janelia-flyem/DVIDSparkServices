"""Defines routines to access image volume source data.

This module contains volumeSrc base class.  Derived classes
must implement the necessary functionality to handle the
specified input data format.
"""

class volumeSrc(object):
    """Iterator class provides interface for importing data.

    Large image data can be loaded in parallel using Apache
    spark and loaded into an RDD.  The data is partitioned
    using the specified partitioning scheme.  For large data,
    volumeSrc supports an iteration through partitions
    of the dataset.
    
    Note:
        Inputs can be 2D or 3D, the output is always 3D.

    """

    def __init__(self, part_schema):
        """Initialization.
            
        Args:
            partschema (partitionSchema): describes the image volume and its partitioning rules
        """
        self.part_schema = part_schema

    def __iter__(self):
        """Defines iterable type.
        """
        pass

    def next(self):
        """Iterates partitions specified in the partSchema.
        """
        pass        

    def extract_volume(self):
        """Retrieve entire volume as numpy array or RDD.
        """
        pass



