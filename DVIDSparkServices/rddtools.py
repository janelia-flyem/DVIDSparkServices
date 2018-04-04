import os
import sys
from math import sqrt
from collections import defaultdict
from itertools import chain
from functools import reduce

from DVIDSparkServices.util import Timer

try:
    from pyspark.rdd import RDD, portable_hash
    _RDD = RDD
except ImportError:
    #import warnings
    #warnings.warn("PySpark is not available.")
    class _RDD: pass

    def portable_hash(x):
        """
        (Copied from pyspark.rdd)
        
        This function returns consistent hash code for builtin types, especially
        for None and tuple with None.
    
        The algorithm is similar to that one used by CPython 2.7
    
        >>> portable_hash(None)
        0
        >>> portable_hash((None, 1)) & 0xffffffff
        219750521
        """
    
        if sys.version_info >= (3, 2, 3) and 'PYTHONHASHSEED' not in os.environ:
            raise Exception("Randomness of hash of string should be disabled via PYTHONHASHSEED")
    
        if x is None:
            return 0
        if isinstance(x, tuple):
            h = 0x345678
            for i in x:
                h ^= portable_hash(i)
                h *= 1000003
                h &= sys.maxsize
            h ^= len(x)
            if h == -1:
                h = -2
            return int(h)
        return hash(x)

class tuple_with_hash(tuple):
    """
    Like tuple, but you can set your own hash value if you want,
    which will be respected by better_hash(), below.
    """
    def set_hash(self, custom_hash):
        self._hash = custom_hash
    def __hash__(self):
        if hasattr(self, '_hash'):
            return self._hash
        else:
            return super().__hash__()

def better_hash(x):
    if sys.version_info >= (3, 2, 3) and 'PYTHONHASHSEED' not in os.environ:
        raise Exception("Randomness of hash of string should be disabled via PYTHONHASHSEED")

    if x is None:
        return 0
    if isinstance(x, tuple_with_hash):
        return hash(x)
    if isinstance(x, tuple):
        h = 0x345678
        for i in x:
            h ^= better_hash(i) ^ better_hash(repr(i))
            h *= 1000003
            h &= sys.maxsize
        h ^= len(x)
        if h == -1:
            h = -2
        return int(h)
    return hash(x)

#
# Functions for working with either PySpark RDDs or ordinary Python iterables.
# Warning: This defines map(), so don't use "from rddtools import *"
#
# Example usage:
#  
#    from DVIDSparkServices import rddtools as rt
#    l = rt.map(lambda x: 2*x, [1,2,3])

builtin_map = map
def map(f, iterable):
    if isinstance(iterable, _RDD):
        return iterable.map(f)
    else:
        return builtin_map(f, iterable)

def flat_map(f, iterable):
    if isinstance(iterable, _RDD):
        return iterable.flatMap(f)
    else:
        return chain(*builtin_map(f, iterable))

def map_partitions(f, iterable):
    if isinstance(iterable, _RDD):
        return iterable.mapPartitions(f, preservesPartitioning=True)
    else:
        # In the pure-python case, there's only one 'partition'.
        return f(iterable)

def map_values(f, iterable):
    if isinstance(iterable, _RDD):
        return iterable.mapValues(f)
    else:
        return ( (k,f(v)) for (k,v) in iterable )

def values(iterable):
    if isinstance(iterable, _RDD):
        return iterable.values()
    else:
        return ( v for (k,v) in iterable )

builtin_filter = filter
def filter(f, iterable):
    if isinstance(iterable, _RDD):
        return iterable.filter(f)
    else:
        return builtin_filter(f, iterable)

def group_by_key(iterable):
    if isinstance(iterable, _RDD):
        return iterable.groupByKey(partitionFunc=better_hash)
    else:
        # Note: pure-python version is not lazy!
        partitions = defaultdict(lambda: [])
        for k,v in iterable:
            partitions[k].append(v)
        return partitions.items()

def zip_with_index(iterable):
    if isinstance(iterable, _RDD):
        return iterable.zipWithIndex()
    else:
        return ((v,i) for (i,v) in enumerate(iterable))

def frugal_group_by_key(iterable):
    """
    Like group_by_key, but uses combineByKey(),
    which involves more steps but is more RAM-efficient in Spark.
    Edit: I'm no longer sure that this makes any difference...
    """
    if isinstance(iterable, _RDD):
        # Use combineByKey to avoid loading
        # all partitions into RAM at once.
        def create_combiner(val):
            return [val]
        
        def merge_value(left_list, right_val):
            left_list.append(right_val)
            return left_list
        
        def merge_combiners( left_list, right_list ):
            left_list.extend(right_list)
            return left_list
        
        return iterable.combineByKey( create_combiner, merge_value, merge_combiners, numPartitions=iterable.getNumPartitions() )
    else:
        # FIXME: Just call the regular group_by_key
        return group_by_key(iterable)
    

def foreach(f, iterable):
    if isinstance(iterable, _RDD):
        iterable.foreach(f)
    else:
        # Force execution
        reduce(lambda *_: None,  builtin_map(f, iterable))

def persist_and_execute(rdd, description, logger=None, storage=None):
    """
    Persist and execute the given RDD or iterable.
    The persisted RDD is returned (in the case of an iterable, it may not be the original)
    """
    if logger:
        logger.info(f"{description}...")

    with Timer() as timer:
        if isinstance(rdd, _RDD):
            if storage is None:
                from pyspark import StorageLevel
                storage = StorageLevel.MEMORY_ONLY

            rdd.persist(storage)
            count = rdd.count() # force eval
            parts = rdd.getNumPartitions()
            partition_counts = rdd.mapPartitions(lambda part: [sum(1 for _ in part)]).collect()
            histogram = defaultdict(lambda : 0)
            for c in partition_counts:
                histogram[c] += 1
            histogram = dict(histogram)
        else:
            rdd = list(rdd) # force eval and 'persist' in a new list
            count = len(rdd)
            parts = 1
            histogram = {count: 1}
    
    if logger:
        logger.info(f"{description} (N={count}, P={parts}, P_hist={histogram}) took {timer.timedelta}")
    
    return rdd

def unpersist(rdd):
    if isinstance(rdd, _RDD):
        # PySpark freaks out if you try to unpersist an RDD
        # that wasn't persisted in the first place
        if rdd.is_cached:
            rdd.unpersist()
    else:
        # Don't to anything for normal iterables
        # Caller must delete this collection to free memory.
        pass

