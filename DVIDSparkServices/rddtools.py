from collections import defaultdict
from itertools import chain
from functools import reduce

from DVIDSparkServices.util import Timer

try:
    from pyspark.rdd import RDD
    _RDD = RDD
except ImportError:
    import warnings
    warnings.warn("PySpark is not available.")
    class _RDD: pass

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

def group_by_key(iterable):
    if isinstance(iterable, _RDD):
        return iterable.groupByKey()
    else:
        # Note: pure-python version is not lazy!
        partitions = defaultdict(lambda: [])
        for k,v in iterable:
            partitions[k].append(v)
        return partitions.items()

def foreach(f, iterable):
    if isinstance(iterable, _RDD):
        iterable.foreach(f)
    else:
        # Force execution
        reduce(lambda *_: None,  builtin_map(f, iterable))

def persist_and_execute(rdd, description, logger=None):
    """
    Persist and execute the given RDD or iterable.
    The persisted RDD is returned (in the case of an iterable, it may not be the original)
    """
    if logger:
        logger.info(f"{description}...")

    with Timer() as timer:
        if isinstance(rdd, _RDD):
            from pyspark import StorageLevel
        
            rdd.persist(StorageLevel.MEMORY_AND_DISK)
            count = rdd.count() # force eval
        else:
            rdd = list(rdd) # force eval and 'persist' in a new list
            count = len(rdd)
    
    if logger:
        logger.info(f"{description} (N={count}) took {timer.timedelta}")
    
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
