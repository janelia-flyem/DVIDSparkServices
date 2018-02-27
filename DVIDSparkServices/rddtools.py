from collections import defaultdict
from itertools import chain
from functools import reduce

from DVIDSparkServices.util import Timer

try:
    from pyspark.rdd import RDD
    _RDD = RDD
except ImportError:
    #import warnings
    #warnings.warn("PySpark is not available.")
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

def frugal_group_by_key(iterable):
    """
    Like group_by_key, but uses combineByKey(),
    which involves more steps but is more RAM-efficient in Spark.
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
