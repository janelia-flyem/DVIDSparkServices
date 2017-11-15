from collections import defaultdict
from itertools import chain
from functools import reduce

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
