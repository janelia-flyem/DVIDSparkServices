import json
import sys
import requests
from pydvid.labelgraph import labelgraph
from pydvid.errors import DvidHttpError
import httplib

#from jsonschema import validate

# ?! define schema (return schema if asked)

from recospark.reconutils import SimpleGraph
from recospark.sparkdvid import sparkdvid

config_data = json.load(open(sys.argv[1]))

# validate schema
"""
try:
    validate(schema, config_data)
except ValidationError, e:
    print e.what()
    quit(1)
"""

# ?? need some mechanism to dynamically set memory or will have to provision at least 2 cpus per and reduce ilastik accordingly, or will need to repartition which might be a good
# probably need to have at least 2 cores for segmentation no matter what -- it would be good to have a lower amount for simple jobs

# ?? should do memory checks and smart partitioning
# ?? how to get the number of available cores to this application (could determine parallelism)
# ?? might need to reduce the size of the jobs to smaller pieces; how to reserve more memory if task is a memory hog??
# ?? max the cpus per can be set based on a sample run of the unknown executable -- if it doesn't fit then error out
# ?? how to repartition

from pyspark import SparkContext
from pyspark import StorageLevel
#sc = SparkContext("local", "Compute Graph")
sc = SparkContext(None, "Compute Graph")
sparkdvid_context = sparkdvid.sparkdvid(sc, config_data["server"], config_data["uuid"])

#  grab ROI
distrois = sparkdvid_context.parallelize_roi(config_data["roi"], config_data["chunk-size"])

# map ROI to label volume
label_chunks = sparkdvid_context.map_labels64(distrois, config_data["label-name"], 1)

# map labels to graph data -- external program
# use default driver for now specifying custom program
# config_data["graph-builder-exe"] = "neuroproof" or ["graph-builder"] and corresponding configs 

# TODO: implement custom driver for neuroproof as a plugin ??
sg = SimpleGraph.SimpleGraph(config_data) 

# extract graph
# ?? should SimpleGraph handle these operations internally ?
graph_elements = label_chunks.flatMap(sg.build_graph) 


# group data for vertices and edges
graph_elements_red = graph_elements.reduceByKey(lambda a, b: a + b) 
graph_elements_red.persist(StorageLevel.MEMORY_ONLY) # ??
graph_vertices = graph_elements_red.filter(sg.is_vertex)
graph_edges = graph_elements_red.filter(sg.is_edge)

# create graph
conn = httplib.HTTPConnection(config_data["server"])
try:
    labelgraph.create_new(conn, config_data["uuid"], config_data["graph-name"])
except DvidHttpError:
    pass


# dump graph -- should this be wrapped through utils or through sparkdvid
# will this result in too many request (should they be accumulated) simple_graph_writer = sparkdvid_context.foreach_graph_elements(graph_vertices,
sparkdvid_context.foreach_graph_elements(graph_vertices, config_data["graph-name"])
sparkdvid_context.foreach_graph_elements(graph_edges, config_data["graph-name"])

graph_elements_red.unpersist() # ?? can I stop persisting earlier

