"""Defines workflow for extracting stats to compare two segmentations."""

from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

class EvaluateSeg(DVIDWorkflow):
    # schema for evaluating segmentation
    Schema = """
{ "$schema": "http://json-schema.org/schema#",
  "title": "Tool to Create DVID blocks from image slices",
  "type": "object",
  "properties": {
    "dvid-info": {
      "description": "Contains DVID information for ground truth volume",
      "type": "object",
      "properties": {
        "dvid-server": { 
          "description": "location of DVID server",
          "type": "string", 
          "minLength": 1,
          "property": "dvid-server"
        },
        "uuid": {
          "description": "version node to store segmentation",
          "type": "string",
          "minLength": 1
        },
        "label-name": { 
          "description": "DVID data instance pointing to label blocks",
          "type": "string" 
        },
        "roi": { 
          "description": "name of DVID ROI for given label-name",
          "type": "string" 
        },
        "point-lists": {
          "description": "List of keyvalue DVID locations that contains seletive ponts (e.g., annotations/synapses)",
          "type": "array",
          "items": { "type": "string", "minLength": 2 },
          "minItems": 0,
          "uniqueItems": true
        },
        "stats-location": {
          "description": "Location of final results (JSON file) stored on DVID.  If there are already results present at that name, a unique number will be appended to the file name",
          "type": "string"
        },
        "user-name": {
          "description": "Name of person submitting the job",
          "type": "string"
        }
      },
      "required" : ["dvid-server", "uuid", "label-name", "roi", "point-lists", "user-name", "stats-location"]
    },
    "dvid-info-comp": {
      "description": "Contains DVID information for comparison/test volume",
      "type": "object",
      "properties": {
        "dvid-server": { 
          "description": "location of DVID server",
          "type": "string", 
          "minLength": 1,
          "property": "dvid-server"
        },
        "uuid": {
              "description": "version node to store segmentation",
              "type": "string",
              "minLength": 1
        },
        "label-name": { 
          "description": "DVID data instance pointing to label blocks",
          "type": "string" 
        }
      },
      "required" : ["dvid-server", "uuid", "label-name"]
    },
    "options": {
      "type": "object",
      "properties": {
        "body-threshold": {
          "description": "Filter GT bodies below this threshold for aggregate stats",
          "type": "integer",
          "default": 1000
        },
        "chunk-size": {
          "description": "size of subvolumes to be processed",
          "type": "integer",
          "default": 256
        },
        "boundary-size": {
          "description": "radial width of boundary for GT to mask out",
          "type": "integer",
          "default": 2
        },
        "important-bodies": {
          "description": "filter metrics based on this list of GT bodies",
          "type": "array",
          "items": {"type": "number"},
          "minItems": 0,
          "uniqueItems": true,
          "default": []
        }
      },
      "required" : ["body-threshold", "chunk-size", "boundary-size"]
    }
  }
}
"""

    # TODO:!! GT cut-off/list should probably be added
    # directly at the GT itself to indicate completeness.  Could also
    # examine a body list of complete neurons

    # TODO: !! Take body list to better pre-filter synapses for summary
    # view -- but summary numbers can mostly be computed from scratch

    chunksize = 256
    
    # 'seg-metrics' at the specified UUID will contain the evaluation results
    writelocation = "seg-metrics"

    def __init__(self, config_filename):
        super(EvaluateSeg, self).__init__(config_filename, self.Schema, "Evaluate Segmentation")
   
    def execute(self):
        # imports here so that schema can be retrieved without installation
        from DVIDSparkServices.reconutils import Evaluate
        from pyspark import SparkContext
        from pyspark import StorageLevel
        import time
        import datetime
        import json

        node_service = retrieve_node_service(self.config_data["dvid-info"]["dvid-server"],
                self.config_data["dvid-info"]["uuid"], self.resource_server, self.resource_port)

        if "chunk-size" in self.config_data["options"]:
            self.chunksize = self.config_data["options"]["chunk-size"]

        #  grab ROI (no overlap and no neighbor checking)
        distrois = self.sparkdvid_context.parallelize_roi(self.config_data["dvid-info"]["roi"],
                self.chunksize)

        # map ROI to two label volumes (0 overlap)
        # this will be used for all volume and point overlaps
        # (preserves partitioner)
        # (key, (subvolume, seggt, seg2)
        lpairs = self.sparkdvid_context.map_labels64_pair(
                distrois, self.config_data["dvid-info"]["label-name"],
                self.config_data["dvid-info-comp"]["dvid-server"],
                self.config_data["dvid-info-comp"]["uuid"],
                self.config_data["dvid-info-comp"]["label-name"], self.config_data["dvid-info"]["roi"])
      
        # filter bodies if there is a body list from GT
        important_bodies = self.config_data["options"]["important-bodies"]

        def filter_bodies(label_pairs):
            from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
            import numpy

            subvolume, labelgtc, label2c = label_pairs

            # extract numpy arrays
            labelgt = labelgtc.deserialize()
            
            # filter bodies from gt
            bodylist = numpy.unique(labelgt)
            intersecting_bodies = set(bodylist).intersection(set(important_bodies))
            mask = numpy.zeros(labelgt.shape)
            for body in intersecting_bodies:
                mask[labelgt==body] = 1
            labelgt[mask==0] = 0

            # compress results
            return (subvolume, CompressedNumpyArray(labelgt), label2c)
       
        if len(important_bodies) > 0:
            lpairs = lpairs.mapValues(filter_bodies)

        def _split_disjoint_labels(label_pairs):
            """Helper function: map subvolumes so disconnected bodies are different labels.

            Function preserves partitioner.

            Args:
                label_pairs (rdd): RDD is of (subvolume id, data)
       
            Returns:
                Original RDD including mappings for gt and the test seg.
        
            """
            from DVIDSparkServices.reconutils.morpho import split_disconnected_bodies
            from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
            
            subvolume, labelgtc, label2c = label_pairs

            # extract numpy arrays
            labelgt = labelgtc.deserialize()
            label2 = label2c.deserialize()

            # split bodies up
            labelgt_split, labelgt_map = split_disconnected_bodies(labelgt)
            label2_split, label2_map = split_disconnected_bodies(label2)
            
            # compress results
            return (subvolume, labelgt_map, label2_map,
                    CompressedNumpyArray(labelgt_split),
                    CompressedNumpyArray(label2_split))


        # split bodies that are merged outside of the subvolume
        # (preserves partitioner)
        # => (key, (subvolume, seggt-split, seg2-split, seggt-map, seg2-map))
        lpairs_split = lpairs.mapValues(_split_disjoint_labels)

        # evaluation tool (support RAND, VI, per body, graph, and
        # histogram stats over different sets of points)
        evaluator = Evaluate.Evaluate(self.config_data)

        ### VOLUMETRIC ANALYSIS ###

        # TODO: !! Grab number of intersecting disjoint faces
        # (might need +1 border) for split edit distance
        
        # grab volumetric body overlap ignoring boundaries as specified
        # and generate overlap stats for substack (compute local)
        # => (key, (subvolume, stats, seggt-split, seg2-split, seggt-map, seg2-map))
        # (preserve partitioner)
        lpairs_proc = evaluator.calcoverlap(lpairs_split, self.config_data["options"]["boundary-size"])
       
        point_data = {}
        ### POINT ANALYSIS ###
        for point_list_name in self.config_data["dvid-info"]["point-lists"]:
            # grab point list from DVID
            keyvalue = point_list_name.split('/')
            if len(keyvalue) != 2:
                raise Exception(str(point_list_name) + "point list key value not properly specified")

            # is this too large to broadcast?? -- default lz4 should help quite a bit
            # TODO: send only necessary data to each job through join might help
            point_data[keyvalue[1]] = node_service.get_json(str(keyvalue[0]),
                    str(keyvalue[1]))
            
            # Generate per substack and global stats for given points.
            # Querying will just be done on the local labels stored.
            # (preserve partitioner)
            lpairs_proc = evaluator.calcoverlap_pts(lpairs_proc, keyvalue[1], point_data[keyvalue[1]])

        # Extract stats by retrieving substacks and stats info and
        # loading into data structures on the driver.
        stats = evaluator.calculate_stats(lpairs_proc)
        
        # none or false

        debug = False
        if "debug" in self.config_data:
            debug = self.config_data["debug"]

        if debug:
            print "DEBUG:", json.dumps(stats)

        # TODO: !! maybe generate a summary view from stats, write that back
        # with simplify output, dump the more complicated file to keyvalue as well

        # write stats and config back to DVID with time stamp
        # (@ name + user name + time stamp)
        # client should use '--' delimeter to parse name
        stats["time-analyzed"] = \
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        stats["config-file"] = self.config_data
        current_time = int(time.time())

        username = str(self.config_data["dvid-info"]["user-name"])
        username = "__".join(username.split('.'))
        
        location = str(self.config_data["dvid-info"]["stats-location"])
        location = "__".join(location.split('.'))
    
        fileloc = str(location + "--" + username + "--" + str(current_time))

        node_service.create_keyvalue(self.writelocation)
        node_service.put(self.writelocation, fileloc, json.dumps(stats))


    @staticmethod
    def dumpschema():
        return EvaluateSeg.Schema
