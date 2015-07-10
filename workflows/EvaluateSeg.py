from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow

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
      "required" : ["dvid-server", "uuid", "label-name", "roi", "point-lists"]
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
        "chunk-size": {
          "description": "size of subvolumes to be processed",
          "type": "integer",
          "default": 256
        },
        "boundary-size": {
          "description": "radial width of boundary for GT to mask out",
          "type": "integer",
          "default": 2
        }
      }
    }
  }
}
"""

    chunksize = 256
    writelocation = "seg-metrics"

    def __init__(self, config_filename):
        super(EvaluateSeg, self).__init__(config_filename, self.Schema, "Evaluate Segmentation")

    def execute(self):
        from DVIDSparkServices.reconutils import Evaluate
        from pyspark import SparkContext
        from pyspark import StorageLevel

        if "chunk-size" in self.config_data["options"]:
            self.chunksize = self.config_data["options"]["chunk-size"]

        #  grab ROI (no overlap and no neighbor checking)
        distrois = self.sparkdvid_context.parallelize_roi_new(self.config_data["dvid-info"]["roi"],
                self.chunksize, 0, False)

        # map ROI to two label volumes (0 overlap)
        # this will be used for all volume and point overlaps
        # (preserves partitioner)
        # (key, (subvolume, seggt, seg2)
        lpairs = self.sparkdvid_context.map_labels64_pair(
                distrois, self.config_data["dvid-info"]["label-name"],
                self.config_data["dvid-info-comp"]["dvid-server"],
                self.config_data["dvid-info-comp"]["uuid"],
                self.config_data["dvid-info-comp"]["label-name"])
        
        # ?! ?? what should I persist since I do not want to have to go back to DVID but
        # there is only 1 stage

        # evaluation tool (support RAND, VI, per body, graph, and
        # histogram stats over different sets of points)
        evaluator = Evaluate.Evaluate(self.config_data)

        # split bodies that are merged outside of the subvolume
        # (preserves partitioner)
        # => (key, (subvolume, seggt-split, seg2-split, seggt-map, seg2-map))
        lpairs_split = evaluator.split_disjoint_labels(lpairs) # could be moved to general utils

        ### VOLUMETRIC ANALYSIS ###

        # TODO: ?! Grab number of intersecting disjoint faces
        # (might need +1 border) for split edit distance
        
        # grab volumetric body overlap ignoring boundaries as specified
        # and generate overlap stats for substack (compute local)
        # => (key, (subvolume, stats, seggt-split, seg2-split, seggt-map, seg2-map))
        # (preserve partitioner)
        lpairs_proc = evaluator.calcoverlap(lpairs_split, self.config_data["options"]["boundary-size"])
        
        for point_list in self.config_data["dvid-info"]["point-lists"]:
            # Generate per substack and global stats for given points.
            # Querying will just be done on the local labels stored.
            # (preserve partitioner)
            lpairs_proc = evaluator.calcoverlap_pts(lpairs_proc, point_list)

        """
        Extract stats by retrieving substacks and stats info
        and loading indto data structures on the driver.

        Stats retrieved for volume and each set of points:

        * Rand and VI across whole set (no filter)
        * Per body VI (fragmentation factors and GT body quality)
        * Per body VI (at different filter thresholds)
        * Histogram of #bodies vs #points/annotations for GT and test
        * Approx edit distance (no filter)
        * Per body edit distance (at different filter thresholds)
        (both sides for recompute of global, plus GT)
        * (synapse points only) Synapse graph stat (at different thresholds)

        Advances:

        * Extensive breakdowns by body, opportunity for average body and outliers
        * Breakdown by different types of points
        * Explicit pruning by bio size
        * Edit distance (TODO: more realistic edit distance)
        * Thresholded graph measures
        * Histogram views and comparisons
        * Subvolume breakdowns for 'heat-map'
        (outliers are important, identify pathological mergers)
        * Ability to handle very large datasets

        Note:

        * GT body VI will be less meaningful over sparse point datasets
        * Test body fragmentation will be less meaningful over sparse point datasets
        * Edit distance can be made better.  Presumably, the actual
        nuisance metric is higher since proofreaders need to verify more than
        they actually correct.  The difference of work at different thresholds
        will indicate that one needs to be careful what one considers important.
        * TODO: compute only over important body list (probably just filter client side)
        """
        stats = evaluator.calculate_stats(lpairs_proc)

        # ?! write stats and config back to DVID with time stamp
        # (@ user_name + job + name + unique number)

        """ 
        OLD STUFF
        # print VI
        accum1 = 0
        accum2 = 0
        total = 0

        # accumulate body VI and normalizing value
        for (body, val, tot) in bodies1_vi_all:
            accum1 += val
            total += tot
       
        if "debug" in self.config_data and self.config_data["debug"]:
            # print values per body
            for (body, val, tot) in bodies1_vi_all:
                print "DEBUG: ", body, val/total # normalize
            print "DEBUG: VI total1: ", accum1/total # normalize
            

            for (body, val, tot) in bodies2_vi_all:
                print "DEBUG: ", body, val/total # normalize
                accum2 += val
            print "DEBUG: VI total2: ", accum2/total # normalize
        """

    @staticmethod
    def dumpschema():
        return EvaluateSeg.Schema
