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

    # TODO:!! provide option to filter GT for small bodies (this
    # might be best as a good body list (presumably dense-ish) -- could
    # be some bias if the GT is incomplete and a similar segmentation
    # algorithm is run against it.  Size filter could bias against border
    # neurons that are 'correct'.  GT cut-off/list should probably be added
    # directly at the GT itself to indicate completeness.

    # TODO: !! Take body list to better pre-filter synapses for summary
    # view -- but summary numbers can mostly be computed from scratch

    chunksize = 256
    writelocation = "seg-metrics"

    def __init__(self, config_filename):
        super(EvaluateSeg, self).__init__(config_filename, self.Schema, "Evaluate Segmentation")
    
    def execute(self):
        # imports here so that schema can be retrieved without installation
        from DVIDSparkServices.reconutils import Evaluate
        from libdvid import DVIDNodeService
        from pyspark import SparkContext
        from pyspark import StorageLevel
        import time
        import datetime
        import json

        node_service = DVIDNodeService(self.server, self.uuid)

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
        
        # evaluation tool (support RAND, VI, per body, graph, and
        # histogram stats over different sets of points)
        evaluator = Evaluate.Evaluate(self.config_data)

        # split bodies that are merged outside of the subvolume
        # (preserves partitioner)
        # => (key, (subvolume, seggt-split, seg2-split, seggt-map, seg2-map))
        lpairs_split = evaluator.split_disjoint_labels(lpairs) # could be moved to general utils

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
        for point_list in self.config_data["dvid-info"]["point-lists"]:
            # grab point list from DVID
            keyvalue = point_list_name.split('/')
            if len(keyvalue) != 2:
                raise Exception(str(point_list_name) + "point list key value not properly specified")

            # is this too large to broadcast?? -- default lz4 should help quite a bit
            point_data[keyvalue[1]] = node_service.get_json(keyvalue[0], keyvalue[1])
            
            # Generate per substack and global stats for given points.
            # Querying will just be done on the local labels stored.
            # (preserve partitioner)
            lpairs_proc = evaluator.calcoverlap_pts(lpairs_proc, keyvalue[1], point_data[keyvalue[1]])

        """
        Extract stats by retrieving substacks and stats info
        and loading indto data structures on the driver.

        Stats retrieved for volume and each set of points:

        * Rand and VI across whole set -- a few GT body thresholds
        * Per body VI (fragmentation factors and GT body quality) -- a few thresholds
        * Per body VI (at different filter thresholds) -- a few filter thresholds
        * Histogram of #bodies vs #points/annotations for GT and test
        * Approx edit distance (no filter) 
        * Per body edit distance 
        (both sides for recompute of global, plus GT)
        * Show connectivity graph of best bodies (correspond bodies and then show matrix)
        (P/R computed for different thresholds but easily handled client side; number
        of best bodies also good)
        * TODO Per body largest corrected -- a few filter thresholds (separate filter?)
        (could be useful for understanding best-case scenarios) -- skeleton version?
        * TODO Best body metrics
        * TODD: dilate points within GT body ?? -- might not really add, an arbitrary but
        well chosen point might be best
        * TODO?? (synapse points only) Synapse graph stat (at different thresholds) -- hard
        to figure out what node correpodence would be -- what does it mean to say that a pathway
        is found by test segmentation? -- probably useless, alternative ...


        Importance GT selections, GT noise filters, and importance filters (test seg side):
        
        * (client) Body stats can be selected by GT size or other preference client side.
        * (client) Bodies selected can be used to compute cumulative VI -- (just have sum
        under GT body list)
        * (client-side bio select): P/R for connectivity graph appox (importance filter could
        be used for what is similar but probably hard-code >50% of body since that will allow
        us to loosely correspond things)
        * (pre-computed) Small bodies can be filtered from GT to reduce noise per body/accum
        (this can be done just by percentages and just record what the threshold was)
        * (pre-computed) Edit distance until body is within a certain threshold. ?? (test seg side)
        Maybe try to get body 90%, 95%, 100% correct -- report per body and total (for convenience);
        compute for bodies that make up 90%, 100% of total volume?
        * (client) ?? Use slider to filter edit distance by body size
        (appoximate edit distance since splits can help multiple bodies?

        Advances:

        * Extensive breakdowns by body, opportunity for average body and outliers
        * Breakdown by different types of points
        * Explicit pruning by bio size
        * Edit distance (TODO: more realistic edit distance)
        * Thresholded synapse measures (graph measures ??) -- probably not
        * Histogram views and comparisons
        * Subvolume breakdowns for 'heat-map'
        (outliers are important, identify pathological mergers)
        * Ability to handle very large datasets
        * Display bodies that are good, help show best-case scenarios (substacks
        inform this as well)

        Note:

        * Filter for accumulative VI total is not needed since per body VI
        cna just be summed at different thresholds
        * GT body VI will be less meaningful over sparse point datasets
        * Test body fragmentation will be less meaningful over sparse point datasets
        * Edit distance can be made better.  Presumably, the actual
        nuisance metric is higher since proofreaders need to verify more than
        they actually correct.  The difference of work at different thresholds
        will indicate that one needs to be careful what one considers important.
        * TODO: compute only over important body list (probably just filter client side)
        """
        stats = evaluator.calculate_stats(lpairs_proc)

        # TODO: !! maybe generate a summary view from stats, write that back
        # with simplify output, dump the more complicated file to keyvalue as well

        # write stats and config back to DVID with time stamp
        # (@ name + user name + time stamp)
        # client should use '--' delimeter to parse name
        stats["time-analyzed"] = \
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        stats["config-file"] = self.config_data
        current_time = time.time()

        username = str(self.config_data["dvid-info"]["user-name"])
        username = username.split('.').join('__')
        
        location = str(self.config_data["dvid-info"]["stats-location"])
        location = location.split('.').join('__')
    
        fileloc = str(location + "--" + user_name + "--" + str(current_time))

        node_service.create_keyvalue(writelocation)
        node_service.put(writelocation, fileloc, json.dumps(stats))

    @staticmethod
    def dumpschema():
        return EvaluateSeg.Schema
