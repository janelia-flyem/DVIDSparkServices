from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow

class EvaluateSeg(DVIDWorkflow):
    # schema for evaluating segmentation
    Schema = """
{ "$schema": "http://json-schema.org/schema#",
  "title": "Tool to Create DVID blocks from image slices",
  "type": "object",
  "properties": {
    "dvid-server": { 
      "description": "location of DVID server",
      "type": "string" 
    },
    "uuid": { "type": "string" },
    "label-name": { 
      "description": "DVID data instance pointing to label blocks",
      "type": "string" 
    },
    "roi": { 
      "description": "name of DVID ROI for given label-name",
      "type": "string" 
    },
    "dvid-server-comp": { 
      "description": "location of DVID server",
      "type": "string" 
    },
    "uuid-comp": { "type": "string" },
    "label-name-comp": { 
      "description": "DVID data instance pointing to label blocks",
      "type": "string" 
    },
    "chunk-size": {
      "description": "size of chunks to be processed",
      "type": "integer",
      "default": 256
    }
  },
  "required" : ["dvid-server", "uuid", "label-name", "roi", "dvid-server-comp", "uuid-comp", "label-name-comp"]
}
    """

    chunksize = 256

    def __init__(self, config_filename):
        super(EvaluateSeg, self).__init__(config_filename, self.Schema, "Evaluate Segmentation")

    def execute(self):
        from DVIDSparkServices.reconutils import Evaluate
        from pyspark import SparkContext
        from pyspark import StorageLevel

        if "chunk-size" in self.config_data:
            self.chunksize = self.config_data["chunk-size"]

        #  grab ROI
        distrois = self.sparkdvid_context.parallelize_roi(self.config_data["roi"],
                self.chunksize)

        # map ROI to two label volumes (0 overlap)
        label_chunk_pairs = self.sparkdvid_context.map_labels64_pair(distrois, self.config_data["label-name"],
                self.config_data["dvid-server-comp"], self.config_data["uuid-comp"],
                self.config_data["label-name-comp"])

        
        # compute overlap within pairs
        eval = Evaluate.Evaluate(self.config_data)
       
        # ?! calculate per subvolume (I could do a map command first to get overlaps, persist,
        # then get stats and flatmap -- can I even do this because stitching will screw up subvolumes
      
        # find overlap and spit out all pairs -- no longer need substacks anymore 
        overlap_pairs = label_chunk_pairs.flatMap(eval.calcoverlap)
        
        # persist so there is no recompute
        overlap_pairs.persist(StorageLevel.MEMORY_ONLY)
       
        def extractoverlap(overlap_pair):
            dumb, overlap = overlap_pair
            b1, b2, amount = overlap
            return (b1, (b2, amount))

        # ?! do not double compute but map to different keys ??
        # grab both sides of overlap
        overlap1 = overlap_pairs.filter(eval.is_vol1).map(extractoverlap)
        overlap2 = overlap_pairs.filter(eval.is_vol2).map(extractoverlap)

        # group by key
        bodies1_overlap = overlap1.groupByKey()
        bodies2_overlap = overlap2.groupByKey()

        # map into VI components
        bodies1_vi = bodies1_overlap.map(eval.body_vi)
        bodies2_vi = bodies2_overlap.map(eval.body_vi)

        # collect results for global VI
        bodies1_vi_all = bodies1_vi.collect()
        bodies2_vi_all = bodies2_vi.collect()
        
        # can release memory for overlap
        overlap_pairs.unpersist()

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


    @staticmethod
    def dumpschema():
        return EvaluateSeg.Schema
