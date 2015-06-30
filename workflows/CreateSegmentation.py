from DVIDSparkServices.reconutils.dvidworkflow import DVIDWorkflow

class CreateSegmentation(DVIDWorkflow):
    # schema for creating segmentation
    # ?! modify base schema so dvid can be found
    Schema = """
{ "$schema": "http://json-schema.org/schema#",
  "title": "Service to create image segmentation from grayscale data",
  "type": "object",
  "properties": {
    "dvid-info" : {
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
        "roi": {
          "description": "region of interest to segment",
          "type": "string",
          "minLength": 1
        },
        "grayscale": {
          "description": "grayscale data to segment",
          "type": "string",
          "minLength": 1,
          "default": "grayscale"
        },
        "segmentation-name": {
          "description": "location for segmentation result",
          "type": "string",
          "minLength": 1
        }
      },
      "required": ["dvid-server", "uuid", "roi", "grayscale", "segmentation-name"]
    },
    "options" : {
      "type": "object",
      "properties": {
        "segmentation-plugin": {
          "description": "segmentation plugin to run",
          "type": "string",
          "default": "default",
          "minLength": 1
        },
        "stitch-algorithm": {
          "description": "determines aggressiveness of segmentation stitching",
          "type": "string",
          "enum": ["none", "conservative", "medium", "aggressive"],
          "default": "medium"
        },
        "plugin-configuration": {
          "description": "custom configuration as a list of key/values for plugins",
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": { "type": "string" },
              "value": { "type": "string" }
            }
          },
          "minItems": 0,
          "uniqueItems": true
        }
      },
      "required": ["segmentation-plugin", "stitch-algorithm"]
    }
  }
}
    """

    # choose reasonably big subvolume to minimize stitching effects
    chunksize = 512

    # assume blocks are 32x32x32
    blocksize = 32

    # overlap between chunks
    overlap = 40

    def __init__(self, config_filename):
        super(CreateSegmentation, self).__init__(config_filename, self.Schema, "Create segmentation")

    def execute(self):
        from pyspark import SparkContext
        from pyspark import StorageLevel

        # ?! log main actions
        # ?! move algorithms to recon (graph creation and seg seem pretty basic
        # ?! keep IO in spark dvid (how to guarantee disjointness of elements)
        # ?! return dataframe for graph to retrieve vertex or edge?
        # ?! rois might be better as dataframe
        # ?1 should roi (future read data) be a special, non-RDD object
        # ?! compress lz4

        # grab ROI (?! will need to recover distrois so probably should have collect built-in -- might not be possible)
        distrois = self.sparkdvid_context.parallelize_roi(self.config_data["roi"],
                self.chunksize)

        # get grayscale chunks with specified overlap
        gray_chunks = self.sparkdvid_context.map_grayscale8(distrois,
                self.config_data["grayscale"], overlap)

        # ?!
        segmentor = SegmentationAlgorithm.get_segmentor(self.config_data["segmentation-plugin"], self.config_data["plugin-configuration"])

        # ?! retrieve segmented chunks (=> (substack with max id, seg))
        seg_chunks = gray_chunks.map(segmentor.segmentor)

        # ?! stitch chunk (probably handle completely in  algorithm and return mappings for each substack
        # flatMap => (boundary id, partial labelvolume)
        # reduceByKey => (boundary_id, substacks, mappings) # map
        # flatMap => (substack, mappings)
        boundary_mappings = seg_chunks.flatMap(segmentor.extract_overlap).reduceByKey(segmentor.stitch_mappings).flatMap(segmentor.extract_substackmappings)

        boundary_mappings.persist()
        seg_chunks.persist()
        
        rois = distrois.collect()
        offset = 0
        new_rois = []
        for roi in rois:
            # ?! set current max
            new_rois.append((rois, offset))
            offset += roi.num_labels

        distrois = self.sc.parallelize(new_rois)

        # ?! apply mappings to DVID (or broadcast all mappings and offsets and pickup during remap??, or make sure each RDD has same partitioner)
        mapped_seg_chunks = seg_chunks.join(disrois).map(segmentor.remap)

        
        # ?! write data to DVID
        self.sparkdvid_context.writelabels(mapped_seg_chunks)




    @staticmethod
    def dumpschema():
        return CreateSegmentation.Schema

