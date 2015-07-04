from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow

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


    # (stitch): => flatmap to boundary, id, cropped labels => reduce to common boundaries maps
    # => flatmap substack and boundary mappings => take ROI max id collect, join offsets with boundary
    # => join offsets and boundary mappings to persisted ROI+label, unpersist => map labels
    # (write): => for each row
    def execute(self):
        from pyspark import SparkContext
        from pyspark import StorageLevel
        from pyspark.storagelevel import StorageLevel

        # grab ROI subvolumes
        distsubvolumes = self.sparkdvid_context.parallelize_roi_new(
                self.config_data["roi"],
                self.chunksize)

        # get grayscale chunks with specified overlap
        gray_chunks = self.sparkdvid_context.map_grayscale8(distsubvolumes,
                self.config_data["grayscale"], overlap/2)

        # check for custom segmentation plugin 
        segmentation_plugin = ""
        if "segmentation-plugin" in self.config_data["options"]:
            segmentation_plugin = self.config_data["options"]["segmentation-plugin"]

        # grab seg options
        seg_options = []
        if "plugin-configuration" in self.config_data["options"]:
            seg_options = self.config_data["options"]["plugin-configuration"]

        # call the correct segmentation plugin (must be installed)
        segmentor = None
        if segmentation_plugin == "":
            segmentor = Segmentor(self.sparkdvid_context, seg_options)
        else:
            import importlib
            # assume there is a plugin folder in reconutil
            # with a module with the provided name

            segmentor_mod = importlib.import_module("DVIDSparkServices.reconutils.plugins." + segmentation_plugin)
            segmentor_class = getattr(segmentor_mod, segmentation_plugin)
            segmentor = segmentor_class(self.sparkdvid_context, seg_options)

        # convert grayscale to compressed segmentation, maintain partitioner
        # save max id as well in substack info
        seg_chunks = segmentor.segment(gray_chunks)

        # really do not need grayscale anymore
        gray_chunks.unpersist()

        # any forced persistence will result in costly
        # pickling, lz4 compressed numpy array should help
        seg_chunks.persist(StorageLevel.MEMORY_AND_DISK_SER)

        # stitch the segmentation chunks
        # (preserves initial partitioning)
        mapped_seg_chunks = segmentor.stitch(seg_chunks)

        # no longer need seg chunks
        seg_chunks.unpersist()

        # write data to DVID
        self.sparkdvid_context.foreach_write_labels3d(options["segmentation-name"], mapped_seg_chunks, overlap/2)
        self.logger.log("Wrote DVID labels") # write to logger after spark job

    @staticmethod
    def dumpschema():
        return CreateSegmentation.Schema

