"""Framework for general subvolume-based segmentation.

Segmentation is defined over overlapping subvolumes.  The user
can provide a custom algorithm for handling subvolume segmentation.
This workflow will run this algorithm (or its default) and stitch
the results together.

"""

from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow

class CreateSegmentation(DVIDWorkflow):
    # schema for creating segmentation
    Schema = """\
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
          "default": "DefaultGrayOnly",
          "minLength": 0
        },
        "stitch-algorithm": {
          "description": "determines aggressiveness of segmentation stitching",
          "type": "string",
          "enum": ["none", "conservative", "medium", "aggressive"],
          "default": "medium"
        },
        "plugin-configuration": {
          "description": "custom configuration as a list of key/values for plugins",
          "type": "object"
        },
        "iteration-size": {
          "description": "Number of tasks per iteration (0 -- max size)",
          "type": "integer",
          "default": 0
        },
        "checkpoint": {
          "description": "Reuse previous results",
          "type": "string",
          "enum": ["none", "voxel", "segmentation"],
          "default": "none"
        }
      },
      "required": ["stitch-algorithm"]
    }
  }
}"""

    # choose reasonably big subvolume to minimize stitching effects
    #chunksize = 128 
    chunksize = 512 

    # assume blocks are 32x32x32
    blocksize = 32

    # overlap between chunks
    overlap = 40

    def __init__(self, config_filename):
        # ?! set number of cpus per task to 2 (make dynamic?)
        super(CreateSegmentation, self).__init__(config_filename, self.Schema, "Create segmentation")


    # (stitch): => flatmap to boundary, id, cropped labels => reduce to common boundaries maps
    # => flatmap substack and boundary mappings => take ROI max id collect, join offsets with boundary
    # => join offsets and boundary mappings to persisted ROI+label, unpersist => map labels
    # (write): => for each row
    def execute(self):
        from pyspark import SparkContext
        from pyspark import StorageLevel
        from pyspark.storagelevel import StorageLevel
        from DVIDSparkServices.reconutils.Segmentor import Segmentor

        if "chunk-size" in self.config_data["options"]:
            self.chunksize = self.config_data["options"]["chunk-size"]

        # grab ROI subvolumes and find neighbors
        distsubvolumes = self.sparkdvid_context.parallelize_roi(
                self.config_data["dvid-info"]["roi"],
                self.chunksize, self.overlap/2, True)

        # do not recompute ROI for each iteration
        distsubvolumes.persist()

        num_parts = len(distsubvolumes.collect())

        # check for custom segmentation plugin 
        segmentation_plugin = ""
        if "segmentation-plugin" in self.config_data["options"]:
            segmentation_plugin = str(self.config_data["options"]["segmentation-plugin"])

        # grab seg options
        seg_options = []
        if "plugin-configuration" in self.config_data["options"]:
            seg_options = self.config_data["options"]["plugin-configuration"]

        # call the correct segmentation plugin (must be installed)
        segmentor = None
        if segmentation_plugin == "":
            segmentor = Segmentor(self.sparkdvid_context, self.config_data, seg_options)
        else:
            import importlib
            # assume there is a plugin folder in reconutil
            # with a module with the provided name

            segmentor_mod = importlib.import_module("DVIDSparkServices.reconutils.plugins." + segmentation_plugin)
            segmentor_class = getattr(segmentor_mod, segmentation_plugin)
            segmentor = segmentor_class(self.sparkdvid_context, self.config_data, seg_options)

        # determine number of iterations
        iteration_size = num_parts
        if self.config_data["options"]["iteration-size"] != 0:
            iteration_size = self.config_data["options"]["iteration-size"]

        num_iters = num_parts/iteration_size
        if num_parts % iteration_size > 0:
            num_iters += 1

        seg_chunks_list = []

        for iternum in range(0, num_iters):
            # it might make sense to randomly map partitions for selection
            # in case something pathological is happening -- if original partitioner
            # is randomish than this should be fine
            def subset_part(roi):
                s_id, data = roi
                if (s_id % num_iters) == iternum:
                    return True
                return False
            
            # should preserve partitioner
            distsubvolumes_part = distsubvolumes.filter(subset_part)

            # get grayscale chunks with specified overlap
            gray_chunks = self.sparkdvid_context.map_grayscale8(distsubvolumes_part,
                    self.config_data["dvid-info"]["grayscale"])


            # convert grayscale to compressed segmentation, maintain partitioner
            # save max id as well in substack info
            seg_chunks = segmentor.segment(gray_chunks)

            # any forced persistence will result in costly
            # pickling, lz4 compressed numpy array should help
            seg_chunks.persist(StorageLevel.MEMORY_AND_DISK_SER)

            seg_chunks_list.append(seg_chunks)

        seg_chunks = seg_chunks_list[0]

        for iter1 in range(1, len(seg_chunks_list)):
            # ?? does this preserve the partitioner (yes, if num partitions is the same)
            seg_chunks = seg_chunks.union(seg_chunks_list[iter1])
            
        # any forced persistence will result in costly
        # pickling, lz4 compressed numpy array should help
        seg_chunks.persist(StorageLevel.MEMORY_AND_DISK_SER)

        # stitch the segmentation chunks
        # (preserves initial partitioning)
        mapped_seg_chunks = segmentor.stitch(seg_chunks)

        # write data to DVID
        self.sparkdvid_context.foreach_write_labels3d(self.config_data["dvid-info"]["segmentation-name"], mapped_seg_chunks, self.config_data["dvid-info"]["roi"])
        
        # no longer need seg chunks
        seg_chunks.unpersist()
        
        self.logger.write_data("Wrote DVID labels") # write to logger after spark job

        if "debug" in self.config_data["options"] and self.config_data["options"]["debug"]:
            # grab 256 cube from ROI 
            from libdvid import DVIDNodeService
            node_service = DVIDNodeService(str(self.config_data["dvid-info"]["dvid-server"]),
                    str(self.config_data["dvid-info"]["uuid"]))
            substacks, packing_factor = node_service.get_roi_partition(str(self.config_data["dvid-info"]["roi"]), 256/self.blocksize)
            label_volume = node_service.get_labels3D( str(self.config_data["dvid-info"]["segmentation-name"]), 
                    (256,256,256), (substacks[0][0], substacks[0][1], substacks[0][2]), compress=True )
             
            # retrieve string
            from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
            vol_compressed = CompressedNumpyArray(label_volume)
           
            # dump checksum
            import md5
            print "DEBUG: ", str(md5.new(vol_compressed.serialized_data[0]).digest())

    @staticmethod
    def dumpschema():
        return CreateSegmentation.Schema

