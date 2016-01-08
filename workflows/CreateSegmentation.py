"""Framework for general subvolume-based segmentation.

Segmentation is defined over overlapping subvolumes.  The user
can provide a custom algorithm for handling subvolume segmentation.
This workflow will run this algorithm (or its default) and stitch
the results together.

"""
import textwrap
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
import DVIDSparkServices

class CreateSegmentation(DVIDWorkflow):
    # schema for creating segmentation
    Schema = textwrap.dedent("""\
    {
      "$schema": "http://json-schema.org/schema#",
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
          "required": ["dvid-server", "uuid", "roi", "grayscale", "segmentation-name"],
          "additionalProperties": false
        },
        "options" : {
          "type": "object",
          "properties": {
            "segmentor": {
              "description": "Custom configuration for segmentor subclass.",
              "type": "object",
              "properties" : {
                "class": {
                  "description": "segmentation plugin to run",
                  "type": "string",
                  "default": "DVIDSparkServices.reconutils.Segmentor.Segmentor"
                },
                "configuration" : {
                  "description": "custom configuration for subclass. Schema should be supplied in subclass source.",
                  "type" : "object",
                  "default" : {}
                }
              },
              "additionalProperties": false,
              "default": {}
            },
            "stitch-algorithm": {
              "description": "determines aggressiveness of segmentation stitching",
              "type": "string",
              "enum": ["none", "conservative", "medium", "aggressive"],
              "default": "medium"
            },
            "chunk-size": {
              "description": "Size of blocks to process independently (and then stitched together).",
              "type": "integer",
              "default": 512
            },
            "label-offset": {
              "description": "Offset for first body id",
              "type": "integer",
              "default": 0
            },
            "iteration-size": {
              "description": "Number of tasks per iteration (0 -- max size)",
              "type": "integer",
              "default": 0
            },
            "checkpoint-dir": {
              "description": "Specify checkpoint directory",
              "type": "string",
              "default": ""
            },
            "checkpoint": {
              "description": "Reuse previous results",
              "type": "string",
              "enum": ["none", "voxel", "segmentation"],
              "default": "none"
            },
            "mutateseg": {
              "description": "Yes to overwrite (mutate) previous segmentation in place; auto will check to see if output label destination already exists",
              "type": "string",
              "enum": ["auto", "no", "yes"],
              "default": "auto"
            },            
            "debug": {
              "description": "Enable certain debugging functionality.  Mandatory for integration tests.",
              "type": "boolean",
              "default": false
            }
          },
          "required": ["stitch-algorithm"],
          "additionalProperties": false
        }
      }
    }
    """)

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
        from DVIDSparkServices.reconutils.Segmentor import Segmentor

        self.chunksize = self.config_data["options"]["chunk-size"]

        # grab ROI subvolumes and find neighbors
        distsubvolumes = self.sparkdvid_context.parallelize_roi(
                self.config_data["dvid-info"]["roi"],
                self.chunksize, self.overlap/2, True)

        # do not recompute ROI for each iteration
        distsubvolumes.persist()

        num_parts = len(distsubvolumes.collect())

        # Instantiate the correct Segmentor subclass (must be installed)
        import importlib
        full_segmentor_classname = self.config_data["options"]["segmentor"]["class"]
        segmentor_classname = full_segmentor_classname.split('.')[-1]
        module_name = '.'.join(full_segmentor_classname.split('.')[:-1])
        segmentor_mod = importlib.import_module(module_name)
        segmentor_class = getattr(segmentor_mod, segmentor_classname)
        segmentor = segmentor_class(self.sparkdvid_context, self.config_data)

        # determine number of iterations
        iteration_size = self.config_data["options"]["iteration-size"]
        if iteration_size == 0:
            iteration_size = num_parts

        num_iters = num_parts/iteration_size
        if num_parts % iteration_size > 0:
            num_iters += 1

        seg_chunks_list = []

        # enable checkpointing if not empty
        checkpoint_dir = self.config_data["options"]["checkpoint-dir"]

        # enable rollback of iterations if necessary
        rollback_seg = False
        if self.config_data["options"]["checkpoint"] == "segmentation":
            rollback_seg = True
       
        # enable rollback of boundary prediction if necessary
        rollback_pred = False
        if self.config_data["options"]["checkpoint"] == "voxel":
            rollback_pred = True

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
            pred_checkpoint_dir = checkpoint_dir
            if checkpoint_dir != "":
                pred_checkpoint_dir = checkpoint_dir + "/prediter-" + str(iternum)

            # disable prediction checkpointing if rolling back at the iteration level
            # as this will cause unnecessary jobs to execute.  In principle, the
            # prediction could be rolled back as well which would only add some
            # unnecessary overhead to the per iteration rollback
            if rollback_seg:
                pred_checkpoint_dir = ""

            # small hack since segmentor is unaware for current iteration
            # perhaps just declare the segment function to have an arbitrary number of parameters
            if type(segmentor) == Segmentor:
                seg_chunks = segmentor.segment(gray_chunks, pred_checkpoint_dir, rollback_pred)
            else:
                seg_chunks = segmentor.segment(gray_chunks)

            # retrieve previously computed RDD or save current RDD
            if checkpoint_dir != "":
                seg_chunks = self.sparkdvid_context.checkpointRDD(seg_chunks, 
                        checkpoint_dir + "/segiter-" + str(iternum), rollback_seg)  

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
        self.sparkdvid_context.foreach_write_labels3d(self.config_data["dvid-info"]["segmentation-name"], mapped_seg_chunks, self.config_data["dvid-info"]["roi"], self.config_data["options"]["mutateseg"])
        
        # no longer need seg chunks
        seg_chunks.unpersist()
        
        self.logger.write_data("Wrote DVID labels") # write to logger after spark job

        if self.config_data["options"]["debug"]:
            # grab 256 cube from ROI 
            from libdvid import DVIDNodeService
            node_service = DVIDNodeService(str(self.config_data["dvid-info"]["dvid-server"]),
                                           str(self.config_data["dvid-info"]["uuid"]))
            
            substacks, packing_factor = node_service.get_roi_partition(str(self.config_data["dvid-info"]["roi"]),
                                                                       256/self.blocksize)

            label_volume = node_service.get_labels3D( str(self.config_data["dvid-info"]["segmentation-name"]), 
                                                      (256,256,256),
                                                      (substacks[0][0], substacks[0][1], substacks[0][2]),
                                                      compress=True )
             
            # retrieve string
            from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
            vol_compressed = CompressedNumpyArray(label_volume)
           
            # dump checksum
            import hashlib
            md5 = hashlib.md5()
            md5.update(vol_compressed.serialized_data[0])
            checksum_text = md5.hexdigest()
            print "DEBUG: ", checksum_text

    @staticmethod
    def dumpschema():
        return CreateSegmentation.Schema
