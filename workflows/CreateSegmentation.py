"""Framework for general subvolume-based segmentation.

Segmentation is defined over overlapping subvolumes.  The user
can provide a custom algorithm for handling subvolume segmentation.
This workflow will run this algorithm (or its default) and stitch
the results together.

"""
import sys
import subprocess
import textwrap
import numpy as np
import DVIDSparkServices
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.util import select_item
from quilted.h5blockstore import H5BlockStore

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
              "type": "number",
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
            "corespertask": {
              "description": "Number of cores for each task (use higher number for memory intensive tasks)",
              "type": "integer",
              "default": 1
            },
            "parallelwrites": {
              "description": "Number volumes that can be simultaneously written to DVID (0 == all)",
              "type": "integer",
              "default": 125
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
        # Start the log server in a separate process
        logserver = subprocess.Popen([sys.executable, '-m', 'logcollector.logserver'])
        try:
            self.execute_impl()
        finally:
            # NOTE: Apparently the flask server doesn't respond
            #       to SIGTERM if the server is used in debug mode.
            #       If you're using the logserver in debug mode,
            #       you may need to kill it yourself.
            #       See https://github.com/pallets/werkzeug/issues/58
            print "Terminating logserver with PID {}".format(logserver.pid)
            logserver.terminate()

    def execute_impl(self):
        from pyspark import SparkContext
        from pyspark import StorageLevel
        from DVIDSparkServices.reconutils.Segmentor import Segmentor

        self.chunksize = self.config_data["options"]["chunk-size"]

        # create datatype in the beginning
        mutateseg = self.config_data["options"]["mutateseg"]
        node_service = retrieve_node_service(self.config_data["dvid-info"]["dvid-server"], 
                self.config_data["dvid-info"]["uuid"])
        success = node_service.create_labelblk(str(self.config_data["dvid-info"]["segmentation-name"]))
        # check whether seg should be mutated
        if (not success and mutateseg == "auto") or mutateseg == "yes":
            mutateseg = "yes"
        else:
            mutateseg = "no"

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
        rollback_seg = (self.config_data["options"]["checkpoint"] == "segmentation")
       
        # enable rollback of boundary prediction if necessary
        rollback_pred = (rollback_seg or self.config_data["options"]["checkpoint"] == "voxel")

        for iternum in range(0, num_iters):
            pred_checkpoint_dir = checkpoint_dir + "/prediter-" + str(iternum)
            sp_checkpoint_dir = checkpoint_dir + "/spiter-" + str(iternum)
            seg_checkpoint_dir = checkpoint_dir + "/segiter-" + str(iternum)

            # Disable rollback by setting checkpoint dirs to empty
            if checkpoint_dir == "" or not rollback_pred:
                pred_checkpoint_dir = sp_checkpoint_dir = seg_checkpoint_dir = ""
            elif not rollback_seg:
                seg_checkpoint_dir = ""

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

            subvols_with_seg_cache, subvols_without_seg_cache = \
                CreateSegmentation._split_subvols_by_cache_status( seg_checkpoint_dir,
                                                                   distsubvolumes_part.values().collect() )

            ##
            ## CACHED SUBVOLS
            ##    
            cached_subvols_rdd = self.sparkdvid_context.sc.parallelize(subvols_with_seg_cache, len(subvols_with_seg_cache) or None)
    
            # Load as many seg blocks from cache as possible
            if subvols_with_seg_cache:
                def retrieve_seg_from_cache(subvol):
                    x1, y1, z1, x2, y2, z2 = subvol.roi_with_border
                    block_bounds = ((z1, y1, x1), (z2, y2, x2))
                    block_store = H5BlockStore(seg_checkpoint_dir, mode='r')
                    h5_block = block_store.read_block( block_bounds )
                    return h5_block[:]
                cached_seg_chunks = cached_subvols_rdd.map(retrieve_seg_from_cache)
            else:
                cached_seg_chunks = self.sparkdvid_context.sc.parallelize([]) # empty rdd

            cached_seg_max_ids = cached_seg_chunks.map(np.max)
            
            # (subvol, (seg, max_id))
            cached_seg_chunks_kv = cached_subvols_rdd.zip( cached_seg_chunks.zip(cached_seg_max_ids) )

            ##
            ## UNCACHED SUBVOLS
            ##    
            uncached_subvols = self.sparkdvid_context.sc.parallelize(subvols_without_seg_cache, len(subvols_without_seg_cache) or None)
            uncached_subvols.persist()

            def prepend_roi_id(subvol):
                return (subvol.roi_id, subvol)
            uncached_subvols_kv_rdd = uncached_subvols.map(prepend_roi_id)

            # get grayscale chunks with specified overlap
            uncached_sv_and_gray = self.sparkdvid_context.map_grayscale8(uncached_subvols_kv_rdd,
                                                                         self.config_data["dvid-info"]["grayscale"])

            uncached_gray_vols = select_item(uncached_sv_and_gray, 1, 1)

            # small hack since segmentor is unaware for current iteration
            # perhaps just declare the segment function to have an arbitrary number of parameters
            if type(segmentor) == Segmentor:
                computed_seg_chunks = segmentor.segment(uncached_subvols, uncached_gray_vols,
                                                        pred_checkpoint_dir, sp_checkpoint_dir, seg_checkpoint_dir)
            else:
                computed_seg_chunks = segmentor.segment(uncached_subvols, uncached_gray_vols)

            computed_seg_max_ids = computed_seg_chunks.map( np.max )
            
            # (subvol, (seg, max_id))
            computed_seg_chunks_kv = uncached_subvols.zip( computed_seg_chunks.zip(computed_seg_max_ids) )
        
            ##
            ## FINAL LIST: COMBINED CACHED+UNCACHED
            ##
        
            # (subvol, (seg, max_id))
            seg_chunks = cached_seg_chunks_kv.union(computed_seg_chunks_kv)
            seg_chunks.persist(StorageLevel.MEMORY_AND_DISK_SER)

            seg_chunks_list.append(seg_chunks)

        seg_chunks = seg_chunks_list[0]

        for iter1 in range(1, len(seg_chunks_list)):
            # ?? does this preserve the partitioner (yes, if num partitions is the same)
            # this could cause a serialization problems if there are a large number of iterations (>100)
            seg_chunks = seg_chunks.union(seg_chunks_list[iter1])
        del seg_chunks_list

        # persist through stitch
        # any forced persistence will result in costly
        # pickling, lz4 compressed numpy array should help
        seg_chunks.persist(StorageLevel.MEMORY_AND_DISK_SER)

        # stitch the segmentation chunks
        # (preserves initial partitioning)
        mapped_seg_chunks = segmentor.stitch(seg_chunks)
        
        def prepend_key(item):
            subvol, _ = item
            return (subvol.roi_id, item)
        mapped_seg_chunks = mapped_seg_chunks.map(prepend_key)
       
        if self.config_data["options"]["parallelwrites"] > 0:
            # coalesce to fewer partition if there is write bandwidth limits to DVID
            mapped_seg_chunks = mapped_seg_chunks.coalesce(self.config_data["options"]["parallelwrites"])

        # write data to DVID
        self.sparkdvid_context.foreach_write_labels3d(self.config_data["dvid-info"]["segmentation-name"], mapped_seg_chunks, self.config_data["dvid-info"]["roi"], mutateseg)
        self.logger.write_data("Wrote DVID labels") # write to logger after spark job

        if self.config_data["options"]["debug"]:
            # grab 256 cube from ROI 
            node_service = retrieve_node_service(self.config_data["dvid-info"]["dvid-server"], 
                    self.config_data["dvid-info"]["uuid"])
            
            substacks, packing_factor = node_service.get_roi_partition(str(self.config_data["dvid-info"]["roi"]),
                                                                       256/self.blocksize)

            label_volume = node_service.get_labels3D( str(self.config_data["dvid-info"]["segmentation-name"]), 
                                                      (256,256,256),
                                                      (substacks[0].z, substacks[0].y, substacks[0].x),
                                                      compress=True )

            # dump checksum
            import numpy
            import hashlib
            md5 = hashlib.md5()
            md5.update( numpy.getbuffer(label_volume) )
            print "DEBUG: ", md5.hexdigest()

    @classmethod
    def _split_subvols_by_cache_status(cls, blockstore_dir, subvol_list):
        assert isinstance(subvol_list, list), "Must be a list, not an RDD"
        if not blockstore_dir:
            return [], subvol_list

        try:
            block_store = H5BlockStore(blockstore_dir, mode='r')
        except H5BlockStore.StoreDoesNotExistError:
            return [], subvol_list

        def is_cached(subvol):
            x1, y1, z1, x2, y2, z2 = subvol.roi_with_border
            if block_store.axes[-1] == 'c':
                return ((z1, y1, x1, 0), (z2, y2, x2, None)) in block_store
            else:
                return ((z1, y1, x1), (z2, y2, x2)) in block_store
                
        subvols_with_cache = filter( is_cached, subvol_list )
        subvols_without_cache = list(set(subvol_list) - set(subvols_with_cache))
        return subvols_with_cache, subvols_without_cache
        
    @staticmethod
    def dumpschema():
        return CreateSegmentation.Schema
