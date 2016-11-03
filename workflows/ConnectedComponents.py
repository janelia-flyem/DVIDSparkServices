"""Framework for large-scale connected components over an ROI."""
import textwrap
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow

class ConnectedComponents(DVIDWorkflow):
    # schema for creating segmentation
    Schema = textwrap.dedent("""\
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to create connected components from segmentation",
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
            "segmentation": {
              "description": "location of original segmentation",
              "type": "string",
              "minLength": 1
            },
            "newsegmentation": {
              "description": "location for segmentation result",
              "type": "string",
              "minLength": 1
            }
          },
          "required": ["dvid-server", "uuid", "roi", "segmentation", "newsegmentation"],
          "additionalProperties": false
        },
        "options" : {
          "type": "object",
          "properties": {
            "chunk-size": {
              "description": "Size of blocks to process independently (and then stitched together).",
              "type": "integer",
              "default": 512
            },
            "debug": {
              "description": "Enable certain debugging functionality.  Mandatory for integration tests.",
              "type": "boolean",
              "default": false
            }
          },
          "additionalProperties": false
        }
      }
    }
    """)

    # assume blocks are 32x32x32
    blocksize = 32

    # overlap between chunks
    overlap = 2

    def __init__(self, config_filename):
        # ?! set number of cpus per task to 2 (make dynamic?)
        super(ConnectedComponents, self).__init__(config_filename, self.Schema, "ConnectedComponents")


    # (stitch): => flatmap to boundary, id, cropped labels => reduce to common boundaries maps
    # => flatmap substack and boundary mappings => take ROI max id collect, join offsets with boundary
    # => join offsets and boundary mappings to persisted ROI+label, unpersist => map labels
    # (write): => for each row
    def execute(self):
        from pyspark import SparkContext
        from pyspark import StorageLevel
        from DVIDSparkServices.reconutils.Segmentor import Segmentor
        import numpy
        import vigra

        self.chunksize = self.config_data["options"]["chunk-size"]

        # grab ROI subvolumes and find neighbors
        distsubvolumes = self.sparkdvid_context.parallelize_roi(
                self.config_data["dvid-info"]["roi"],
                self.chunksize, self.overlap/2, True)
        distsubvolumes.persist(StorageLevel.MEMORY_AND_DISK_SER)

        # grab seg chunks 
        seg_chunks = self.sparkdvid_context.map_labels64(distsubvolumes,
                self.config_data["dvid-info"]["segmentation"],
                self.overlap/2, self.config_data["dvid-info"]["roi"])

        # pass substack with labels (no shuffling)
        seg_chunks2 = distsubvolumes.join(seg_chunks) # (sv_id, (subvolume, segmentation))
        distsubvolumes.unpersist()
        
        # run connected components
        def connected_components(seg_chunk):
            from DVIDSparkServices.reconutils.morpho import split_disconnected_bodies

            subvolume, seg = seg_chunk
            seg2, _mapping = split_disconnected_bodies(seg)

            # renumber from one
            #
            # FIXME: The next version of vigra will have a function to do this in one step, like this:
            #        seg2 = vigra.analysis.relabelConsecutive( seg2,
            #                                                  start_label=1,
            #                                                  keep_zeros=True,
            #                                                  out=np.empty_like(seg2, dtype=np.uint32))
            seg2 = vigra.taggedView(seg2, 'zyx')
            vals = numpy.sort( vigra.analysis.unique(seg2) )
            if vals[0] == 0:
                # Leave zero-pixels alone
                remap = dict(zip(vals, range(len(vals))))
            else:
                remap = dict(zip(vals, range(1, 1+len(vals))))

            out_seg = numpy.empty_like(seg2, dtype=numpy.uint32)
            vigra.analysis.applyMapping(seg2, remap, out=out_seg)
            return (subvolume, (out_seg, out_seg.max()))

        # (sv_id, (subvolume, labels)) -> (sv_id, (subvolume, (newlabels, max_id)))
        seg_chunks_cc = seg_chunks2.mapValues(connected_components)
        
        # stitch the segmentation chunks
        # (preserves initial partitioning)
        from DVIDSparkServices.reconutils.morpho import stitch
        mapped_seg_chunks = stitch(self.sparkdvid_context.sc, seg_chunks_cc.values())

        # This is to make the foreach_write_labels3d() function happy
        def prepend_key(item):
            subvol, _ = item
            return (subvol.roi_id, item)
        mapped_seg_chunks = mapped_seg_chunks.map(prepend_key)
       
        # use fewer partitions (TEMPORARY SINCE THERE ARE WRITE BANDWIDTH LIMITS TO DVID)
        #mapped_seg_chunks = mapped_seg_chunks.repartition(125)

        # write data to DVID
        self.sparkdvid_context.foreach_write_labels3d(self.config_data["dvid-info"]["newsegmentation"], mapped_seg_chunks)
        self.logger.write_data("Wrote DVID labels") # write to logger after spark job


    @staticmethod
    def dumpschema():
        return ConnectedComponents.Schema
