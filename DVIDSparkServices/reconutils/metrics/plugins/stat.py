# !! only for point/volume stats -- will need separate one for skeletons, what about connectivity?? -- maybe can reuse but metric will need to shut off if no overlap

"""Base class for all stats used in metric service.

This supports point or volume type stats that contain
overlap information.  Derived types do not need to access
Spark RDDs directly if the stat can be computed with a
simple map operation (computestate) and reduce operation
(reducestate).
"""
class StatType:
    def __init__(self):
        self.segstats = None

    @static_method 
    def iscustom_workflow():
        """Indicates whether plugin requires a specialized workflow.
        """
        return False

    def custom_workflow(segroichunks_rdd):
        """Takes RDD of segmentation and produces stats.

        Args:
            segroichunks_rdd(RDD): RDD of segmentation partitioned into chunks
        Returns:
            summarystats, bodystats, roistats
        """
        return [], [], {}

    def set_segstats(self, segstats):
        """Set segstats to allow access to overlap tables.
        """
        self.segstats = segstats

    def compute_subvolume_before_remapping(self):
        """Generates state for a given subvolume state before remapping/reduce.

        Note:
            For some stats, the state is simply the maintained
            overlap table in segstats.  In these cases, this will
            be a no-op.
        """
        return 

    def reduce_subvolume(self, stat):
        """Combine the state for two subvolumes (if there is state).
        """
        return 

    def write_subvolume_stats(self):
        """Write stats for the subvolume.

        This calling function should guarantee that the stat has only
        one subvolume id and that subvolume stats are enabled.
        """
        return []

    def write_summary_stats(self):
        """Write stats for the volume.

        The stat can report summary subvolume stats
        if subvolume stats are enabled:
        (self.segstats.disable_subvolumes == False)
        """

        return []

    def write_body_stats(self):
        """Write body stats stats if available.
        """
        return []
   
    def write_bodydebug(self):
        """Generate debug information if available.
        """
        return []

    def _get_body_volume(self, overlapset, ignorebodies=None):
        """Internal helper function to compute body size.
        """
        total = 0
        for body, overlap in overlapset:
            # do not count size related to ignored bodies
            if ignorebodies is not None and body in ignorebodies:
                continue
            total += overlap
        return total



