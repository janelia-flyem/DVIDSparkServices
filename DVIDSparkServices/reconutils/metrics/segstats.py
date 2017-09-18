from __future__ import division
from math import log

"""Contains all overlap information and stat types for a given subvolume.

It provides functionality to hold stats for a given volume and top-level
functionality to combine volume stats together.
"""
class SubvolumeStats(object):
    def __init__(self, subvolume, voxelfilter=1000, ptfilter=10, num_displaybodies=100):
        # TODO: support for skeletons
        self.subvolumes = [subvolume]
        self.disable_subvolumes = False 

        # contains "rand", "vi", etc for substack
        self.subvolume_stats = []

        # contains overlaps over the various sets
        # of overlap computed
        self.gt_overlaps = []
        self.seg_overlaps = []

        self.voxelfilter = voxelfilter
        self.ptfilter = ptfilter
        self.num_displaybodies = num_displaybodies

    def compute_subvolume(self):
        """Performs metric computation for each stat and saves relevant state.

        Note: not every stat has state that needs to be
        computed at the substack level.  This should only be run
        when the number of subvolume ids is 1 as reduce should
        handle adjustments to the state.
        """
        
        assert len(self.subvolumes) == 1

        for stat in enumerate(self.subvolume_stats):
            stat.compute_subvolume_before_remapping()

    def write_subvolume_stats(self):
        """For each stat, returns subvolume stats as an array.
        """

        # should actually be a subvolume
        assert len(self.subvolumes) == 1
        assert self.disable_subvolumes


        # not all stats will support subvolume stats
        subvolumestats = []
        for stat in enumerate(self.subvolume_stats):
            subvolumestats.extend(stat.write_subvolume_stats())

        return subvolumestats

    def write_summary_stats(self): 
        """For each stat, returns summary stats as an array.
        """
        summarystats = []
        for stat in enumerate(self.subvolume_stats):
            summarystats.extend(stat.write_summary_stats())

        return summarystats

    def write_body_stats(self):
        """For each stat, returns body stats as an array.
        """
        bodystats = []
        for stat in enumerate(self.subvolume_stats):
            bodystats.extend(stat.write_body_stats())

        return bodystats

    def write_bodydebug(self):
        """For each stat, returns various debug information.
        """
        debuginfo = []
        for stat in enumerate(self.subvolume_stats):
            debuginfo.extend(stat.write_bodydebug())
        return debuginfo

    # drops subvolume stats and subvolume
    def merge_stats(self, subvolume):
        assert(len(self.seg_overlaps) == len(subvolume.seg_overlaps))
        assert(len(self.gt_overlaps) == len(subvolume.gt_overlaps))

        for iter1 in range(0, len(self.gt_overlaps)):
            self.gt_overlaps[iter1].combine_tables(subvolume.gt_overlaps[iter1])           
        for iter1 in range(0, len(self.seg_overlaps)):
            self.seg_overlaps[iter1].combine_tables(subvolume.seg_overlaps[iter1])           

        assert(len(self.subvolume_stats) == len(subvolume.subvolume_stats))

        for iter1 in enumerate(self.subvolume_stats):
            self.subvolume_stats[iter1].reduce_subvolume(subvolume.subvolume_stats[iter1])
        
        self.subvolumes.extend(subvolume.subvolumes)

    def add_gt_overlap(self, table):
        self.gt_overlaps.append(table)
    
    def add_seg_overlap(self, table):
        self.seg_overlaps.append(table)

    def add_stat(self, value):
        value.set_segstats(self)
        self.subvolume_stats.append(value)


