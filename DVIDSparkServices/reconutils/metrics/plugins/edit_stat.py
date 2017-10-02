from .stat import *

"""Class provides edit distance based on simple voxel/point overlap.

It performs a greedy-based assignment of bodies from the segmentation
to groundtruth, performing mergers or splits based on simple overlap.
If volume correct threshold is less than 100 percent, the greedy algorithm
will require a relative estimate for splitting work vs merge work.

Note: does not provide subvolume stats or body stats at this time.

Summary stats:
    * Number of splits required to get to threshold
    * Number of mergers required to get to threshold
"""
class edit_stat(StatType):
    def __init__(self, volthres = 90, splitfactor=5):
        """Init.

        Args:
            volthres (int): percent volume to be corrected
            splitfactor (float): split work to merge work ratio
        """
        super(edit_stat, self).__init__()
        self.volthres = volthres
        self.splitfactor = splitfactor
        self.supported_types = ["voxels", "synapse"]

    def write_summary_stats(self):
        """Write stats for the volume.

        Note: assignment is done by greatest overlap.

        Summary stats:
            * Number of splits to get to threshold
            * Number of mergers to get to threshold
        """
        summarystats = []

        # for each comparison type, find overlaps
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            body_threshold = self.segstats.ptfilter
            if gotable.get_comparison_type() == "voxels":
                body_threshold = self.segstats.voxelfilter
            sotable = self.segstats.seg_overlaps[onum]
        
            # prune large bodies
            ignore_bodies = set()
            for (gtbody, overlapset) in gotable.overlap_map.items():
                if self._get_body_volume(overlapset) < body_threshold:
                    # ignore as if it never existed
                    ignore_bodies.add(gtbody)

            # 1
            seg2bestgt = {}
            target = 0
            for body, overlapset in sotable.overlap_map.items():
                max_val = 0
                max_id = 0
                for body2, overlap in overlapset:
                    if body2 in ignore_bodies:
                        continue
                    target += overlap
                    if overlap > max_val:
                        max_val = overlap
                        max_id = body2
                seg2bestgt[body] = max_id
        
            target *= (self.volthres / 100.0)

            # 2, 3
            sorted_splits = []
            sorted_mergers = []
            current_accum = 0
            for body, overlapset in gotable.overlap_map.items():
                if body in ignore_bodies:
                    continue
                temp_merge = []
                for body2, overlap in overlapset:
                    # doesn't own body
                    if seg2bestgt[body2] != body:
                        sorted_splits.append(overlap)
                    else:
                        temp_merge.append(overlap)
                temp_merge.sort()
                temp_merge.reverse()
                if len(temp_merge) > 0:
                    current_accum += temp_merge[0]
                if len(temp_merge) > 1:
                    sorted_mergers.extend(temp_merge[1:])

            # sort lists for optimal traversal
            sorted_splits.sort()
            sorted_splits.reverse()
            sorted_mergers.sort()
            sorted_mergers.reverse()
            
            # greedily choose merges and splits based on
            # relative cost of split vs merge
            midx = 0
            sidx = 0
            current_accum_rat = current_accum

            while current_accum_rat < target:
                take_split = False
                if midx == len(sorted_mergers) and sidx == len(sorted_splits):
                    break

                if midx == len(sorted_mergers):
                    take_split = True
                elif sidx == len(sorted_splits):
                    pass
                elif (sorted_splits[sidx] / float(self.splitfactor)) > sorted_mergers[midx]:
                    take_split = True

                if take_split:
                    current_accum_rat += sorted_splits[sidx]
                    sidx += 1
                else:
                    current_accum_rat += sorted_mergers[midx]
                    midx += 1

            # a larger/smaller value is not obviously better or worse
            sumstat = {"name": "edit.splits", "typename": gotable.get_name(), "val": sidx, "higher-better": False}
            sumstat["description"] = "Number of split operations (split:merge cost = %0.2f and correct threshold = %d)" % (self.splitfactor, self.volthres)
            summarystats.append(sumstat)

            sumstat = {"name": "edit.merges", "typename": gotable.get_name(), "val": midx, "higher-better": False}
            sumstat["description"] = "Number of merge operations (split:merge cost = %0.2f and correct threshold = %d)" % (self.splitfactor, self.volthres)
            summarystats.append(sumstat)

        return summarystats

