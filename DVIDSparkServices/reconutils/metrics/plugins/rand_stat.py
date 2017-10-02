from .stat import *

"""Class provides stats based on Rand index.

Note: does not implement stats per body.

"""
class rand_stat(StatType):
    def __init__(self):
        super(rand_stat, self).__init__()

        # subvolume state computed
        self.fmergebest = {}
        self.fmergeworst = {}
        self.fsplitbest = {}
        self.fsplitworst = {}
        self.fmergefsplitave = {}

        self.supported_types = ["voxels", "synapse"]

    def compute_subvolume_before_remapping(self):
        """Generates min/max/average Rand state for a subvolume.

        Note:
            Disable if subvolume stats are turned off.
        """
    
        # should not compute state if multiple subvolumes
        assert len(self.segstats.subvolumes) <= 1

        if self.segstats.disable_subvolumes:
            return

        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            # does not work with synapse connectivity overlap
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            assert gotable.get_name() == self.segstats.seg_overlaps[onum].get_name()

            # restrict body sizes considered
            fmerge, fsplit = self._calculate_rand(gotable, self.segstats.seg_overlaps[onum], True)

            name = gotable.get_name()
            sid = self.segstats.subvolumes[0].sv_index
            self.fmergebest[name] = [fmerge, sid]
            self.fsplitbest[name] = [fsplit, sid]
            self.fmergeworst[name] = [fmerge, sid]
            self.fsplitworst[name] = [fsplit, sid]
            self.fmergefsplitave[name] = [fmerge, fsplit]

    def reduce_subvolume(self, stat):
        """Combine subvolume stats.
        """

        for name, val1 in self.fmergebest.items():
            val2 = stat.fmergebest[name]
            if val2[0] > val1[0]:
                self.fmergebest[name] = val2

        for name, val1 in self.fmergeworst.items():
            val2 = stat.fmergeworst[name]
            if val2[0] < val1[0]:
                self.fmergeworst[name] = val2

        for name, val1 in self.fsplitbest.items():
            val2 = stat.fsplitbest[name]
            if val2[0] > val1[0]:
                self.fsplitbest[name] = val2

        for name, val1 in self.fsplitworst.items():
            val2 = stat.fsplitworst[name]
            if val2[0] < val1[0]:
                self.fsplitworst[name] = val2

        num1 = len(self.segstats.subvolumes)
        num2 = len(stat.segstats.subvolumes)
        for name, val1 in self.fmergefsplitave.items():
            val2 = stat.fmergefsplitave[name]
            fmerge = (val1[0]*num1 + val2[0]*num2) / float(num1+num2)
            fsplit = (val1[1]*num1 + val2[1]*num2) / float(num1+num2)
            self.fmergefsplitave[name] = [fmerge, fsplit]

    def write_subvolume_stats(self):
        """Write subvolume summary stats for the subvolume.
        """
        summarystats = []

        # get summary rand stats for subvolume
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            # add rand summary stats
            self._write_rand(summarystats, gotable, self.segstats.seg_overlaps[onum], True)

        return summarystats


    def write_summary_stats(self):
        """Write stats for the volume.

        Substack summary stats are only produced if substacks are enable.
        """
        summarystats = []

        # calculate summary and body stats
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            
            # add rand summary stats
            self._write_rand(summarystats, gotable, self.segstats.seg_overlaps[onum])

        if self.segstats.disable_subvolumes:
            return summarystats

        # generate subvolume stats
        for name, val1 in self.fmergebest.items():
            sumstat = {"name": "S-BEST-FM-RD", "higher-better": True, "typename": name, "val": val1[0]}
            sumstat["description"] = "Best False Merge Rand for a Subvolume. Subvolume=%d" % val1[1]
            summarystats.append(sumstat)

        for name, val1 in self.fmergeworst.items():
            sumstat = {"name": "S-WRST-FM-RD", "higher-better": True, "typename": name, "val": val1[0]}
            sumstat["description"] = "Worst False Merge Rand for a Subvolume. Subvolume=%d" % val1[1]
            summarystats.append(sumstat)

        for name, val1 in self.fsplitbest.items():
            sumstat = {"name": "S-BEST-FS-RD", "higher-better": True, "typename": name, "val": val1[0]}
            sumstat["description"] = "Best False Split Rand for a Subvolume. Subvolume=%d" % val1[1]
            summarystats.append(sumstat)

        for name, val1 in self.fsplitworst.items():
            sumstat = {"name": "S-WRST-FS-RD", "higher-better": True, "typename": name, "val": val1[0]}
            sumstat["description"] = "Worst False Split Rand for a Subvolume. Subvolume=%d" % val1[1]
            summarystats.append(sumstat)

        for name, val1 in self.fmergefsplitave.items():
            sumstat = {"name": "S-AVE-RD", "higher-better": True, "typename": name, "val": 2*(val1[0]*val1[1])/(val1[0]+val1[1])}
            sumstat["description"] = "Average Substack Rand"
            summarystats.append(sumstat)

        return summarystats

    def _write_rand(self, summarystats, gotable, sotable, disablefilter=False):
        # restrict body sizes considered
        fmerge, fsplit = self._calculate_rand(gotable, sotable, disablefilter)
        name = gotable.get_name()

        sumstat = {"name": "Rand", "higher-better": True, "typename": name, "val": 2*fmerge*fsplit/(fmerge+fsplit)}
        sumstat["description"] = "Rand"
        summarystats.append(sumstat)

        sumstat = {"name": "FM-RD", "higher-better": True, "typename": name, "val": fmerge}
        sumstat["description"] = "False Merge Rand"
        summarystats.append(sumstat)

        sumstat = {"name": "FS-RD", "higher-better": True, "typename": name, "val": fsplit}
        sumstat["description"] = "False Split Rand"
        summarystats.append(sumstat)

        return

    # calculate Rand Index
    def _calculate_rand(self, gtoverlap, segoverlap, disablefilter=False):
        """Caculate rand index using overlap tables.
        """
        body_threshold = self.segstats.ptfilter
        if gtoverlap.get_comparison_type() == "voxels":
            body_threshold = self.segstats.voxelfilter
        
        if disablefilter:
            body_threshold = 0

        fsplit_total = 0
        overlap_total = 0
        ignore_bodies = set()
        
        # examine fragmentation of gt (fsplit=oversegmentation)
        for (gtbody, overlapset) in gtoverlap.overlap_map.items():
            if self._get_body_volume(overlapset) < body_threshold:
                # ignore as if it never existed
                ignore_bodies.add(gtbody)
                continue
            total = 0
            for (segid, overlap) in overlapset:
                total += overlap
                overlap_total += (overlap*(overlap-1)/2)

            fsplit_total += (total*(total-1)/2)

        fmerge_total = 0
        # examine fragmentation of seg (fmerge=undersegmentation)
        for (segbody, overlapset) in segoverlap.overlap_map.items():
            # filter small bodies
            filtered_overlapset = set()
            for (gtbody, overlap) in overlapset:
                if gtbody not in ignore_bodies:
                    filtered_overlapset.add((gtbody, overlap))
      
            total = 0
            for (segid, overlap) in filtered_overlapset:
                total += overlap

            fmerge_total += (total*(total-1)/2)

        merge = 1
        split = 1

        if fmerge_total != 0:
            merge  = round(overlap_total / float(fmerge_total),4)
        if  fsplit_total != 0:
            split = round(overlap_total / float(fsplit_total),4)
        return merge, split


