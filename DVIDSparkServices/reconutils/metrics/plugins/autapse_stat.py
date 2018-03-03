from __future__ import division
import numpy
from .stat import *

"""Class provides autapse stats based on synapse annotations.

Note: assume that there is only one "synapse" type
"""
class autapse_stat(StatType):
    def __init__(self, thresholds = [1, 5, 10]):
        super(autapse_stat, self).__init__()
        self.thresholds = thresholds
        
    def write_summary_stats(self):
        """Write stats for the volume.
        """
        
        gotable, gtseg = self._retrieve_overlap_tables()
        typename = gotable.get_name()
        gtautapses, segautapses = self._extract_autapses(gtseg)
        sumstats = []

        gtcount = 0
        for gt, count in gtautapses.items():
            gtcount += count
        segcount = 0
        for seg, count in segautapses.items():
            segcount += count

        if self.segstats.selfcompare:
            # just provide GT counts
            sumstat = {"name": "num.autapse", "description": "num autapses", "higher-better": False, "typename": typename, "val": gtcount}
            sumstats.append(sumstat)

        else:
            # summary is seg - GT (should be close to 0, but more autapses is probably not a good if no real GT)
            sumstat = {"name": "num.autapsediff", "description": "num test - num gt autapses", "higher-better": False, "typename": typename, "val": segcount - gtcount}
            sumstats.append(sumstat)
        
        return sumstats
        
    def write_body_stats(self):
        """Write stats per body.
        """
        gotable, gtseg = self._retrieve_overlap_tables()
        typename = gotable.get_name()
        gtautapses, segautapses = self._extract_autapses(gtseg)
        bodystats = []
        
        num_displaybodies = self.segstats.num_displaybodies

        # grab worst bodies largest to smallest
        gtbodies = []
        for gt, count in gtautapses.items():
            gtbodies.append((count, gt))
        gtbodies.sort()
        gtbodies.reverse()
        gtbodies = gtbodies[0:num_displaybodies]
        
        
        bodystat = {"typename": typename, "name": "GT Autapse", "largest2smallest": True, "isgt": True}

        dgtbodies = {}
        for (val, bid) in gtbodies:
            dgtbodies[bid] = [val]

        bodystat["bodies"] = dgtbodies
        bodystats.append(bodystat)

        if not self.segstats.selfcompare:
            # add body stats for test volume
            segbodies = []
            for seg, count in segautapses.items():
                segbodies.append((count, seg))
            segbodies.sort()
            segbodies.reverse()
            segbodies = segbodies[0:num_displaybodies]
        
            bodystat = {"typename": typename, "name": "Seg Autapse", "largest2smallest": True, "isgt": False}
            dsegbodies = {}
            for (val, bid) in segbodies:
                dsegbodies[bid] = [val]
            bodystat["bodies"] = dsegbodies

            bodystats.append(bodystat)

        return bodystats
   
    def _extract_autapses(self, gtseg):
        gtautapses = {}
        segautapses = {}

        for pre, overlapset in gtseg.overlap_map.items():
            pre1 = int(pre >> 64)
            pre2 = int(pre & 0xffffffffffffffff)
            
            for (post, overlap) in overlapset:
                post1 = int(post >> 64)
                post2 = int(post & 0xffffffffffffffff)
                if pre1 == post1:
                    if post1 not in gtautapses:
                        gtautapses[post1] = 0
                    gtautapses[pre1] += overlap
                if pre2 == post2:
                    if post2 not in segautapses:
                        segautapses[post2] = 0
                    segautapses[post2] += overlap                  

        return gtautapses, segautapses

    def _retrieve_overlap_tables(self):
        gotable_main = None
        gtseg_table = None

        # grab overlap tables
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            if gotable.get_comparison_type() == "synapse":
                # only one synapse-graph allowed
                assert gotable_main is None
                gotable_main = gotable

            if gotable.get_comparison_type() == "synapse-graph-gtseg":
                # only one synapse-graph allowed
                assert gtseg_table is None
                gtseg_table = gotable

        assert gotable_main is not None
        assert gtseg_table is not None

        return gotable_main, gtseg_table

