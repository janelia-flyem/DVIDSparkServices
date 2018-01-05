from .stat import *
from math import log

"""Class provides stats based on Variation of Information.
"""
class vi_stat(StatType):
    def __init__(self):
        super(vi_stat, self).__init__()

        # subvolume state computed
        self.fmergebest = {}
        self.fmergeworst = {}
        self.fsplitbest = {}
        self.fsplitworst = {}
        self.fmergefsplitave = {}

        self.supported_types = ["voxels", "synapse"]

    def compute_subvolume_before_remapping(self):
        """Generates min/max/average VI state for a subvolume.

        Note:
            Disable if subvolume stats are turned off.
        """
        
        # disable if self compare
        if self.segstats.selfcompare:
            return
    
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
            fmerge, fsplit, dummy1, dummy2, dummy3 = self._calculate_vi(gotable, self.segstats.seg_overlaps[onum], True)

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
        
        # disable if self compare
        if self.segstats.selfcompare:
            return

        # use sum stats from other volume
        if self.segstats.ignore_subvolume:
            self.fmergebest = stat.fmergebest
            self.fmergeworst = stat.fmergeworst
            self.fsplitbest = stat.fsplitbest
            self.fsplitworst = stat.fsplitworst
            self.fmergefsplitave = stat.fmergefsplitave
            return
        if stat.segstats.ignore_subvolume:
            return

        for name, val1 in self.fmergebest.items():
            val2 = stat.fmergebest[name]
            if val2[0] < val1[0]:
                self.fmergebest[name] = val2

        for name, val1 in self.fmergeworst.items():
            val2 = stat.fmergeworst[name]
            if val2[0] > val1[0]:
                self.fmergeworst[name] = val2

        for name, val1 in self.fsplitbest.items():
            val2 = stat.fsplitbest[name]
            if val2[0] < val1[0]:
                self.fsplitbest[name] = val2

        for name, val1 in self.fsplitworst.items():
            val2 = stat.fsplitworst[name]
            if val2[0] > val1[0]:
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
        
        # disable if self compare
        if self.segstats.selfcompare:
            return []
        
        summarystats = []

        # get summary vi stats for subvolume
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            # add VI summary stats
            if gotable.get_comparison_type() not in self.supported_types:
                continue

            self._write_vi(summarystats, gotable, self.segstats.seg_overlaps[onum], True)

        return summarystats


    def write_summary_stats(self):
        """Write stats for the volume.

        Substack summary stats are only produced if substacks are enable.
        """
        
        # disable if self compare
        if self.segstats.selfcompare:
            return []
        
        summarystats = []

        # calculate summary and body stats
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            
            # add VI summary stats
            fmerge_bodies, fsplit_bodies, vi_bodies = self._write_vi(summarystats, gotable, self.segstats.seg_overlaps[onum])

            # get body summary stats
            self._calculate_bodystats(summarystats, None, gotable, fmerge_bodies, fsplit_bodies, vi_bodies) 

        if self.segstats.disable_subvolumes:
            return summarystats

        # generate subvolume stats
        for name, val1 in self.fmergebest.items():
            sumstat = {"name": "S-BEST-FM-VI", "higher-better": False, "typename": name, "val": round(val1[0], 4)}
            sumstat["description"] = "Best False Merge VI for a Subvolume. Subvolume=%d" % val1[1]
            summarystats.append(sumstat)

        for name, val1 in self.fmergeworst.items():
            sumstat = {"name": "S-WRST-FM-VI", "higher-better": False, "typename": name, "val": round(val1[0], 4)}
            sumstat["description"] = "Worst False Merge VI for a Subvolume. Subvolume=%d" % val1[1]
            summarystats.append(sumstat)

        for name, val1 in self.fsplitbest.items():
            sumstat = {"name": "S-BEST-FS-VI", "higher-better": False, "typename": name, "val": round(val1[0], 4)}
            sumstat["description"] = "Best False Split VI for a Subvolume. Subvolume=%d" % val1[1]
            summarystats.append(sumstat)

        for name, val1 in self.fsplitworst.items():
            sumstat = {"name": "S-WRST-FS-VI", "higher-better": False, "typename": name, "val": round(val1[0], 4)}
            sumstat["description"] = "Worst False Split VI for a Subvolume. Subvolume=%d" % val1[1]
            summarystats.append(sumstat)

        for name, val1 in self.fmergefsplitave.items():
            sumstat = {"name": "S-AVE-VI", "higher-better": False, "typename": name, "val": round(val1[0]+val1[1], 4)}
            sumstat["description"] = "Average Substack VI"
            summarystats.append(sumstat)

        return summarystats

    def write_body_stats(self):
        """Write body stats stats if available.
        """
        
        # disable if self compare
        if self.segstats.selfcompare:
            return []
        
        bodystats = []

        # calculate summary and body stats
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            dummy1, dummy2, fmerge_bodies, fsplit_bodies, vi_bodies = self._calculate_vi(gotable, self.segstats.seg_overlaps[onum])
 
            # load body stats for a given type
            self._calculate_bodystats(None, bodystats, gotable, fmerge_bodies, fsplit_bodies, vi_bodies) 

        return bodystats

 
    def _write_vi(self, summarystats, gotable, sotable, disablefilter=False):
        # restrict body sizes considered
        name = gotable.get_name()
        fmerge, fsplit, fmerge_bodies, fsplit_bodies, vi_bodies = self._calculate_vi(gotable, sotable, disablefilter)
         
        sumstat = {"name": "VI", "higher-better": False, "typename": name, "val": round(fmerge+fsplit, 4)}
        sumstat["description"] = "Total VI"
        summarystats.append(sumstat)

        sumstat = {"name": "FM-VI", "higher-better": False, "typename": name, "val": round(fmerge, 4)}
        sumstat["description"] = "False Merge VI"
        summarystats.append(sumstat)

        sumstat = {"name": "FS-VI", "higher-better": False, "typename": name, "val": round(fsplit, 4)}
        sumstat["description"] = "False Split VI"
        summarystats.append(sumstat)

        return fmerge_bodies, fsplit_bodies, vi_bodies


    def _calculate_vi(self, gtoverlap, segoverlap, disablefilter=False):
        """Caculate variation of information metric using overlap tables.
        """
        body_threshold = self.segstats.ptfilter
        if gtoverlap.get_comparison_type() == "voxels":
            body_threshold = self.segstats.voxelfilter
        
        if disablefilter:
            body_threshold = 0

        fsplit_bodies = {}
        fmerge_bodies = {}

        # determine how bad a given body is
        perbody = {}

        glb_total = 0
        fmerge_vi = 0
        fsplit_vi = 0

        ignore_bodies = set()

        # examine fragmentation of gt (fsplit=oversegmentation)
        # filtering bodies will bias things to oversegmentation errors
        # since we are ignoring small errors in GT which would have
        # otherwise been mostly good in the other segmentation -- this
        # is appropriate since the filtering is to handle inacuracies
        # in the voxel ground truth -- needing filtering might result in
        # less useable total VI body metrics since the GT is sparser
        for (gtbody, overlapset) in gtoverlap.overlap_map.items():
            if self._get_body_volume(overlapset) < body_threshold:
                # ignore as if it never existed
                ignore_bodies.add(gtbody)
                continue
            # ignore bodies in sparse mode 
            if self.segstats.enable_sparse:
                if gtbody not in self.segstats.important_bodies:
                    continue

            vi_unnorm, total, dummy = self._body_vi(overlapset)
            fsplit_bodies[gtbody] = vi_unnorm
            perbody[gtbody] = vi_unnorm
            glb_total += total
            fsplit_vi += vi_unnorm

        # examine fragmentation of seg (fmerge=undersegmentation)
        for (segbody, overlapset) in segoverlap.overlap_map.items():
            # filter small bodies
            filtered_overlapset = set()
            for (gtbody, overlap) in overlapset:
                if gtbody not in ignore_bodies:
                    filtered_overlapset.add((gtbody, overlap))

            vi_unnorm, total, gtcontribs = self._body_vi(filtered_overlapset)
            fmerge_bodies[segbody] = vi_unnorm
            fmerge_vi += vi_unnorm

            for key, val in gtcontribs.items():
                if key in perbody:
                    perbody[key] += val
        
        # TODO !! Add per body
        if glb_total == 0:
            return 0, 0, fmerge_bodies, fsplit_bodies, perbody

        for key, val in fsplit_bodies.items():
            fsplit_bodies[key] = round(val / float(glb_total),4)
        
        for key, val in fmerge_bodies.items():
            fmerge_bodies[key] = round(val / float(glb_total),4)
       
        # calculate total vi normalized by filtered bodies in sparse bmde
        totalvi_perbody = 0
        for key, val in perbody.items():
            perbody[key] = round(val / float(glb_total),4)
            totalvi_perbody += val

        if self.segstats.enable_sparse:
            # use re-normlized vi for sparse mode
            return round((totalvi_perbody - fsplit_vi) / float(glb_total), 4), round(fsplit_vi / float(glb_total),4), fmerge_bodies, fsplit_bodies, perbody 
        else:
            return round(fmerge_vi / float(glb_total),4), round(fsplit_vi / float(glb_total),4), fmerge_bodies, fsplit_bodies, perbody 


    def _body_vi(self, overlapset):
        total = 0
        for (segid, overlap) in overlapset:
            total += overlap

        decomp_bodies = {}
        vi_unnorm = 0
        for (segid, overlap) in overlapset:
            vi_unnorm += overlap*log(total / float(overlap))/log(2.0)
            if segid not in decomp_bodies:
                decomp_bodies[segid] = 0
            decomp_bodies[segid] += overlap*log(total / float(overlap))/log(2.0)

        return vi_unnorm, total, decomp_bodies

    def _calculate_bodystats(self, summarystats, bodystats, gtoverlap, fmerge_bodies, fsplit_bodies, vi_bodies):
        """Calculates body summary stats and per body stats for a given overlap.
        """
        # do not report smallest bodies 
        body_threshold = self.segstats.ptfilter
        if gtoverlap.get_comparison_type() == "voxels":
            body_threshold = self.segstats.voxelfilter
        
        # limit number of bodies to examine
        num_displaybodies = self.segstats.num_displaybodies

        # examine gt bodies
        worst_gt_body = 0 
        worst_gt_val = 0
        worst_fsplit = 0
        worst_fsplit_body = 0
        
        # stats for worst vi and worst fsplit
        bodystat = {"typename": gtoverlap.get_name(), "name": "Worst GT", "largest2smallest": True, "isgt": True}
        bodystat2 = {"typename": gtoverlap.get_name(), "name": "GT Frag", "largest2smallest": True, "isgt": True}
        bodies1 = []
        bodies2 = []

        for body, overlapset in gtoverlap.overlap_map.items():
            total = 0
            for body2, overlap in overlapset:
                total += overlap
            if total < body_threshold:
                continue
            if self.segstats.enable_sparse:
                if body not in self.segstats.important_bodies:
                    continue
            fsplit = fsplit_bodies[body]
            vitot = vi_bodies[body]

            # add body
            bodies1.append((vitot, body))
            bodies2.append((fsplit, body))

            if fsplit > worst_fsplit:
                worst_fsplit = fsplit
                worst_fsplit_body = body
            if vitot > worst_gt_val:
                worst_gt_val = vitot
                worst_gt_body = body


        # examine seg bodies
        worst_fmerge = 0
        worst_fmerge_body = 0

        # stats for worst vi and worst fsplit
        
        bodystat3 = {"typename": gtoverlap.get_name(), "name": "Test Frag", "largest2smallest": True, "isgt": False}
        bodies3 = []
        
        #for body, overlapset in seg_overlap.overlap_map.items():
        for body, fmerge in fmerge_bodies.items():
            # ignore body size filter since results are already properly
            # filted and the display cut-off only takes biggest errors
            fmerge = fmerge_bodies[body]
            bodies3.append((fmerge, body))

            if fmerge > worst_fmerge:
                worst_fmerge = fmerge
                worst_fmerge_body = body
        
        # add body stats
        if bodystats is not None:
            # restrict number of bodies to top num_displaybodies
            bodies1.sort()
            bodies2.sort()
            bodies3.sort()
            bodies1.reverse()
            bodies2.reverse()
            bodies3.reverse()
           
            bodies1 = bodies1[0:num_displaybodies]
            bodies2 = bodies2[0:num_displaybodies]
            bodies3 = bodies3[0:num_displaybodies]

            dbodies1 = {}
            dbodies2 = {}
            dbodies3 = {}

            for (val, bid) in bodies1:
                dbodies1[bid] = [val]
            for (val, bid) in bodies2:
                dbodies2[bid] = [val]
            for (val, bid) in bodies3:
                dbodies3[bid] = [val]

            # add body stats
            # disable Worst GT VI if non-gt mode
            if not self.segstats.nogt:
                bodystat["bodies"] = dbodies1
                bodystats.append(bodystat)
            
            bodystat2["bodies"] = dbodies2
            bodystat3["bodies"] = dbodies3
            bodystats.append(bodystat2)
        
            # disable body stats for false merge for sparse
            if not self.segstats.enable_sparse:
                bodystats.append(bodystat3)
        
        # body summary stats
        if summarystats is not None:
            # disable Worst GT VI if non-gt mode
            if not self.segstats.nogt:
                sumstat = {"name": "B-WRST-GT-VI", "higher-better": False, "typename": gtoverlap.get_name(), "val": round(worst_gt_val, 4)}
                sumstat["description"] = "Worst body VI. GT body ID = %d" % worst_gt_body
                summarystats.append(sumstat)

            sumstat = {"name": "B-WRST-GT-FR", "higher-better": False, "typename": gtoverlap.get_name(), "val": round(worst_fsplit, 4)}
            sumstat["description"] = "Worst body fragmentation VI. GT body ID = %d" % worst_fsplit_body
            summarystats.append(sumstat)

            # disable body stats for false merge for sparse
            if not self.segstats.enable_sparse:
                sumstat = {"name": "B-WRST-TEST-FR", "higher-better": False, "typename": gtoverlap.get_name(), "val": round(worst_fmerge, 4)}
                sumstat["description"] = "Worst body fragmentation VI. Test body ID = %d" % worst_fmerge_body
                summarystats.append(sumstat)
 

