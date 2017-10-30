from .stat import *

"""Class provides stats based on simple voxel/point overlap.

Body stats:
    * (Qualifying) Test bodies with largest overlap with ground truth
    * Largest GT bodies
    * Largest seg bodies

Summary stats:
    * Largest correct body overlap
    * +Number of  bodies that make up X% of the volume compared to GT
    (provides a lower-bound on the number of edits required -- edit distance
    metrics are more generally more specific.)

Substack stats:
    * Number of seg, gt bodies (currently configured to not display in the viewer)

Debug stats include:
    * Top body overlap for the largest seg/gt bodies
    * Size of the whole volume

TODO: better formalize histogram and best test stats
(hungarian matching might not add value for just getting top bodies)

"""
class count_stat(StatType):
    def __init__(self, thresholds = [50, 75], debugthreshold=90):
        super(count_stat, self).__init__()

        self.thresholds = thresholds
        
        # maximum overlap matches considered
        self.nummatches = 10 
        
        # pecentage threshold for showing bodies
        self.debugthreshold = debugthreshold

        self.supported_types = ["voxels", "synapse"]

    def write_subvolume_stats(self):
        """Write subvolume summary stats for the subvolume.
        """
        summarystats = []

        # get summary rand stats for subvolume
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            # add rand summary stats
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            self._write_subcount(summarystats, gotable, self.segstats.seg_overlaps[onum], True)

        return summarystats

    def write_summary_stats(self):
        """Write stats for the volume.
        
        Summary stats:
            * Largest correcy body overlap
            * Number of additional bodies that make up thresholds of the volume
             
        """
        summarystats = []

        # calculate summary stats from bodies
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            # get body summary stats
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            self._calculate_bodystats(summarystats, None, gotable, self.segstats.seg_overlaps[onum]) 

        return summarystats


    def write_body_stats(self):
        """Write stats for each body.

        Body stats:
            * Best test segmentation in terms of overlap
            * Largest GT bodies
            * Largest test bodies
        """

        bodystats = []

        # calculate body stats
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            # get body summary stats
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            self._calculate_bodystats(None, bodystats, gotable, self.segstats.seg_overlaps[onum]) 

        return bodystats


    def write_bodydebug(self):
        """Generate overlap information.
        """
        debuginfo = []
       
        # for each comparison type, find overlaps
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            sotable = self.segstats.seg_overlaps[onum]
        
            # grab seg and gt dists limited by num display bodies
            gtdist, gtcount, ignorebodies = self._extract_distribution(gotable)
            segdist, segcount, ignorebodies2 = self._extract_distribution(sotable, ignorebodies)
            gtdist = gtdist[0:self.segstats.num_displaybodies]
            segdist = segdist[0:self.segstats.num_displaybodies]

            # find bodies that overlap with gt
            top_overlapgt = {}
            nummatches = self.nummatches # find only up to 10 matches
           
            # examine bodies <= threshold
            curramt = 0
            for distiter in range(0, len(gtdist)):
                curramt += gtdist[distiter][0]/float(gtcount)*100.0
                if curramt > self.debugthreshold:
                    break
                overlapmap = gotable.overlap_map[gtdist[distiter][1]]
                matchlist = []
                for (body2, overlap) in overlapmap:
                    matchlist.append([overlap, body2])
                matchlist.sort()
                matchlist.reverse()
                top_overlapgt[gtdist[distiter][1]] = matchlist[0:nummatches]
            
            debuginfo1 = {"typename": gotable.get_name(), "name": "gtoverlap"}
            debuginfo1["info"] = top_overlapgt
           
            # find bodies that overlap with test seg
            top_overlapseg = {}

            # examine bodies <= threshold
            curramt = 0
            for distiter in range(0, len(segdist)):
                curramt += segdist[distiter][0]/float(gtcount)*100.0
                if curramt > self.debugthreshold:
                    break
                overlapmap = sotable.overlap_map[segdist[distiter][1]]
                matchlist = []
                for (body2, overlap) in overlapmap:
                    if body2 in ignorebodies:
                        continue
                    matchlist.append([overlap, body2])
                matchlist.sort()
                matchlist.reverse()
                top_overlapseg[segdist[distiter][1]] = matchlist[0:nummatches]
            
            debuginfo2 = {"typename": gotable.get_name(), "name": "segoverlap"}
            debuginfo2["info"] = top_overlapseg

            debuginfo.append(debuginfo1)
            debuginfo.append(debuginfo2)

        return debuginfo

    def _write_subcount(self, summarystats, gotable, sotable, disablefilter):
        """Find the number of bodies for the volume.

        Note: The stat is configured to not be displayed.
        """
        body_threshold = self.segstats.ptfilter
        if gotable.get_comparison_type() == "voxels":
            body_threshold = self.segstats.voxelfilter
        
        if disablefilter:
            body_threshold = 0
        
        gtbodies = 0
        segbodies = 0
        ignorebodies = set()
        for (gtbody, overlapset) in gotable.overlap_map.items():
            if self._get_body_volume(overlapset) >= body_threshold:
                gtbodies += 1
            else:
                ignorebodies.add(gtbody)
       
        for (segbody, overlapset) in sotable.overlap_map.items():
            # filter is only applied to GT (as if bodies don't exist)
            # if a body no longer has volume, we can safely ignore
            if self._get_body_volume(overlapset, ignorebodies) >= 0:
                segbodies += 1 

        # a larger/smaller value is not obviously better or worse
        name = gotable.get_name()
        sumstat = {"name": "num.gt.bodies", "typename": name, "val": gtbodies, "display": False}
        sumstat["description"] = "Number of GT bodies"
        summarystats.append(sumstat)

        sumstat = {"name": "num.seg.bodies", "typename": name, "val": segbodies, "display": False}
        sumstat["description"] = "Number of seg bodies"
        summarystats.append(sumstat)

    def _extract_distribution(self, body_overlap, ignorebodies=None):
        """Helper function: extracts sorted list of body sizes.

        Args:
            body_overlap (dict): body to set(body2, overlap)
        Returns:
            sorted body size list (largest to smallest)
        """

        body_threshold = self.segstats.ptfilter
        if body_overlap.get_comparison_type() == "voxels":
            body_threshold = self.segstats.voxelfilter
 
        count = 0
        cumdisttemp = []
        ignorebodies_temp = set()
        for body, overlapset in body_overlap.overlap_map.items():
            # could be 0
            localcount = self._get_body_volume(overlapset, ignorebodies) 
            if localcount == 0:
                continue
            count += localcount

            # ignore small bodies if ignorebodies is not already set
            if ignorebodies is None:
                if localcount < body_threshold:
                    ignorebodies_temp.add(body)
                    continue

            cumdisttemp.append([localcount, body])

        cumdisttemp.sort()
        cumdisttemp.reverse()
        return cumdisttemp, count, ignorebodies_temp



    def _calculate_bodystats(self, summarystats, bodystats, gotable, sotable):
        body_threshold = self.segstats.ptfilter
        if gotable.get_comparison_type() == "voxels":
            body_threshold = self.segstats.voxelfilter
            
        # limit number of bodies to examine
        num_displaybodies = self.segstats.num_displaybodies

        # candidate bodies must have >50% in the GT body to be considered
        # hungarian matching would probably be overkill
        important_segbodies = {}
        for gt, overlapset in gotable.overlap_map.items():
            total = 0
            max_id = 0
            max_val = 0
            for seg, overlap in overlapset:
                total += overlap
                if overlap > max_val:
                    max_val = overlap
                    max_id = seg
            # find size of seg body
            total2 = 0
            overlapset2 = sotable.overlap_map[max_id]
            for seg2, overlap2 in overlapset2:
                total2 += overlap2
            
            # match body if over half of the seg
            if max_val > (total2 // 2):
                important_segbodies[max_id] = max_val
      
        # add dist bodies
        bodytest_diststat = {"typename": gotable.get_name(), "name": "Largest Test", "largest2smallest": True, "isgt": False}
        bodygt_diststat = {"typename": gotable.get_name(), "name": "Largest GT", "largest2smallest": True, "isgt": True}

        segbodies = {}
        gtbodies = {}
        
        bodytest_diststat["bodies"] = segbodies
        bodygt_diststat["bodies"] = gtbodies
   
        # grab filtered bodies from largest to smallest and the total size 
        gtdist, gtcount, ignorebodies = self._extract_distribution(gotable)
        segdist, segcount, ignorebodies2 = self._extract_distribution(sotable, ignorebodies)

        # iterate top X for each and add to diststat
        for index, val in enumerate(gtdist):
            if index == num_displaybodies:
                break
            gtbodies[val[1]] = [val[0], [val[0]/float(gtcount)*100.0]]
        for index, val in enumerate(segdist):
            if index == num_displaybodies:
                break
            segbodies[val[1]] = [val[0], [val[0]/float(gtcount)*100.0]]
        
        # add max overlap body stats
        bodystat = {"typename": gotable.get_name(), "name": "Best Test", "largest2smallest": True, "isgt": False}
        bodies1 = []

        # examine seg bodies
        best_size = 0
        best_body_id = 0

        for body, overlapset in sotable.overlap_map.items():
            total = self._get_body_volume(overlapset, ignorebodies)
            
            # probably okay to skip these bodies since
            # since the match volume would be less than
            # what is allowable for GT bodies
            if total < body_threshold:
                continue

            maxoverlap = 0
            if body in important_segbodies:
                maxoverlap = important_segbodies[body]

            if maxoverlap > best_size:
                best_size = maxoverlap
                best_body_id = body
            bodies1.append((maxoverlap, body, total))

        # add body stat
        if bodystats is not None:
            # restrict number of bodies to top num_displaybodies
            bodies1.sort()
            bodies1.reverse()
            bodies1 = bodies1[0:num_displaybodies]
            dbodies1 = {}
            for (val, bid, total) in bodies1:
                dbodies1[bid] = [val, [total]]

            # add body stats
            bodystat["bodies"] = dbodies1
            bodystats.append(bodystat)
            
            bodystats.append(bodytest_diststat)
            bodystats.append(bodygt_diststat)
       
        # add summary stat
        if summarystats is not None:

            # find #body count diff at threshold and put more info in description 
            if self.thresholds is not None:
                numgtbodies = []
                numsegbodies = []
                
                # grab number of gt bodies at each threshold 
                thresholds = self.thresholds.copy()
                thresholds.sort()
                curr_thres = thresholds[0]
                thresholds = thresholds[1:]
                curramt = 0
                for index, val in enumerate(gtdist):
                    curramt += val[1]/float(gtcount)*100.0
                    if curramt >= curr_thres:
                        numgtbodies.append(index+1)         
                        if len(thresholds) > 0:
                            curr_thres = thresholds[0]
                            thresholds = thresholds[1:]
                        else:
                            break
                # if threshold is unreachable just put all bodies
                while len(numgtbodies) < len(self.thresholds):
                    numgtbodies.append(len(gtdist))
            
                # grab number of seg bodies at each threshold 
                thresholds = self.thresholds.copy()
                thresholds.sort()
                curr_thres = thresholds[0]
                thresholds = thresholds[1:]
                curramt = 0
                for index, val in enumerate(segdist):
                    curramt += val[1]/float(gtcount)*100.0
                    if curramt >= curr_thres:
                        numsegbodies.append(index+1)         
                        if len(thresholds) > 0:
                            curr_thres = thresholds[0]
                            thresholds = thresholds[1:]
                        else:
                            break
                # if threshold is unreachable just put all bodies
                while len(numsegbodies) < len(self.thresholds):
                    numsegbodies.append(len(segdist))

                # write stat for each histogram threshold
                thresholds = self.thresholds.copy()
                thresholds.sort()
                for iter1, threshold in enumerate(thresholds):
                    sumstat = {"name": "HIST-%d"%threshold, "higher-better": False, "typename": gotable.get_name(), "val": abs(numgtbodies[iter1]-numsegbodies[iter1])}
                    sumstat["description"] = "#Body difference between GT (%d) and test segmentation (%d) for %d percent of volume" % (numgtbodies[iter1], numsegbodies[iter1], threshold) 
                    summarystats.append(sumstat)

            # write body summary stat
            sumstat = {"name": "B-BST-TST-OV", "higher-better": True, "typename": gotable.get_name(), "val": best_size}
            sumstat["description"] = "Test segment with greatest (best) correct overlap.  Test body ID = %d" % best_body_id
            summarystats.append(sumstat)

        


