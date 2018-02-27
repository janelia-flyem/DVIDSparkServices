from .stat import *
import math

"""Class finds orphan bodies in volume.
"""
class orphans_stat(StatType):
    def __init__(self):
        super(orphans_stat, self).__init__()
        self.supported_types = ["voxels", "synapse"]

    @staticmethod 
    def iscustom_workflow():
        """Indicates whether plugin requires a specialized workflow.
        """
        return True
    
    # special passes for orphan counts per roi
    # (first pass to get total count and filter, second pass for substack stats)
    def custom_workflow(self, segroichunks_rdd):
        # extract segstats rdd
        allstats = segroichunks_rdd.map(lambda x: x[1][0]) 
        allstats.persist()

        # get summary stats
        
        def combinestats(stat1, stat2):
            stat1.merge_stats(stat2, False)
            return stat1
        segstats_accum = allstats.treeReduce(combinestats, int(math.log(allstats.getNumPartitions(),2)))

        # should not be called in sparse mode
        summary_stats = []
        if len(segstats_accum.important_bodies) > 0:
            return summary_stats, [], {}

        bigbodies_gt = []
        bigbodies_seg = []
        for onum, gotable in enumerate(segstats_accum.gt_overlaps):
            if gotable.get_comparison_type() not in self.supported_types:
                continue
            # filter that determines when something is an orphan
            body_threshold = segstats_accum.ptfilter
            if gotable.get_comparison_type() == "voxels":
                body_threshold = segstats_accum.voxelfilter

            # gt big bodies and small body count
            bigbodies = set()
            segbigbodies = set()
            smallbodies = 0
    
            for gt, overlapset in gotable.overlap_map.items():
                total = 0
                for seg, overlap in overlapset:
                    total += overlap
                if total >= body_threshold:
                    bigbodies.add(gt)
                elif gt in segstats_accum.boundarybodies:
                    # completely ignore small bodies on boundary
                    pass
                else:
                    smallbodies += 1

            # if not self mode -- get seg stats
            if not segstats_accum.selfcompare:
                sotable = segstats_accum.seg_overlaps[onum]
                segsmallbodies = 0
     
                for seg, overlapset in sotable.overlap_map.items():
                    total = 0
                    for gt, overlap in overlapset:
                        if gt in bigbodies:
                            total += overlap
                    if total >= body_threshold:
                        segbigbodies.add(seg)
                    elif seg in segstats_accum.boundarybodies2:
                        # completely ignore small bodies on boundary
                        pass
                    else:
                        segsmallbodies += 1
                
                # ideally there would be no orphans in the GT
                sumstat = {"name": "orphans", "typename": gotable.get_name(), "val": segsmallbodies - smallbodies, "higher-better": False}
                sumstat["description"] = "%d seg orphans, %d gt orphans" % (segsmallbodies, smallbodies)                 
                summary_stats.append(sumstat)

            else:
                sumstat = {"name": "orphans", "typename": gotable.get_name(), "val": smallbodies, "higher-better": False}
                sumstat["description"] = "%d orphans" % (smallbodies)                 
                summary_stats.append(sumstat)
            
            bigbodies_gt.append(bigbodies)
            bigbodies_seg.append(segbigbodies)
        
        # roadcast big bodies, another pass on subvolumes
        def count_suborphans(stat):
            summary_stats = []
            subval = {}
            for onum, gotable in enumerate(stat.gt_overlaps):
                if gotable.get_comparison_type() not in self.supported_types:
                    continue
                # create subvolume stat
                # (hide if small)
                num_gtorphans = 0
                
                subvolumesize = 0
                for gt, overlapset in gotable.overlap_map.items():
                    total = self._get_body_volume(overlapset)
                    if gt not in bigbodies_gt and gt not in stat.boundarybodies:
                        # small bodies not touching the edge
                        num_gtorphans += 1
                        subvolumesize += total
                    if gt in bigbodies_gt:
                        subvolumesize += total
      
                # filter small subvolumes (remove boundary orphans)
                ignoresubvolume = False
                if stat.subvolume_threshold > float(subvolumesize / stat.subvolsize):
                    ignoresubvolume = True    


                if not stat.selfcompare:
                    num_segorphans = 0
                    sotable = stat.seg_overlaps[onum]
                    for seg, overlapset in sotable.overlap_map.items():
                        if seg not in bigbodies_seg and seg not in stat.boundarybodies2:
                            # small bodies not touching the edge
                            num_segorphans += 1
                    sumstat = {"name": "orphans", "typename": gotable.get_name(), "val": num_segorphans-num_gtorphans, "higher-better": False, "ignore": ignoresubvolume}
                    sumstat["description"] = "%d seg orphans, %d gt orphans" % (num_segorphans, num_gtorphans)                 
                    summary_stats.append(sumstat)
                    if not stat.ignore_subvolume:
                        subval[gotable.get_name()] = num_segorphans-num_gtorphans
                else:
                    sumstat = {"name": "orphans", "typename": gotable.get_name(), "val": num_gtorphans, "higher-better": False, "ignore": stat.ignore_subvolume}
                    sumstat["description"] = "%d gt orphans" % (num_gtorphans)                 
                    summary_stats.append(sumstat)
                    if not stat.ignore_subvolume:
                        subval[gotable.get_name()] = num_gtorphans
            
            return (stat.subvolumes[0].sv_index, subval, summary_stats)

        subvolres = allstats.map(count_suborphans).collect()

        # load subvol stats
        # add largest, smallest value to substack summary
        subvol_metrics = {}
        maxvals = {}
        minvals = {} 
        for (sid, subval, sumstats) in subvolres:
            subvol_metrics[sid] = sumstats 
            for typename, val in subval.items():
                val2 = -999999999999
                if typename in maxvals:
                    (sidold, val2) = maxvals[typename]
                if val > val2:
                    maxvals[typename] = (sid, val)
                val2 = 999999999999
                if typename in minvals:
                    (sidold, val2) = minvals[typename]
                if val < val2:
                    minvals[typename] = (sid, val)
                        
        for typename, (sid, val) in maxvals.items():
            sumstat = {"name": "S-WRST-orphans", "typename": typename, "val": val, "higher-better": False}
            sumstat["description"] = "Num orphans in worst substack (Subvolume=%d)" % sid 
            summary_stats.append(sumstat)
        for typename, (sid, val) in minvals.items():
            sumstat = {"name": "S-BEST-orphans", "typename": typename, "val": val, "higher-better": False}
            sumstat["description"] = "Num ophans in best substack (Subvolume=%d)" % sid 
            summary_stats.append(sumstat)

        return summary_stats, [], subvol_metrics
