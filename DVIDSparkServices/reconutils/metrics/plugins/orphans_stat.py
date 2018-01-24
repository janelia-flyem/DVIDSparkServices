from .stat import *

"""Class finds orphan bodies in volume.
"""
class orphans_stat(StatType):
    def __init__(self):
        super(orphans_stat, self).__init__()

    @static_method 
    def iscustom_workflow():
        """Indicates whether plugin requires a specialized workflow.
        """
        return True
    
    # special passes for orphan counts per roi
    # (first pass to get total count and filter, second pass for substack stats)
    def custom_workflow(segroichunks_rdd):
        # extract segstats rdd
        allstats = segroichunks_rdd.map(lambda x: x[1][0]) 
        allstats.persist()

        # get summary stats
        
        def combinestats(stat1, stat2):
            stat1.merge_stats(stat2, False)
            return stat1
        segstats_accum = allstats.treeReduce(combinestats, int(math.log(len(allstats.getNumPartitions()),2)))

        # should not be called in sparse mode
        summary_stats = []
        if len(segstats_accum.important_bodies) > 0:
            return summary_stats, [], {}


        # ?! need segstat to keep track of bodies that leave (each subvolume can access ROI, make a mask 32 pixels larger than subvolume -- erode mask by one pixel and find bodies that touch global boundary)
        # ?! need to create ignore subvolume with orphans accounted for in Evaluate
        # ?! if body is smaller than threshold but touches global, filter out

        bigbodies_gt = []
        bigbodies_seg = []
        for onum, gotable in enumerate(segstats_accum.gt_overlaps):
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
                # create subvolume stat
                # (hide if small)
                num_gtorphans = 0
                for gt, overlapset in gotable.overlap_map.items():
                    if gt not in bigbodies_gt:
                        num_gtorphans += 1
           
                if not segstats_accum.selfcompare:
                    num_segorphans = 0
                    for seg, overlapset in sotable.overlap_map.items():
                        num_segorphans += 1
                    sumstat = {"name": "orphans", "typename": gotable.get_name(), "val": num_segorphans-num_gtorphans, "higher-better": False}
                    sumstat["description"] = "%d seg orphans, %d gt orphans" % (num_segorphans, num_gtorphans, "ignore": stat.ignore_subvolume)                 
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
        for (sid, subval, sumstats) in subvolres
            subvol_metrics[sid] = subvolstat
            for typename, val in subval.items():
                val2 = 0
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
            sumstat["description"] = "Num ophans in worst substack (%d)" % sid 
            summary_stats.append(sumstat)
        for typename, (sid, val) in minvals.items():
            sumstat = {"name": "S-BEST-orphans", "typename": typename, "val": val, "higher-better": False}
            sumstat["description"] = "Num ophans in best substack (%d)" % sid 
            summary_stats.append(sumstat)

        return summary_stats, [], subvol_metrics
