"""Contains functionality for evaluating segmentation quality.

This module consists of the Evaluate class works with the
evaluate workflow.  Code requires pyspark to execute.

"""

import libNeuroProofMetrics as npmetrics
import numpy

# contains helper functions
from segstats import *
from morpho import *
from metrics.connectivity import *

class Evaluate(object):
    """Class to handle various aspects of segmentation evaluation workflow.

    This class performs various Spark operations over subvolumes to
    compute metrics to evaluate segmentation.  Each subvolume is examined
    and their results are combined to generate aggregate statistics.

    Note:
        No shuffling occurs in these operations and the partitioner is
        maintained until the stats are aggregated.  A global broadcast
        is currently needed by the caller to send point data to subvolumes.
        The data returned by collect() should be not be overwhelming.
    
    """
    
    # limit the size of body lists
    BODYLIST_LIMIT = 1000

    def __init__(self, config):
        self.server = config["dvid-info"]["dvid-server"]
        self.uuid = config["dvid-info"]["uuid"]
        self.body_threshold = config["options"]["body-threshold"]
        self.point_threshold = config["options"]["point-threshold"]
        self.debug = False
        if "debug" in config:
            self.debug = config["debug"]

    def _extract_distribution(self, body_overlap):
        """Helper function: extracts sorted list of body sizes.

        Args:
            body_overlap (dict): body to set(body2, overlap)
        Returns:
            sorted body size list (largest to smallest)
        """

        count = 0
        cumdisttemp = []
        for body, overlapset in body_overlap.items():
            localcount = 0
            for (body2, overlap) in overlapset:
                localcount += overlap
            count += localcount
            cumdisttemp.append([localcount, body])

        # round to 4 decimals
        cumdist = [ [round(val[0]/float(count),4), val[1]] for val in cumdisttemp]
        cumdist.sort()
        cumdist.reverse()
        return cumdist

    def _extract_subvolume_connections(self, index2body_seg, parent_list, adjacency_list):
        """Helper function: find synapse connections within a subvolume.

        This function finds connections for a list of points.
        In some cases, there are synapse points without a body
        identity and therefore they must be analyzed during 'reduce'.

        Note:
            point = synapse (pre or post determined by edge direction)

        Args:
            index2body_seg (dict): point index -> body id
            parent_list (set): local children without local parents 
            adjacency_list (dict): point -> set of child points

        Returns:
            (
                (pre-syn body, post-syn body, # connections),
                (body id -> unassigned children points,
                    point with unknown father-> body id)
            )

        """
        
        leftover_seg = {}
        seg_connections = {}
        prop_indices = {}

        # if a child is not in this subvolume and its parent is
        # propagate parent id
        for child in parent_list:
            # should be in index2body_seg by construction
            prop_indices[child] = index2body_seg[child]

        for index, body in index2body_seg.items():
            if body != -1:
                for index2 in adjacency_list[index]:
                    if index2 not in index2body_seg:
                        # resolve index later in reduce
                        if body not in leftover_seg:
                            leftover_seg[body] = set()
                        leftover_seg[body].add(index2)
                    else:
                        # add connection
                        body2 = index2body_seg[index2]
                        if body2 != -1:
                            if (body, body2) not in seg_connections:
                                seg_connections[(body, body2)] = 0
                            seg_connections[(body,body2)] += 1

        # put in same format as other overlap structures
        seg_overlap_syn = []
        for (body1, body2), overlap in seg_connections.items():
            seg_overlap_syn.append((body1, body2, overlap))
  
        return seg_overlap_syn, (leftover_seg, prop_indices)

    def _load_subvolume_stats(self, stats, overlap_gt, overlap_seg, comparison_type,
               labelgt_map, label2_map):
        """Helper function: loads various stats for the subvolume.

        Args:
            stats (SubvolumeStats): contains stats for substack
            overlap_gt (list): Overlap data for gt->seg
            overlap_seg (list): Overlap data for seg->gt
            comparison_type (ComparisonType): type of point comparison
            labelgt_map (dict): split label -> old label
            label2_map (dict): split label -> old label
       
        Returns:
            stats is updated.

        """

        gt_overlap = OverlapTable(overlap_gt, comparison_type)
        seg_overlap = OverlapTable(overlap_seg, comparison_type)

        # compute VI and rand stats
        fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies, dummy = \
                                calculate_vi(gt_overlap, seg_overlap) 
        fmerge_rand, fsplit_rand  = \
                                calculate_rand(gt_overlap, seg_overlap)

        # load substack stats
        stats.add_stat(VIStat(comparison_type, fmerge_vi, fsplit_vi))
        stats.add_stat(RandStat(comparison_type, fmerge_rand, fsplit_rand))

        # add total size
        stats.add_stat(TotalStat(comparison_type, gt_overlap.get_size()))
       
        # add number of gt and seg bodies
        stats.add_stat(NumBodyStat(comparison_type, gt_overlap.num_bodies(),
            seg_overlap.num_bodies()))

        # handle maps and load overlaps
        gt_overlap.partial_remap(labelgt_map, label2_map)
        seg_overlap.partial_remap(label2_map, labelgt_map)
        
        # add partial global results
        stats.add_gt_overlap(gt_overlap)
        stats.add_seg_overlap(seg_overlap)
    
    def calcoverlap(self, lpairs_split, boundary_iter=0):
        """Calculates voxel overlap across RDD and records subvolume stats.

        Note:
            Function preservers RDD partitioner.

        Args:
            lpairs_split (RDD): RDD is of (subvolume id, data)
            boundary_iter (int): optional specify how much GT boundary to ignore
        
        Returns:
            Original RDD with new subvolume stats and overlap stats included.

        """

        def _calcoverlap(label_pairs):
            subvolume, labelgt_map, label2_map, labelgt, label2 = label_pairs
            
            # TODO !! avoid conversion
            labelgt = labelgt.astype(numpy.float64)
            label2 = label2.astype(numpy.float64)

            # creates stack and adds boundary padding
            stackgt = npmetrics.Stack(labelgt, boundary_iter)
            stack2 = npmetrics.Stack(label2, 0)

            # returns list of (body1, body2, overlap)
            overlaps12 = stackgt.find_overlaps(stack2)

            # reverse overlaps for mappings
            overlaps21 = []
            for (body1, body2, overlap) in overlaps12:
                overlaps21.append((body2, body1, overlap))

            # ?! run TED and get scores per body (accum later for per body stat off GT and total edits needed)
            # ?! ?? should I be able to input GT skeletons -- if so maybe make another pipeline pass

            stats = SubvolumeStats(subvolume)
            
            # ?? should I return stats back
            self._load_subvolume_stats(stats, overlaps12, overlaps21,
                    ComparisonType(), labelgt_map, label2_map)

            # TODO:!! Add more distribution stats 
           
            # keep volumes for subsequent point queries
            return (stats, labelgt_map, label2_map,
                    labelgt.astype(numpy.uint64),
                    label2.astype(numpy.uint64))
            
        return lpairs_split.mapValues(_calcoverlap)


    def calcoverlap_pts(self, lpairs_split, point_list_name, point_data):
        """Calculates point overlap across RDD and records subvolume stats.

        This function can be called over any list of points.  "synapse" points
        will result in synapse-specific computation.  The list of points is
        a way to really focus on the most important parts of the segmentation.
        Filtering unimportant bodies should be slightly less relevant here
        compared to comprehensive voxel comparisons (e.g., probably less sensitive
        to boundary conditions for instance).

        Note:
            Function preservers RDD partitioner.  Currently, the points are
            broadcast from the driver.  If this data gets big, it might make
            sense to partition the points, use the same partitioner, and
            do a trivial join.  Synapse connections can be of the type
            one to many but not many to one.

        Args:
            lpairs_split (RDD): RDD is of (subvolume id, data).
            point_list_name (str): Name of the point list for identification.
            point_data (list): list of [x,y,z,ptr1,ptr2,..].

        Returns:
            Original RDD with new subvolume stats and overlap stats included.

        """

        # set type, name, sparse
        comparsison_type = ComparisonType(str(point_data["type"]),
                str(point_list_name), point_data["sparse"])

        # TODO combine labels with relevant points 
        #lpairs_split = lpairs_split.join(distpoints)
        
        def _calcoverlap_pts(label_pairs):
            stats, labelgt_map, label2_map, labelgt, label2 = label_pairs

            # for connected points, connections are encoded by using
            # the position index number X and extra numbers Y1,Y2... indicating
            # indicating that X -> Y1, Y2, etc
            # synapses encoded as presynapse -> PSD
            adjacency_list = {}
    
            # children without local parents
            parent_list = set()

            subvolume_pts = {}
            box = stats.subvolume.box
            # grab points that overlap
            for index, point in enumerate(point_data["point-list"]):
                if point[0] < box.x2 and point[0] >= box.x1 and point[1] < box.y2 and point[1] >= box.y1 and point[2] < box.z2 and point[2] >= box.z1:
                    subvolume_pts[index] = [point[0]-box.x1, point[1]-box.y1, point[2]-box.z1]
                    adjacency_list[index] = set()
                    for iter1 in range(3, len(point)):
                        adjacency_list[index].add(point[iter1])

            # find points that have a parent (connection) outside of subvolume
            for index, point in enumerate(point_data["point-list"]):
                if point[0] >= box.x2 or point[0] < box.x1 or point[1] >= box.y2 or point[1] < box.y1 or point[2] >= box.z2 or point[2] < box.z1:
                    for iter1 in range(3, len(point)):
                        if point[iter1] in adjacency_list:
                            parent_list.add(point[iter1])

            # find overlap for set of points (density should be pretty low)
            overlap_gt = []
            overlap_seg = []

            temp_overlap = {}
        
            index2body_gt = {}
            index2body_seg = {}
            index2body_gtseg = {}

            for index, point in subvolume_pts.items():
                # z,y,x -- c order
                # get bodies on points

                # ints are json serializable
                gtbody = int(labelgt[(point[2], point[1], point[0])])
                segbody = int(label2[(point[2], point[1], point[0])])
          
                # !!ignore all 0 points (assume segbody is not 0 anywhere for now)
                #if gtbody == 0 or segbody == 0:
                if gtbody == 0:
                    index2body_gt[index] = -1
                    index2body_seg[index] = -1
                    index2body_gtseg[index] = -1
                    continue

                # get point index to actual global body 
                gtbody_mapped = gtbody
                if gtbody in labelgt_map:
                    gtbody_mapped = labelgt_map[gtbody]
                segbody_mapped = segbody
                if segbody in label2_map:
                    segbody_mapped = label2_map[segbody]
                
                index2body_gt[index] = gtbody_mapped
                index2body_seg[index] = segbody_mapped
                index2body_gtseg[index] = (gtbody_mapped << 64) | segbody_mapped

                # load overlaps
                if (gtbody, segbody) not in temp_overlap:
                    temp_overlap[(gtbody,segbody)] = 0
                temp_overlap[(gtbody,segbody)] += 1
                
            for (gt, seg), overlap in temp_overlap.items():
                overlap_gt.append((gt,seg,overlap))
                overlap_seg.append((seg,gt,overlap))

            comparison_type = ComparisonType(str(point_data["type"]),
                    str(point_list_name), point_data["sparse"])
            self._load_subvolume_stats(stats, overlap_gt, overlap_seg,
                    comparison_type, labelgt_map, label2_map)

            # if synapse type load total pair matches (save boundary point deps) 
            if point_data["type"] == "synapse":
                # grab partial connectivity graph for gt
                gt_overlap_syn, leftover_gt = \
                    self._extract_subvolume_connections(index2body_gt, parent_list, adjacency_list)
            
                # add custom synapse type
                stats.add_gt_syn_connections(OverlapTable(gt_overlap_syn,
                        ComparisonType("synapse-graph", str(point_list_name),
                        point_data["sparse"])), leftover_gt)

                # grab partial connectivity graph for seg
                seg_overlap_syn, leftover_seg = \
                    self._extract_subvolume_connections(index2body_seg,
                            parent_list, adjacency_list)
                
                # add custom synapse type
                stats.add_seg_syn_connections(OverlapTable(seg_overlap_syn,
                        ComparisonType("synapse-graph", str(point_list_name),
                        point_data["sparse"])), leftover_seg)
    
                # add table showing intersection of gtseg
                gtseg_overlap_syn, leftover_gtseg = \
                    self._extract_subvolume_connections(index2body_gtseg,
                            parent_list, adjacency_list)
                
                # add custom synapse type
                stats.add_gtseg_syn_connections(OverlapTable(gtseg_overlap_syn,
                        ComparisonType("synapse-graph", str(point_list_name),
                        point_data["sparse"])), leftover_gtseg)

            # points no longer needed
            return (stats, labelgt_map, label2_map, labelgt, label2)

        return lpairs_split.mapValues(_calcoverlap_pts)

    def calculate_stats(self, lpairs_splits):
        """Generates subvolume and aggregate stats.

        Stats retrieved for volume and each set of points:

            * Rand and VI across whole set -- a few GT body thresholds
            * Per body VI (fragmentation factors and GT body quality) -- 
            a few thresholds
            * Per body VI (at different filter thresholds) -- 
            a few filter thresholds
            * Histogram of #bodies vs #points/annotations for GT and test
            * Approx edit distance (no filter) 
            * Per body edit distance 
            (both sides for recompute of global, plus GT)
            * Show connectivity graph of best bodies
            (correspond bodies and then show matrix)
            (P/R computed for different thresholds but easily handled client side)
            * TODO Per body largest corrected -- (could be useful for
            understanding best-case scenarios) -- skeleton version?
            * TODO Best body metrics
            * TODO P/R type metric on clusterings run over connectivity graph??
            (perhaps run K-means on both sides and then do synapse VI, or agglomerative
            clustering with a touching or locality constraints )
        
        Importance GT selections, GT noise filters, and importance filters
        (test seg side):
        
            * (client) Body stats can be selected by GT size or other
            preference client side.
            * (client) Bodies selected can be used to compute cumulative VI --
            (just have sum under GT body list)
            * (client-side bio select): P/R for connectivity graph appox
            (importance filter could be used for what is similar but probably
            hard-code >50% of body since that will allow us to loosely
            correspond things)
            * (pre-computed) Small bodies can be filtered from GT to reduce
            noise per body/accum (this can be done just by percentages and
            just record what the threshold was)
            * (pre-computed) Edit distance until body is within a certain threshold.
            * (client) ?? Use slider to filter edit distance by body size
            (appoximate edit distance since splits can help multiple bodies?)

        Advances:
            * Extensive breakdowns by body, opportunity for average body and outliers
            * Breakdown by different types of points
            * Explicit pruning by bio size
            * Edit distance (TODO: more realistic edit distance)
            * Thresholded synapse measures (graph measures ??) -- probably not
            * Histogram views and comparisons
            * Subvolume breakdowns for 'heat-map'
            (outliers are important, identify pathological mergers)
            * Ability to handle very large datasets
            * Display bodies that are good, help show best-case scenarios (substacks
            inform this as well)
            * Multiple metrics that help to inform biology, instruct where to
            impove, what can be used automatically, etc

        Note:
            * Filter for accumulative VI total is not needed since per body VI
            can just be summed at different thresholds
            * GT body VI will be less meaningful over sparse point datasets
            * Test body fragmentation will be less meaningful over sparse
            point datasets
            * Edit distance can be made better.  Presumably, the actual
            nuisance metric is higher since proofreaders need to verify more than
            they actually correct.  The difference of work at different thresholds
            will indicate that one needs to be careful what one considers important.

        Args:
            lpairs_split (RDD): RDD is of (subvolume id, data).
       
        Returns:
            dict containing metrics.

        """

        # extract subvolume stats and send to the driver
        all_stats = lpairs_splits.map(lambda x: x[1][0]).collect()
        
        metric_results = {}

        # accumulate all subvolume stats
        whole_volume_stats = None # SubvolumeStats(None)

        ##### SUBSTACK STATS ######
        
        allsubvolume_metrics = {}
        allsubvolume_metrics["types"] = {}
        allsubvolume_metrics["ids"] = {}

        synapse_index_position =  0
            
        for subvol_stats in all_stats:
            subvolume_metrics = {}
            # load substack location/size info 
            subvolume_metrics["roi"] = subvol_stats.subvolume.box
            subvolume_metrics["types"] = {}

            # subvolume stats per comparison type (defines good)
            # (SubID -> ComparisonType -> stat type -> values)
            # 1. merge/split rand (high)
            # 2. merge/split vi (low)
            # 3. size
            # 4. # bodies low/high
            
            # all subvolume stats (defines good)
            # (ComparisonType -> stat type -> values)
            # 1. mean merge/split rand (high)
            # 2. mean merge/split vi (low)
            # 3. worst merge/split VI subvolume (with ID)
            # 4. worst merge/split rand subvolume (with ID)
            
            for stat in subvol_stats.subvolume_stats:
                # write local stats and accumulate
                stat.write_to_dict(subvolume_metrics["types"], allsubvolume_metrics, subvol_stats.subvolume.sv_index)

            allsubvolume_metrics["ids"][subvol_stats.subvolume.sv_index] = subvolume_metrics
            
            # accumulate stats
            if whole_volume_stats is None:
                whole_volume_stats = subvol_stats
            else:
                whole_volume_stats.merge_stats(subvol_stats)



        """
        # verify that leftover list is empty
        for connections in whole_volume_stats.seg_syn_connections:
            stats, (leftovers, props) = connections
            if len(leftovers) != 0:
                raise Exception("Synapse leftovers are non-empty")
        for connections in whole_volume_stats.gt_syn_connections:
            stats, (leftovers, props) = connections
            if len(leftovers) != 0:
                raise Exception("Synapse leftovers are non-empty")
        """

        # store subvolume results 
        metric_results['subvolumes'] = allsubvolume_metrics

        # write results for each comparison type
        comparison_type_metrics = {}
        comparison_type_metrics["summarystats"] = []

        for iter1 in range(0, len(whole_volume_stats.gt_overlaps)):
            # check for comparison and create if necessary
            comptype = whole_volume_stats.gt_overlaps[iter1].comparison_type
            typename = str(comptype)
            if typename not in comparison_type_metrics:
                comparison_type_metrics[typename] = {}

            if comptype.typename == "synapse":
                synapse_index_position = iter1
            
            ##### HISTOGRAM STATS #######
            distgt = self._extract_distribution(whole_volume_stats.gt_overlaps[iter1].overlap_map)
            distseg = self._extract_distribution(whole_volume_stats.seg_overlaps[iter1].overlap_map)

            # TODO !! take subset of distribution
            comparison_type_metrics[typename]["dist-gt"] = [val[0] for val in distgt]
            comparison_type_metrics[typename]["dist-seg"] = [val[0] for val in distseg]

            # add top body overlaps (top 1000)
           
            # for GT
            top_overlapgt = {}
            numoverlaps = self.BODYLIST_LIMIT
            nummatches = 10 # find only up to 10 matches
            if len(distgt) < numoverlaps:
                numoverlaps = len(distgt)
            for distiter in range(0, numoverlaps):
                overlapmap = whole_volume_stats.gt_overlaps[iter1].overlap_map[distgt[distiter][1]]
                overlapsize = []
                matchlist = []
                for (body2, overlap) in overlapmap:
                    matchlist.append([overlap, body2])
                matchlist.sort()
                matchlist.reverse()
                top_overlapgt[distgt[distiter][1]] = matchlist[0:nummatches]
            comparison_type_metrics[typename]["top-overlap-gt"] = top_overlapgt
           
            # for seg
            top_overlapseg = {}
            numoverlaps = self.BODYLIST_LIMIT
            nummatches = 10 # find only up to 10 matches
            if len(distseg) < numoverlaps:
                numoverlaps = len(distseg)
            for distiter in range(0, numoverlaps):
                overlapmap = whole_volume_stats.seg_overlaps[iter1].overlap_map[distseg[distiter][1]]
                overlapsize = []
                matchlist = []
                for (body2, overlap) in overlapmap:
                    matchlist.append([overlap, body2])
                matchlist.sort()
                matchlist.reverse()
                top_overlapseg[distseg[distiter][1]] = matchlist[0:nummatches]
            comparison_type_metrics[typename]["top-overlap-seg"] = top_overlapseg

            # TODO !! generate statitcs histogram

            ###### AGGREGATE STATS #######
            gt_table = whole_volume_stats.gt_overlaps[iter1]
            seg_table = whole_volume_stats.seg_overlaps[iter1]
            gt_overlap = gt_table.overlap_map
            seg_overlap = seg_table.overlap_map
            
            # TODO !! Add normalized per body vi by body size
           
            body_threshold_loc = self.body_threshold
            if comptype.typename != "voxels":
                body_threshold_loc = self.point_threshold # assume everything else uses point threshold
            """else:
                # ?! temporary
                body_mappings = []
                for body, overlapset in gt_overlap.items():
                    for body2, overlap in overlapset:
                        body_mappings.append([body, body2])
                tfile = open("/groups/scheffer/home/plazas/remappings.json", 'w')
                import json
                tfile.write(json.dumps(body_mappings))
            """

            # ignore smallest bodies (GT noise could complicate comparisons)
            fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies, vi_bodies = \
                                    calculate_vi(gt_table, seg_table,
                                            body_threshold_loc)
            fmerge_rand, fsplit_rand  = \
                                    calculate_rand(gt_table, seg_table,
                                            body_threshold_loc)
           
            # add rand, vi global,  for each type
            statvi = VIStat(comptype, fmerge_vi, fsplit_vi)
            statrand = RandStat(comptype, fmerge_rand, fsplit_rand)
            statvi.write_to_dict(comparison_type_metrics)
            statrand.write_to_dict(comparison_type_metrics)

            ####### EDIT DISTANCE ######
           
            # pretty resistant to noise since not going to 100%
            edit_distance = EditDistanceStat(comptype, gt_overlap, seg_overlap, body_threshold_loc) 
            edit_distance.write_to_dict(comparison_type_metrics)
           
            ###### BEST BODY STATS ######
            # just use max overlap if >50%
            important_segbodies = {}
            for gt, overlapset in gt_overlap.items():
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
                for seg2, overlap in seg_overlap[max_id]:
                    total2 += overlap 
                
                # match body if over half of the seg
                if max_val > total2/2:
                    important_segbodies[max_id] = max_val


            ###### PER BODY VI STATS ######
            
            # id -> [vifsplit, vitotal, size]
            comparison_type_metrics[typename]["gt-bodies"] = {}  # fsplit
            # id -> [vifmerge, vitotal, size]
            comparison_type_metrics[typename]["seg-bodies"] = {} # fmerge
          
            # Do not report smallest bodies 
            # examine gt bodies
            worst_gt_body = 0 
            worst_gt_val = 0
            worst_fsplit = 0
            worst_fsplit_body = 0
            for body, overlapset in gt_overlap.items():
                total = 0
                for body2, overlap in overlapset:
                    total += overlap
                if total < body_threshold_loc:
                    continue
                fsplit = fsplit_bodies[body]
                vitot = vi_bodies[body]
                comparison_type_metrics[typename]["gt-bodies"][body] = \
                        [fsplit, vitot, total]
                if fsplit > worst_fsplit:
                    worst_fsplit = fsplit
                    worst_fsplit_body = body
                if vitot > worst_gt_val:
                    worst_gt_val = vitot
                    worst_gt_body = body

            comparison_type_metrics[typename]["worst-vi"] = [worst_gt_val, worst_gt_body]
            comparison_type_metrics[typename]["worst-fsplit"] = [worst_fsplit, worst_fsplit_body]
           
            # examine seg bodies
            worst_fmerge = 0
            worst_fmerge_body = 0

            best_size = 0
            best_body_id = 0

            for body, overlapset in seg_overlap.items():
                total = 0
                for body2, overlap in overlapset:
                    total += overlap
                if total < body_threshold_loc:
                    continue
                fmerge = fmerge_bodies[body]

                maxoverlap = 0
                if body in important_segbodies:
                    maxoverlap = important_segbodies[body]

                if maxoverlap > best_size:
                    best_size = maxoverlap
                    best_body_id = body

                comparison_type_metrics[typename]["seg-bodies"][body] = \
                        [fmerge, maxoverlap, total]
                
                if fmerge > worst_fmerge:
                    worst_fmerge = fmerge
                    worst_fmerge_body = body

            comparison_type_metrics[typename]["worst-fmerge"] = [worst_fmerge, worst_fmerge_body]
            comparison_type_metrics[typename]["greatest-overlap"] = [best_size, best_body_id]
     
        #### Connection Matrix and Connection Stats ############

        gt_synapses = {}
        seg_synapses = {}
        comparison_type_metrics["connection-matrix"] = {}
        if len(whole_volume_stats.seg_syn_connections) != 0:
            # grab non sparse synapse info if available
            # !! Assume first synapse file is the one used for the calculating the graph
            comptype = whole_volume_stats.seg_syn_connections[0][0].comparison_type
            if not comptype.sparse:
                tgt_synapses = whole_volume_stats.gt_syn_connections[0][0].overlap_map
                tseg_synapses = whole_volume_stats.seg_syn_connections[0][0].overlap_map
                gtseg_synapses = whole_volume_stats.gtseg_syn_connections[0][0].overlap_map

                gt_overlap = whole_volume_stats.gt_overlaps[synapse_index_position].overlap_map
                seg_overlap = whole_volume_stats.seg_overlaps[synapse_index_position].overlap_map
                
                # !! Using simple 50 percent threshold to find body matches

                # TODO: !! better strategy than taking simple threshold for important
                # bodies -- for now only take GT > 50 synapses
                importance_threshold = body_threshold_loc 
                if self.debug: # examine all synapses for small example
                    importance_threshold = 0
                
                # --- compute new connectivity table ---
                bodymatches = compute_bodymatch(gt_overlap, importance_threshold)   
                sumstatsbm, bodystatsbm, tablestatsbm = compute_tablestats(bodymatches, gtseg_synapses, typename, [1, 5, 10])
                
                comparison_type_metrics["connection-matrix2"] = {}
                comparison_type_metrics["summarystats"].extend(sumstatsbm)
                comparison_type_metrics["connection-matrix2"]["body"] = bodystatsbm
                comparison_type_metrics["connection-matrix2"]["table"] = tablestatsbm
                # --- end compute new connectivity table ---
                
                
                important_gtbodies = set() 
                important_segbodies = {}
                for gt, overlapset in gt_overlap.items():
                    total = 0
                    max_id = 0
                    max_val = 0
                    for seg, overlap in overlapset:
                        total += overlap
                        if overlap > max_val:
                            max_val = overlap
                            max_id = seg
                    if total < importance_threshold:
                        break
                    # find size of seg body
                    total2 = 0
                    for seg2, overlap in seg_overlap[max_id]:
                        total2 += overlap 
                    
                    # match body if over half of the seg is in over half of the GT
                    if max_val > total/2 and max_val > total2/2:
                        important_gtbodies.add(gt)
                        important_segbodies[max_id] = gt
                
                for pre, overlapset in tgt_synapses.items():
                    if pre not in important_gtbodies:
                        continue
                    for (post, overlap) in overlapset:
                        if post in important_gtbodies:
                            if pre not in gt_synapses:
                                gt_synapses[pre] = set()
                            gt_synapses[pre].add((post, overlap))
                
                for pre, overlapset in tseg_synapses.items():
                    if pre not in important_segbodies:
                        continue
                    for post, overlap in overlapset:
                        if post in important_segbodies:
                            if pre not in seg_synapses:
                                seg_synapses[pre] = set()
                            seg_synapses[pre].add((post, overlap))

                # look for synapse points and compute graph measure (different cut-offs?)
                # provide graph for matching bodies; provide stats at different thresholds
                if len(gt_synapses) != 0:
                    gt_connection_matrix = []
                    gtpathway = {}
                    segpathway = {}
                    for gtbody, overlapset in gt_synapses.items():
                        for (partner, overlap) in overlapset:
                            gt_connection_matrix.append([gtbody, partner, overlap])
                            gtpathway[(gtbody, partner)] = overlap
                    seg_connection_matrix = []
                    for segbody, overlapset in seg_synapses.items():
                        for (partner, overlap) in overlapset:
                            # add mapping to gt in matrix
                            seg_connection_matrix.append([important_segbodies[segbody],
                                segbody, important_segbodies[partner], partner, overlap])
                            segpathway[(important_segbodies[segbody],
                                important_segbodies[partner])] = overlap

                    # load stats        
                    comparison_type_metrics["connection-matrix"]["gt"] = gt_connection_matrix
                    comparison_type_metrics["connection-matrix"]["seg"] = seg_connection_matrix
                    # dump a few metrics to add to the master table
                    # ignore connections <= threshold
                    for threshold in [0, 5, 10]:
                        recalled_gt = 0
                        for connection, weight in gtpathway.items():
                            if weight <= threshold:
                                continue
                            if connection in segpathway and segpathway[connection] > threshold:
                                recalled_gt += 1
                        false_conn = 0
                        for connection, weight in segpathway.items():
                            if weight <= threshold:
                                continue
                            if connection not in gtpathway or gtpathway[connection] < threshold:
                                false_conn += 1
                        if "thresholds" not in comparison_type_metrics["connection-matrix"]:
                            comparison_type_metrics["connection-matrix"]["thresholds"] = []
                        comparison_type_metrics["connection-matrix"]["thresholds"].append([false_conn, recalled_gt, threshold])
                else:
                    comparison_type_metrics["connection-matrix"]["gt"] = []
                    comparison_type_metrics["connection-matrix"]["seg"] = []
                    comparison_type_metrics["connection-matrix"]["thresholds"] = []

        metric_results["types"] = comparison_type_metrics
        return metric_results

