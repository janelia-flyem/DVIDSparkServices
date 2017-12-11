"""Contains functionality for evaluating segmentation quality.

This module consists of the Evaluate class works with the
evaluate workflow.  Code requires pyspark to execute.

"""
from __future__ import absolute_import
from __future__ import division
import libNeuroProofMetrics as npmetrics
import numpy
import math

# contains helper functions
from DVIDSparkServices.reconutils.morpho import *

# metric specific modules
from .segstats import *
from .plugins.stat import *
from .comptype import *
from .overlap import *
from .synoverlap import *


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
        self.no_gt = config["options"]["no-gt"]
        self.enable_sparse = config["options"]["enable-sparse"]
        self.important_bodies = config["options"]["important-bodies"]
        self.num_displaybodies = config["options"]["num-displaybodies"]
        self.selfcompare = False
        if "dvid-info-comp" not in config:
            self.selfcompare = True
        self.debug = False
        self.comptypes = []

        if "debug" in config:
            self.debug = config["debug"]

        # load all metric plugins
        self.metrics = []
        self.config = config
            
        import importlib
        for metric in config["options"]["plugins"]:
            modname = importlib.import_module('DVIDSparkServices.reconutils.metrics.plugins.'+metric["name"]+"_stat")
            metric_class = getattr(modname, metric["name"] + "_stat")
            self.metrics.append((metric_class, metric["parameters"]))


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

        gt_overlap = OverlapTable(overlap_gt, comparison_type, labelgt_map, label2_map)
        seg_overlap = OverlapTable(overlap_seg, comparison_type, label2_map, labelgt_map)

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
        comptype = ComparisonType()
        self.comptypes.append(comptype.get_name())

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

            stats = SubvolumeStats(subvolume, self.body_threshold, self.point_threshold, self.num_displaybodies, self.no_gt, self.selfcompare, self.enable_sparse, self.important_bodies)
            stats.disable_subvolumes = self.config["options"]["disable-subvolumes"]

            # load different metrics available
            for (metric, config) in self.metrics:
                stats.add_stat(metric(**config))

            # ?! disable if only one substack ?? -- maybe still good for viewer

            self._load_subvolume_stats(stats, overlaps12, overlaps21,
                    comptype, labelgt_map, label2_map)
            
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
        comparison_type = ComparisonType(str(point_data["type"]),
                str(point_list_name), point_data["sparse"])
        self.comptypes.append(comparison_type.get_name())

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
            box = stats.subvolumes[0].box
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
        
            #index2body_gt = {}
            #index2body_seg = {}
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
                    #index2body_gt[index] = -1
                    #index2body_seg[index] = -1
                    index2body_gtseg[index] = -1
                    continue

                # get point index to actual global body 
                gtbody_mapped = gtbody
                if gtbody in labelgt_map:
                    gtbody_mapped = labelgt_map[gtbody]
                segbody_mapped = segbody
                if segbody in label2_map:
                    segbody_mapped = label2_map[segbody]
                
                #index2body_gt[index] = gtbody_mapped
                #index2body_seg[index] = segbody_mapped
                index2body_gtseg[index] = (gtbody_mapped << 64) | segbody_mapped

                # load overlaps
                if (gtbody, segbody) not in temp_overlap:
                    temp_overlap[(gtbody,segbody)] = 0
                temp_overlap[(gtbody,segbody)] += 1
                
            for (gt, seg), overlap in temp_overlap.items():
                overlap_gt.append((gt,seg,overlap))
                overlap_seg.append((seg,gt,overlap))

            self._load_subvolume_stats(stats, overlap_gt, overlap_seg,
                    comparison_type, labelgt_map, label2_map)

            # if synapse type load total pair matches (save boundary point deps) 
            if point_data["type"] == "synapse":
                # grab partial connectivity graph for gt
                #gt_overlap_syn, leftover_gt = \
                #    self._extract_subvolume_connections(index2body_gt, parent_list, adjacency_list)
            
                # add custom synapse type
                #stats.add_gt_overlap(SynOverlapTable(gt_overlap_syn,
                #        ComparisonType("synapse-graph", str(point_list_name),
                #        point_data["sparse"]), leftover_gt))

                # grab partial connectivity graph for seg
                #seg_overlap_syn, leftover_seg = \
                #    self._extract_subvolume_connections(index2body_seg,
                #            parent_list, adjacency_list)
                
                # add custom synapse type
                #stats.add_seg_overlap(SynOverlapTable(seg_overlap_syn,
                #        ComparisonType("synapse-graph", str(point_list_name),
                #        point_data["sparse"]), leftover_seg))
    
                # add table showing intersection of gtseg
                gtseg_overlap_syn, leftover_gtseg = \
                    self._extract_subvolume_connections(index2body_gtseg,
                            parent_list, adjacency_list)
                
                # add custom synapse type
                stats.add_gt_overlap(SynOverlapTable(gtseg_overlap_syn,
                        ComparisonType("synapse-graph-gtseg", str(point_list_name),
                        point_data["sparse"]), leftover_gtseg))
               
                # add dummy placeholder (TODO: refactor segstats to avoid this)
                stats.add_seg_overlap(SynOverlapTable([],
                        ComparisonType("synapse-graph-gtseg", str(point_list_name),
                        point_data["sparse"]), ({}, {})))

            # points no longer needed
            return (stats, labelgt_map, label2_map, labelgt, label2)

        return lpairs_split.mapValues(_calcoverlap_pts)

    def calculate_stats(self, lpairs_splits):
        """Generates subvolume and aggregate stats.

        Args:
            lpairs_split (RDD): RDD is of (subvolume id, data).
       
        Returns:
            dict containing metrics.

        """

        # TODO: allow stat to customize RDD workflow if needed (like 2-pass orphan counts)

        # metric results divided into summarystats, bodystats, and subvolumes
        metric_results = {}
        metric_results["summarystats"] = []
        metric_results["bodystats"] = []
        metric_results["bodydebug"] = []
        allsubvolume_metrics = {}
        metric_results['subvolumes'] = allsubvolume_metrics
        metric_results['types'] = self.comptypes


        # no longer need volumes
        allstats = lpairs_splits.map(lambda x: x[1][0]) 
       
        
        # generate stat state if needed
        def compute_subvolume(stat):
            # stat will produce an array of stats for each substack
            stat.compute_subvolume()
            return stat 
        subvolstats_computed = allstats.map(compute_subvolume)

        # save stats to enable subvolumes to be printed
        subvolstats_computed.persist()
      
        # only compute subvolume if enabled
        if not self.config["options"]["disable-subvolumes"]:
            # each subvolume will extract subvol relevant stats
            def extractstats(stat):
                # stat will produce an array of stats for each substack
                sumstats = stat.write_subvolume_stats()
                # add bbox type
                sumstats.append({"name": "bbox", "val": stat.subvolumes[0].box}) 
                return (stat.subvolumes[0].sv_index, sumstats)
            subvolstats = subvolstats_computed.map(extractstats).collect()
            # set allsubvolume_metrics
            for (sid, subvolstat) in subvolstats:
                allsubvolume_metrics[sid] = subvolstat

        # combine all the stats (max/min/average substack stacks maintained as well)
        def combinestats(stat1, stat2):
            stat1.merge_stats(stat2)
            return stat1
        whole_volume_stats = subvolstats_computed.treeReduce(combinestats, int(math.log(len(subvolstats),2)))
        
        # compute summary, body stats, and debug information
        # iterate stats and collect summary and body stats
        metric_results["summarystats"].extend(whole_volume_stats.write_summary_stats())
        metric_results["bodystats"].extend(whole_volume_stats.write_body_stats())
        # iterate stats and collect debug info "bodydebug"
        metric_results["bodydebug"].extend(whole_volume_stats.write_bodydebug())
        
        return metric_results

