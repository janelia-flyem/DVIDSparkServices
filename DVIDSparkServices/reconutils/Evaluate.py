import libNeuroProofMetrics as np
from math import log
import numpy

# ?! refactor into general VI calc 
def body_vi(self, bodies_overlap):
    # input: (body1id, list[(body2id, count)])
    # output: (body1id, VI_unnorm, count)

    body1, overlap_list = bodies_overlap

    # accumulate values
    body2counts = {}
    total = 0
    for (body2, val)  in overlap_list:
        if (body1, body2) in body2counts:
            body2counts[(body1, body2)] += val
        else:
            body2counts[(body1, body2)] = val
        total += val

    vi_unnorm = 0
    for key, val in body2counts.items():
        # compute VI
        vi_unnorm += val*log(total/val)/log(2.0)

    return (body1, vi_unnorm, total)


class StatType:
    VI = 1
    RAND = 2

class ComparisonType(object):
    def __init(self, typename="voxels", instance_name="voxels", sparse=False):
        self.typename = typename
        self.instance_name = voxels
        self.sparse = sparse

class OverlapStats(object):
    def __init__(self, overlaps, comarison_type):
        self.comparison_type = comparison_type
        self.overlap_map = {}

        for item in overlaps:
            body1, body2, overlap = item
            if body1 not in self.overlap_map:
                self.overlap_map[body1] = set()
            self.overlap_map[body1].add((body2, overlap))


class SubvolumeOverlapStats(object):
    def __init__(self, subvolume):
        self.subvolume = subvolume 

        # contains "rand", "vi", etc for substack
        self.subvolume_stats = []

        # contains overlaps over the various sets
        # of overlap computed
        self.gt_overlaps = []
        self.seg_overlaps = []

        gtstats = OverlapStats(gt_list)
        segstats = OverlapStats(seg_list)
        
        # list of cumulative stats
        self.subvolume_stats = []
    
    def add_gt_overlap(self, stats):
        self.gt_overlaps.append(stats)
    
    def add_seg_overlap(self, stats):
        self.seg_overlaps.append(stats)

    def add_stat(self, comparison_type, stat_type, value)
        self.subvolume_stats.append((comparison_type, stat_type, value))

class Evaluate(object):
    def __init__(self, config):
        pass

    # ?! refactor internal function to Morpho.py ??
    def split_disjoint_labels(label_pairs):
        def _split_disjoint(label_pairs):
            subvolume, labelgtc, label2c = label_pairs

            # ?! handle compression
            

            # ?! create edge boundaries and run connected components (NP?)
            labelgt_map, labelsgt = morpho.connected_components(labelgt)
            label2_map, label2 = morpho.connected_components(label2))

            # ?! compress results

            return (subvolume, labelgt_map, label2_map, labelgt, label2)

        return label_pairs.mapValues(_split_disjoint)

    def calcoverlap(self, lpairs_split, boundary_iter=0):
        def _calcoverlap(label_pairs):
            subvolume, labelgt_map, label2_map, labelgt, label2 = label_pairs

            # ?! handle compression

            # ?! avoid conversion
            labelgt = labelgt.astype(numpy.float64)
            label2 = label2.astype(numpy.float64)

            # ?! creates stack and adds boundary padding
            stackgt = np.Stack(labelgt, boundary_iter)
            stack2 = np.Stack(label2, 0)

            # returns list of (body1, body2, overlap)
            overlaps12 = stackgt.find_overlaps(stack2)
            overlaps21 = stack2.find_overlaps(stackgt)

            stats = SubvolumeStats(subvolume)
            
            gt_overlap = OverlapStats(overlap12)
            seg_overlap = OverlapStats(overlap21)
        
            # ?! compute VI and rand stats
            fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies = 
                                    calculate_vi(gt_overlap, seg_overlap)
            fmerge_rand, fsplit_rand, fmerge_bodies, fsplit_bodies =
                                    calculate_vi(gt_overlap, seg_overlap)

            # ?! handle maps and load overlaps
            
            stats.add_gt_overlap(gt_overlap)
            stats.add_seg_overlap(seg_overlap)

            # load substack stats
            stats.add_stat(ComparisonType(), StatType.VI, [fmerge_vi, fsplit_vi])
            stats.add_stat(ComparisonType(), StatType.RAND, [fmerge_rand, fsplit_rand])

            # ?? add orphans or distribution stats ??

            # keep volumes for subsequent point queries
            return (stats, labelgt_map, label2_map, labelgt, label2)
            
        return lpairs_split.mapValues(_calcoverlap)


    def calcoverlap_pts(lpairs_split, point_list_name):
        # ?! grab point list from DVID
        
        # ?! set type, name, sparse
        
        # ?! partition into subvolumes (?? get subvolumes somehow)

        # combine labels with relevant points 
        lpairs_split = lpairs_split.join(distpoints)
        
        def _calcoverlap_pts(label_pairs):
            stats, labelgt_map, label2_map, labelgtc, label2c, points = label_pairs
            
            # ?! decompress

            # ?! find overlap for set of points


            gt_overlap = OverlapStats(overlap_gt, comparison_type)
            seg_overlap = OverlapStats(overlap_seg, comparison_type)

            # ?! compute VI and rand stats
            fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies = 
                                    calculate_vi(gt_overlap, seg_overlap)
            fmerge_rand, fsplit_rand, fmerge_bodies, fsplit_bodies =
                                    calculate_vi(gt_overlap, seg_overlap)

            # ?! handle maps and load overlaps
            
            stats.add_gt_overlap(gt_overlap)
            stats.add_seg_overlap(seg_overlap)

            # ?! if synapse type load total pair matches

            # load substack stats
            stats.add_stat(comparison_type, StatType.VI, [fmerge_vi, fsplit_vi])
            stats.add_stat(comparison_type, StatType.RAND, [fmerge_rand, fsplit_rand])
            
            # ?! compress
           
            # points no longer needed
            return (stats, labelgt_map, label2_map, labelgt, label2)

        return lpairs_split.mapValues(_calcoverlap_pts)

    # ?! ?? how to calculate at different thresholds, get better edit distance
    def calculate_stats(lpairs_splits):
        # extract subvolume stats and send to the driver
        all_stats.map(lamda x: x[1][0]).collect()
        
        metric_results = {}

        # ?! load substack stats (SubID -> ComparisonType -> stat type -> values)
        
        # ?! load substack info (SubID -> coordinates)

        # ?! accumulate stats for each comparison type

        # ?! call rand, vi to get body list and accumative

        # ?! load whole stats (load for different thresholds)
        # (ComparisonType -> rand -> values)
        # (ComparisonType -> vi -> values)
        # (ComparisonType -> vi -> bodygt -> values)
        # (ComparisonType -> bodyvi -> bodygt -> values)
        # (ComparisonType -> vi -> bodyseg -> values)

        # ?! calculate cumulative graph of #bodies -> total volume
        # (ComparisonType -> (X,Y) vals

        # ?! calculate histogram threshold -> # bodies
        # (ComparisonType -> (binval, #)

        # ?! cumulative edit distance (report for a few different thresholds??)
        # (ComparisonType -> merge -> #edits)
        # (ComparisonType -> split -> #edits)

        # ?! look for synapse points and compute graph measure (different cut-offs?)

        # ?! pack everythin in map
        return metric_results


