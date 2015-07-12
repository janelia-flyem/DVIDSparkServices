import libNeuroProofMetrics as np
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
from math import log
import numpy

# TODO !!: make base class for different types -- serialize to json??
class StatType:
    VI = 1
    RAND = 2
    TOTALPTS = 3
    NUMBODIES = 4

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

# ?! per body VI

def body_vi(overlapset):
    total = 0
    for (segid, overlap) in overlapset:
        total += overlap

    decomp_bodies = {}
    vi_unnorm = 0
    for (segid, overlap) in overlapset:
        vi_unnorm += overlap*log(total/overlap)/log(2.0)
        if segid not in decomp_bodies:
            decomp_bodies[segid] = 0
        decomp_bodies[segid] += overlap*log(total/overlap)/log(2.0)

    return vi_unnorm, total, decomp_bodies

# calculate Variation of Information metric
def calculate_vi(gtoverlap, segoverlap):
    fsplit_bodies = {}
    fmerge_bodies = {}

    perbody = {}

    glb_total = 0
    fmerge_vi = 0
    fsplit_vi = 0

    # examine fragmentation of gt (fsplit=oversegmentation)
    for (gtbody, overlapset) in gtoverlap.overlap_map.items():
        vi_unorm, total, dummy = body_vi(overlapset)
        fsplit_bodies[gtbody] = vi_unnorm
        perbody[gtbody] = vi_unnorm
        glb_total += total
        fmerge_vi += vi_unnorm

    # examine fragmentation of seg (fmerge=undersegmentation)
    for (segbody, overlapset) in segoverlap.overlap_map.items():
        vi_unorm, total, gtcontribs = body_vi(overlapset)
        fmerge_bodies[segbody] = vi_unnorm
        fsplit_vi += vi_unnorm

        for key, val in gtcontribs.items():
            perbody[key] += val

    for key, val in fsplit_bodies.items():
        fsplit_bodies[key] = val/glb_total
    
    for key, val in fmerge_bodies.items():
        fmerge_bodies[key] = val/glb_total

    return fmerge_vi/glb_total, fsplit_vi/glb_total, fmerge_bodies, fsplit_bodies, perbody 

# calculate Rand Index
def calculate_rand(gtoverlap, segoverlap):
    fsplit_total = 0
    overlap_total = 0
    
    # examine fragmentation of gt (fsplit=oversegmentation)
    for (gtbody, overlapset) in gtoverlap.overlap_map.items():
        total = 0
        for (segid, overlap) in overlapset:
            total += overlap
            overlap_total += (overlap*(overlap-1)/2)

        fsplit_total += (total*(total-1)/2)

    fmerge_total = 0
    # examine fragmentation of seg (fmerge=undersegmentation)
    for (gtbody, overlapset) in segoverlap.overlap_map.items():
        total = 0
        for (segid, overlap) in overlapset:
            total += overlap

        fmerge_total += (total*(total-1)/2)

    return overlap_total / fmerge_total, overlap_total / fsplit_total

class SubvolumeOverlapStats(object):
    def __init__(self, subvolume):
        self.subvolume = subvolume 

        # contains "rand", "vi", etc for substack
        self.subvolume_stats = []

        # contains overlaps over the various sets
        # of overlap computed
        self.gt_overlaps = []
        self.seg_overlaps = []

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
        self.server = config["dvid-info"]["dvid-server"]
        self.uuid = config["dvid-info"]["uuid"]

    # TODO !! refactor internal function to Morpho.py
    def split_disjoint_labels(label_pairs):
        def _split_disjoint(label_pairs):
            from skimage.measure import label
            subvolume, labelgtc, label2c = label_pairs

            # extract numpy arrays
            labelgt = labelgtc.deserialize()
            label2 = label2c.deserialize()

            # run connected components
            labelgt_split = label(labelgt)
            label2_split = label(label2)
            
            # find maps and remap partition with rest
            stackgt = np.Stack(labelgt, boundary_iter)
            stackgt_split = np.Stack(labelgt_split, boundary_iter)
            overlap_gt_split = stackgt.find_overlaps(stackgt_split)
            statsgt = OverlapStats(ovelap_gt_split, ComparisonType())

            # needed to remap split labels into original coordinates
            remapping = {}
            labelgt_map = {}
            remap_id = labelgt.max() + 1

            for orig, newset in statsgt.overlap_map.items():
                if len(newset) == 1:
                    remapping[next(iter(newset))] = orig
                else:
                    for newbody in newset:
                        # guarantee no class of label ids
                        remapping[newbody] = remap_id
                        remap_id += 1
                        labelgt_map[remap_id] = orig

            # relabel volume (minimize size of map that needs to be communicated)
            # TODO: !! refactor into morpho
            vectorized_relabel = numpy.frompyfunc(remapping.__getitem__, 1, 1)
            labelgt_split = vectorized_relabel(labelgt_split).astype(numpy.uint64))

            # find maps and remap partition with rest
            stack2 = np.Stack(label2, boundary_iter)
            stack2_split = np.Stack(label2_split, boundary_iter)
            overlap_2_split = stack2.find_overlaps(stack2_split)
            stats2 = OverlapStats(ovelap_2_split, ComparisonType())

            # needed to remap split labels into original coordinates
            remapping = {}
            label2_map = {}
            remap_id = label2.max() + 1

            for orig, newset in stats2.overlap_map.items():
                if len(newset) == 1:
                    remapping[next(iter(newset))] = orig
                else:
                    for newbody in newset:
                        remapping[newbody] = remap_id
                        remap_id += 1
                        label2_map[remap_id] = orig

            # relabel volume (minimize size of map that needs to be communicated)
            # TODO: !! refactor into morpho
            vectorized_relabel = numpy.frompyfunc(remapping.__getitem__, 1, 1)
            label2_split = vectorized_relabel(label2_split).astype(numpy.uint64))

            # compress results
            return (subvolume, labelgt_map, label2_map,
                    CompressedNumpyArray(labelgt_split),
                    CompressedNumpyArray(label2_split))

        return label_pairs.mapValues(_split_disjoint)

    def calcoverlap(self, lpairs_split, boundary_iter=0):
        def _calcoverlap(label_pairs):
            subvolume, labelgt_map, label2_map, labelgtc, label2c = label_pairs
            
            # decompression
            labelgt = labelgtc.deserialize()
            label2 = label2c.deserialize()

            # TODO !! avoid conversion
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
        
            # compute VI and rand stats
            fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies, dummy = 
                                    calculate_vi(gt_overlap, seg_overlap)
            fmerge_rand, fsplit_rand = calculate_rand(gt_overlap, seg_overlap)
            
            # load substack stats
            stats.add_stat(ComparisonType(), StatType.VI, [fmerge_vi, fsplit_vi])
            stats.add_stat(ComparisonType(), StatType.RAND, [fmerge_rand, fsplit_rand])

            # add total size
            stats.add_stat(ComparisonType(), StatType.TOTALPTS, total_size)
            
            # add number of gt and seg bodies
            stats.add_stat(ComparisonType(), StatType.NUMBODIES,
                    len(gt_overlap.overlap_map), len(seg_overlap.overlap_map))

          
            # handle maps and load overlaps
            del_keys = []
            for key, val in gt_overlap.overlap_map.items():
                if key in labelgt_map:
                    del_keys.append(key)
                    gt_overlap.overlap_map[labelgt_map[key]] = val
            for key in del_keys:
                del gt_overlap.overlap_map[key]
            
            del_keys = []
            for key, val in seg_overlap.overlap_map.items():
                if key in label2_map:
                    del_keys.append(key)
                    seg_overlap.overlap_map[label2_map[key]] = val
            for key in del_keys:
                del seg_overlap.overlap_map[key]

            # add partial global results
            stats.add_gt_overlap(gt_overlap)
            stats.add_seg_overlap(seg_overlap)

            # TODO:!! Add more distribution stats 
            total_size = (subvolume.roi.x2 - subvolume.roi.x1) * (subvolume.roi.y2 - subvolume.roi.y1) * (subvolume.roi.z2 - subvolume.roi.z1)
            
           
            # keep volumes for subsequent point queries
            return (stats, labelgt_map, label2_map,
                    CompressedNumpyArray(labelgt), CompresedNumpyArray(label2))
            
        return lpairs_split.mapValues(_calcoverlap)


    def calcoverlap_pts(lpairs_split, point_list_name, point_data):
        # set type, name, sparse
        comparsison_type = ComparisonType(str(point_data["type"]),
                str(point_list_name), point_data["sparse"])

        # combine labels with relevant points 
        lpairs_split = lpairs_split.join(distpoints)
        
        def _calcoverlap_pts(label_pairs):
            stats, labelgt_map, label2_map, labelgtc, label2c, points = label_pairs
            
            # decompression
            labelgt = labelgtc.deserialize()
            label2 = label2c.deserialize()

            # for connected points, connections are encoded by using
            # the position index number X and extra numbers Y1,Y2... indicating
            # indicating that X -> Y1, Y2, etc
            # synapses encoded as PSD -> pre-synapse for parsing convenience
            adjacency_list = {}
    
            subvolume_pts = {}
            roi = stats.subvolume.roi
            # grab points that overlap
            for index, point in enumerate(points["point-list"]):
                if point[0] < roi.x2 and point[0] >= roi.x1 and point[1] < roi.y2 and point[1] >= roi.y2 and point[2] < roi.z2 and point[2] >= roi.z1:
                    subvolume_pts[index] = [point[0]-roi.x1, point[1]-roi.y1, point[2]-roi.z1]
                    adjacency_list[index] = set()
                    for iter1 in range(3, len(point)):
                        adjacency_list[index].add(iter1)

            # find overlap for set of points (density should be pretty low)
            overlap_gt = []
            overlap_seg = []

            temp_overlap_gt = {}
            temp_overlap_seg = {}
            
            for index, point in subvolume_pts.items():
                # z,y,x -- c order
                gtbody = labelgt[(point[2], point[1], point[0])]
                segbody = label2[(point[2], point[1], point[0])]

                if (gtbody, segbody) not in temp_overlap_gt:
                    temp_overlap_gt[(gtbody,segbody)] = 0
                temp_overlap_gt[(gtbody,segbody)] += 1
                
                if (segbody, gtbody) not in temp_overlap_seg:
                    temp_overlap_seg[(segbody, gtbody)] = 0
                temp_overlap_seg[(segbody, gtbody)] += 1

            for (gt, seg), overlap in temp_overlap_gt.items():
                overlap_gt.append((gt,seg,overlap))

            for (seg, gt), overlap in temp_overlap_seg.items():
                overlap_seg.append((seg,gt,overlap))

            gt_overlap = OverlapStats(overlap_gt, comparison_type)
            seg_overlap = OverlapStats(overlap_seg, comparison_type)

            # compute VI and rand stats
            fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies, dummy = 
                                    calculate_vi(gt_overlap, seg_overlap)
            fmerge_rand, fsplit_rand,  =
                                    calculate_rand(gt_overlap, seg_overlap)

            # load substack stats
            stats.add_stat(comparison_type, StatType.VI, [fmerge_vi, fsplit_vi])
            stats.add_stat(comparison_type, StatType.RAND, [fmerge_rand, fsplit_rand])

             # add total size
            stats.add_stat(comparison_type, StatType.TOTALPTS, len(points))
            
            # add number of gt and seg bodies
            stats.add_stat(ComparisonType(), StatType.NUMBODIES,
                    len(gt_overlap.overlap_map), len(seg_overlap.overlap_map))

            # handle maps and load overlaps
            del_keys = []
            for key, val in gt_overlap.overlap_map.items():
                if key in labelgt_map:
                    del_keys.append(key)
                    gt_overlap.overlap_map[labelgt_map[key]] = val
            for key in del_keys:
                del gt_overlap.overlap_map[key]
            
            del_keys = []
            for key, val in seg_overlap.overlap_map.items():
                if key in label2_map:
                    del_keys.append(key)
                    seg_overlap.overlap_map[label2_map[key]] = val
            for key in del_keys:
                del seg_overlap.overlap_map[key]

            # add partial global results
            stats.add_gt_overlap(gt_overlap)
            stats.add_seg_overlap(seg_overlap)

            # ?! if synapse type load total pair matches (save boundary point deps) 

            # points no longer needed
            return (stats, labelgt_map, label2_map,
                    CompressedNumpyArray(labelgt), CompresedNumpyArray(label2))

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

        # ?! dump body sizes (could be useful for selecting which bodies to consider)
        # (ComparisonType) -> bodyid -> size

        # ?! find filter thresholds for each comparison type

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


