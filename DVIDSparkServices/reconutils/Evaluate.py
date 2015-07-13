import libNeuroProofMetrics as np
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
import numpy
from segstats import *

# TODO!! add stat jsonschema

class Evaluate(object):
    def __init__(self, config):
        self.server = config["dvid-info"]["dvid-server"]
        self.uuid = config["dvid-info"]["uuid"]

    def extact_distribution(self, body_overlap):
        count = 0
        cumdisttemp = []
        for body, overlapset in body_overlap.items():
            for (body2, overlap) in overlapset:
                count += overlap
                cumdisttemp.append(overlap)

        cumdist = [for val in cumdist: val/count]
        cumdist.sort()
        cumdist.reverse()
        return cumdist

    def extract_subvolume_connections(self, index2body_seg, parent_list, adjacency_list):
        leftover_seg = {}
        seg_connections = {}
        prop_indices = {}

        for index, body in index2body_seg.items():
            # propagate indices that have unassigned parent
            if index in parent_list and parent_list[index] not in index2body_seg:
                prop_indices[index] = body

            for index2 in adjacency_list[index]:
                if index2 not in index2body_seg:
                    # resolve index later in reduce
                    if body not in leftover_seg:
                        leftover_seg[body] = set()
                    leftover_seg[body].add(index2)
                else:
                    # add connection
                    body2 = index2body_seg[index2]
                    if (body, body2) not in seg_connections:
                        seg_connections[(body1, body2)] = 0
                    seg_connections[(body1,body2)] += 1

        # put in same format as other overlap structures
        seg_overlap_syn = []
        for (body1, body2), overlap in seg_connections.items():
            seg_overlap_syn.append((body1, body2, overlap))
   
        return seg_overlap_syn, (leftover_seg, prop_indices)


    def load_subvolume_stats(self, stats, overlap_gt, overlap_seg, comparison_type,
               labelgt_map, label2_map):
        gt_overlap = OverlapStats(overlap_gt, comparison_type)
        seg_overlap = OverlapStats(overlap_seg, comparison_type)

        # compute VI and rand stats
        fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies, dummy = 
                                calculate_vi(gt_overlap, seg_overlap)
        fmerge_rand, fsplit_rand  =
                                calculate_rand(gt_overlap, seg_overlap)
        
        # load substack stats
        stats.add_stat(VIStat(comparison_type, fmerge_vi, fsplit_vi))
        stats.add_stat(RandStat(comparison_type, fmerge_rand, fsplit_rand))

        volume_size = 0
        # find total volume being compared
        for key, valset in gt_overlap.overlap_map.items():
            for (key, val2) in valset:
                volume_size += volume_size
        
         # add total size
        stats.add_stat(TotalStat(comparison_type, volume_size))
       
        # add number of gt and seg bodies
        stats.add_stat(NumBodyStat(comparison_type, len(gt_overlap.overlap_map),
            len(seg_overlap.overlap_map)))

        # handle maps and load overlaps
        del_keys = []
        for key, val in gt_overlap.overlap_map.items():
            if key in labelgt_map:
                del_keys.append(key)
            del_keys2 = []
            for (key2, val2) in val:
                if key2 in label2_map:
                    del_keys2.append((key2,val2))
            for (key2, val2) in del_keys2:
                del val[(key2,val2)]
                val.add((label2_map[key2],val2))
        for key in del_keys:
            val = gt_overlap.overlap_map[key]
            del gt_overlap.overlap_map[key]
            gt_overlap.overlap_map[labelgt_map[key]] = val

        del_keys = []
        for key, val in label_overlap.overlap_map.items():
            if key in label_map:
                del_keys.append(key)
            del_keys2 = []
            for (key2, val2) in val:
                if key2 in labelgt_map:
                    del_keys2.append((key2,val2))
            for (key2, val2) in del_keys2:
                del val[(key2,val2)]
                val.add((labelgt_map[key2],val2))
        for key in del_keys:
            val = label_overlap.overlap_map[key]
            del label_overlap.overlap_map[key]
            label_overlap.overlap_map[label_map[key]] = val

        # add partial global results
        stats.add_gt_overlap(gt_overlap)
        stats.add_seg_overlap(seg_overlap)


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
            
            # ?? should I return stats back
            self.load_subvolume_stats(stats, overlap12, overlap21,
                    ComparisonType(), labelgt_map, label2_map)

            # TODO:!! Add more distribution stats 
            
           
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
            # synapses encoded as presynapse -> PSD
            adjacency_list = {}
    
            # unique parents for each point
            parent_list = {}

            subvolume_pts = {}
            roi = stats.subvolume.roi
            # grab points that overlap
            for index, point in enumerate(points["point-list"]):
                if point[0] < roi.x2 and point[0] >= roi.x1 and point[1] < roi.y2 and point[1] >= roi.y2 and point[2] < roi.z2 and point[2] >= roi.z1:
                    subvolume_pts[index] = [point[0]-roi.x1, point[1]-roi.y1, point[2]-roi.z1]
                    adjacency_list[index] = set()
                    for iter1 in range(3, len(point)):
                        adjacency_list[index].add(iter1)
                        parent_list[iter1] = index


            # find overlap for set of points (density should be pretty low)
            overlap_gt = []
            overlap_seg = []

            temp_overlap_gt = {}
            temp_overlap_seg = {}
        
            index2body_gt = {}
            index2body_seg = {}

            for index, point in subvolume_pts.items():
                # z,y,x -- c order
                # get bodies on points
                gtbody = labelgt[(point[2], point[1], point[0])]
                segbody = label2[(point[2], point[1], point[0])]
           
                # get point index to actual global body 
                gtbody_mapped = gtbody
                if gtbody in labelgt_map:
                    gtbody_mapped = labelgt_map[gtbody]
                segbody_mapped = segbody
                
                if segbody in label2_map:
                    segbody_mapped = label2_map[segbody]
                
                index2body_gt[index] = gtbody_mapped
                index2body_seg[index] = segbody_mapped

                # load overlaps
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

            self.load_subvolume_stats(stats, overlap_gt, overalp_seg,
                    comparison_type, labelgt_map, label2_map)

            # if synapse type load total pair matches (save boundary point deps) 
            if point_data["type"] == "synapse":
                # grab partial connectivity graph for gt
                gt_overlap_syn, leftover_gt = 
                    self.extract_subvolume_connections(index2body_gt, parent_list, adjacency_list)
            
                # add custom synapse type
                stats.add_gt_syn_connections(OverlapStats(gt_overlap_syn,
                        ComparisonType("synapse-graph", str(point_list_name],
                        point_data["sparse"), leftover_gt)

                # grab partial connectivity graph for seg
                seg_overlap_syn, leftover_seg = 
                    self.extract_subvolume_connections(index2body_seg, parent_list, adjacency_list)
                
                # add custom synapse type
                stats.add_seg_syn_connections(OverlapStats(seg_overlap_syn,
                        ComparisonType("synapse-graph", str(point_list_name],
                        point_data["sparse"), leftover_seg)

            # points no longer needed
            return (stats, labelgt_map, label2_map, labelgtc, label2c)

        return lpairs_split.mapValues(_calcoverlap_pts)

    def calculate_stats(lpairs_splits):
        # extract subvolume stats and send to the driver
        all_stats.map(lamda x: x[1][0]).collect()
        
        metric_results = {}

        # accumulate all subvolume stats
        whole_volume_stats = SubvolumeOverlapStats(None)

        ##### SUBSTACK STATS ######
        
        allsubvolume_metrics = {}
        allsubvolume_metrics["types"] = {}
        allsubvolume_metrics["ids"] = {}

        for subvol_stats in all_stats:
            subvolume_metrics = {}
            # load substack location/size info 
            subvolume_metrics["roi"] = subvol_stats.subvolume.roi
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
                stat.write_to_dict(subvolume_metrics["types"], allsubvolume_metrics, subvol_stats.subvolume.roi_id)

            allsubvolume_metrics["ids"][subvol_stats.subvolume.roi_id] = subvolume_metrics
            
            # accumulate stats
            whole_volume_stats.merge_stats(subvol_stat)

        # verify that leftover list is empty
        for connections in whole_volume_stats.gt_syn_connections:
            stats, leftovers, props = connections
            if len(leftovers) != 0:
                raise Exception("Synapse leftovers are non-empty")
        for connections in whole_volume_stats.seg_syn_connections:
            stats, leftovers, props = connections
            if len(leftovers) != 0:
                raise Exception("Synapse leftovers are non-empty")

        # store subvolume results 
        metric_results['subvolumes'] = allsubvolume_metrics

        # write results for each comparison type
        comparison_type_metrics = {}

        # TODO: compute vi/rand metrics with bottom 10 percent of 'volume' ignored
        # should help with noise primarily (other noise filters??)

        for iter1 in range(0, len(whole_volume_stats.gt_overlaps)):
            ##### HISTOGRAM STATS #######
            distgt = extract_distribution(whole_volume_stats.gt_overlaps[iter1].overlap_map)
            distseg = extract_distribution(whole_volume_stats.seg_overlaps[iter1].overlap_map)
            # check for comparison and create if necessary
            comptype = whole_volume_stats.gt_overlaps[iter1].comparison_type
            typename = comptype.typename+":"+comptype.instance_name
            if typename not in comparison_type_metrics:
                comparison_type_metrics[typename] = {}

            # TODO !! take subset of distribution
            comparison_type_metrics[typename]["dist-gt"] = distgt
            comparison_type_metrics[typename]["dist-seg"] = distseg

            # TODO !! generate statitcs histogram

            ###### AGGREGATE STATS #######
            gt_overlap = whole_volume_stats.gt_overlaps[iter1].overlap_map
            seg_overlap = whole_volume_stats.seg_overlaps[iter1].overlap_map
            
            # TODO !! Add normalized per body vi by body size
            fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies, vi_bodies = 
                                    calculate_vi(gt_overlap, seg_overlap)
            fmerge_rand, fsplit_rand  =
                                    calculate_rand(gt_overlap, seg_overlap)
           
            # add rand, vi global,  for each type
            statvi = VIStat(comptype, fmerge_vi, fsplit_vi)
            statrand = RandStat(comptype, fmerge_rand, fsplit_rand)
            statvi.write_to_dict(comparison_type_metrics)
            statrand.write_to_dict(comparison_type_metrics)


            ####### EDIT DISTANCE ######
            # provide only 90% threshold (TODO: consider other thresholds)
            # number of operations to get 90 percent correct for given comparison
            # will not mean as much for sparse points, but per-body volumetric
            # sparse should be reasonably good

            # Assumptions: 1) 90% is a good threshold -- probably should vary
            # depending on the biological goal and 2) greatest overlap is
            # sufficient for assigment

            # 1. Examine all test segs and find GT body it overlaps most
            # 2. Examine all GT bodies, accumulate eligible test seg that overlaps most
            # 3. Add eligible to merge list, add illegible to split list
            # 4. Try a few ratios for optimizations (10:1, 5:1, 1:1)
            # 5. Merge and split until 90% of volume is fixed

            # 1
            seg2bestgt = {}
            target = 0
            for body, overlapset in seg_overlap.items():
                max_val = 0
                max_id = 0
                for body2, overlap in overlapset:
                    target += overlap
                    if overlap > max_val:
                        max_val = overlap
                        max_id = body2
                seg2bestgt[max_id] = body2

            target *= 0.90 # consider 90 percent correctness
           
            # 2, 3
            sorted_splits = []
            sorted_mergers = []
            current_accum = 0
            for body, overlapset in gt_overlap.items():
                temp_merge = []
                for body2, overlap in overlapset:
                    # doesn't own body
                    if seg2bestgt[body2] != body:
                        sorted_splits.append(overlap)
                    else:
                        temp_merge.append(overlap)
                temp_merge.sort()
                temp_merge.reverse()
                current_accum += temp_merge[0]
                sorted_mergers.extend(temp_merge[1:])

            # actually sort lists for optimal traversal
            sorted_splits.sort()
            sorted_splits.reverse()
            sorted_mergers.sort()
            sorted_mergers.reverse()

            # 4
            comparison_type_metrics[typename]["edit-distance"] = {}
            for ratio in [1, 5, 10]:
                midx = 0
                sidx = 0
                current_accum_rat = current_accum

                while current_accum_rat < target:
                    take_split = False
                    
                    if midx == len(sorted_mergers):
                        take_split = True
                    elif sidx == len(sorted_splits):
                        pass
                    elif sorted_splits[sidx] / float(ratio) > sorted_mergers[midx]:
                        take_split = True

                    if take_split:
                        current_accum_rat += sorted_splits[sidx]
                        sidx += 1
                    else:
                        current_accum_rat += sorted_mergers[midx]
                        midx += 1

                # comparisontype -> edit-distance -> thres -> [merge, split]
                comparison_type_metrics[typename]["edit-distance"][ratio] = [midx, sidx]

            ###### PER BODY VI STATS ######
            
            # TODO!! Per body edit distance (probably pretty easy)

            # id -> [vifsplit, vitotal, size]
            comparison_type_metrics[typename]["gt-bodies"] = {}  # fsplit
            # id -> [vifmerge, vitotal, size]
            comparison_type_metrics[typename]["seg-bodies"] = {} # fmerge
           
            # examine gt bodies
            worst_gt_body = 0
            worst_gt_val = 0
            worst_fsplit = 0
            worst_fsplit_body = 0
            for body, overlapset in gt_overlap.items():
                total = 0
                for body2, overlap in overlapset:
                    total += overlap
                fsplit = fsplit_bodies[body]
                vitot = vi_bodies[body]
                comparison_type_metrics[typename]["gt-bodies"][body] = 
                        [fsplit, vitot, total]
                if fsplit > worst_fsplit:
                    worst_fsplit = fsplit
                    worst_fsplit_body = body
                if vitot >  worst_gt_val:
                    worst_gt_val = vitot
                    worst_gt_body = body

            comparison_type_metrics[typename]["worst-vi"] = [worst_gt_val, worst_gt_body]
            comparison_type_metrics[typename]["worst-fsplit"] = [worst_fsplit, worst_fsplit_body]
           
            # examine seg bodies
            worst_fmerge = 0
            worst_fmerge_body = 0
            for body, overlapset in seg_overlap.items():
                total = 0
                for body2, overlap in overlapset:
                    total += overlap
                fsplit = fsplit_bodies[body]
                vitot = vi_bodies[body]
                comparison_type_metrics[typename]["seg-bodies"][body] = 
                        [fsplit, vitot, total]
                
                if fmerge > worst_fmerge:
                    worst_fmerge = fmerge
                    worst_fmerge_body = body

            comparison_type_metrics[typename]["worst-fmerge"] = [worst_fmerge, worst_fmerge_body]
        
        # ?! look for synapse points and compute graph measure (different cut-offs?)


        metric_results["types"] = comparison_type_metrics
        return metric_results


