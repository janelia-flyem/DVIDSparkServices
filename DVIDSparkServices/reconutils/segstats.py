from math import log

class ComparisonType(object):
    def __init__(self, typename="voxels", instance_name="voxels", sparse=False):
        self.typename = typename
        self.instance_name = instance_name
        self.sparse = sparse

    def get_name(self):
        return self.typename + ":" + self.instance_name

    def __str__(self):
        return self.get_name()

    def __eq__(self, other):
        return self.typename == other.typename and self.instance_name == other.instance_name and self.sparse == other.sparse

    def __ne__(self, other):
        return self.typename != other.typename or self.instance_name != other.instance_name or self.sparse != other.sparse

class StatType:
    """
    VI = 1
    RAND = 2
    TOTALPTS = 3
    NUMBODIES = 4
    """
    def __init__(self, comptype):
        self.comptype = comptype

    def write_to_dict(self, localmap, globalmap = None, sid=0):
        pass

class TotalStat(StatType):
    def __init__(self, comptype, total):
        StatType.__init__(self, comptype)
        self.total = total

    def write_to_dict(self, localmap, globalmap = None, sid=0):
        tname = self.comptype.get_name()
        if tname not in localmap:
            localmap[tname] = {}
        # no average stats here
        localmap[tname]["size"] =  self.total

class NumBodyStat(StatType):
    def __init__(self, comptype, n1, n2):
        StatType.__init__(self, comptype)
        self.n1 = n1
        self.n2 = n2

    def write_to_dict(self, localmap, globalmap = None, sid=0):
        tname = self.comptype.get_name()
        if tname not in localmap:
            localmap[tname] = {}
        # no average stats here
        localmap[tname]["num-gt-bodies"] = self.n1
        localmap[tname]["num-seg-bodies"] = self.n2

class VIStat(StatType):
    def __init__(self, comptype, fmerge, fsplit):
        StatType.__init__(self, comptype)
        self.fmerge = fmerge
        self.fsplit = fsplit

    def write_to_dict(self, localmap, globalmap = None, sid=0):
        tname = self.comptype.get_name()
        if tname not in localmap:
            localmap[tname] = {}

        localmap[tname]["VI"] =  [self.fmerge, self.fsplit]
        if globalmap is not None:
            if tname not in globalmap["types"]:
                # no stat has created the type yet
                globalmap["types"][tname] = {}
            if "VI" not in globalmap["types"][tname]:
                # VI stat has not been considered yet
                globalmap["types"][tname]["VI"] = {}
                globalmap["types"][tname]["VI"]["fmerge-best"] = [self.fmerge, sid] 
                globalmap["types"][tname]["VI"]["fsplit-best"] = [self.fsplit, sid] 
                globalmap["types"][tname]["VI"]["fmerge-worst"] = [self.fmerge, sid] 
                globalmap["types"][tname]["VI"]["fsplit-worst"] = [self.fsplit, sid] 
                globalmap["types"][tname]["VI"]["average"] = [self.fmerge, self.fsplit]
            else:
                # get best and worst
                best = globalmap["types"][tname]["VI"]["fmerge-best"][0]
                worst = globalmap["types"][tname]["VI"]["fmerge-worst"][0]
                if self.fmerge < best:
                    globalmap["types"][tname]["VI"]["fmerge-best"] = [self.fmerge,sid]
                if self.fmerge > worst:
                    globalmap["types"][tname]["VI"]["fmerge-worst"] = [self.fmerge,sid]

                best = globalmap["types"][tname]["VI"]["fsplit-best"][0]
                worst = globalmap["types"][tname]["VI"]["fsplit-worst"][0]
                if self.fsplit < best:
                    globalmap["types"][tname]["VI"]["fsplit-best"] = [self.fsplit,sid]
                if self.fsplit > worst:
                    globalmap["types"][tname]["VI"]["fsplit-worst"] = [self.fsplit,sid]

                # get average
                num_subvolumes = len(globalmap["ids"])
                fmerge = globalmap["types"][tname]["VI"]["average"][0] 
                fsplit = globalmap["types"][tname]["VI"]["average"][1]

                fmerge = (num_subvolumes*fmerge + self.fmerge)/float(num_subvolumes+1)
                fsplit = (num_subvolumes*fsplit + self.fsplit)/float(num_subvolumes+1)

                globalmap["types"][tname]["VI"]["average"] = [fmerge, fsplit]

class RandStat(StatType):
    def __init__(self, comptype, fmerge, fsplit):
        StatType.__init__(self, comptype)
        self.fmerge = fmerge
        self.fsplit = fsplit

    def write_to_dict(self, localmap, globalmap = None, sid=0):
        tname = self.comptype.get_name()
        if tname not in localmap:
            localmap[tname] = {}

        localmap[tname]["rand"] =  [self.fmerge, self.fsplit]
        if globalmap is not None:
            if tname not in globalmap["types"]:
                # no stat has created the type yet
                globalmap["types"][tname] = {}
            if "rand" not in globalmap["types"][tname]:
                # rand stat has not been considered yet
                globalmap["types"][tname]["rand"] = {}
                globalmap["types"][tname]["rand"]["fmerge-best"] = [self.fmerge, sid] 
                globalmap["types"][tname]["rand"]["fsplit-best"] = [self.fsplit, sid] 
                globalmap["types"][tname]["rand"]["fmerge-worst"] = [self.fmerge, sid] 
                globalmap["types"][tname]["rand"]["fsplit-worst"] = [self.fsplit, sid] 
                globalmap["types"][tname]["rand"]["average"] = [self.fmerge, self.fsplit]
            else:
                # get best and worst
                best = globalmap["types"][tname]["rand"]["fmerge-best"][0]
                worst = globalmap["types"][tname]["rand"]["fmerge-worst"][0]
                if self.fmerge > best:
                    globalmap["types"][tname]["rand"]["fmerge-best"] = [self.fmerge,sid]
                if self.fmerge < worst:
                    globalmap["types"][tname]["rand"]["fmerge-worst"] = [self.fmerge,sid]

                best = globalmap["types"][tname]["rand"]["fsplit-best"][0]
                worst = globalmap["types"][tname]["rand"]["fsplit-worst"][0]
                if self.fsplit > best:
                    globalmap["types"][tname]["rand"]["fsplit-best"] = [self.fsplit,sid]
                if self.fsplit < worst:
                    globalmap["types"][tname]["rand"]["fsplit-worst"] = [self.fsplit,sid]

                # get average
                num_subvolumes = len(globalmap["ids"])
                fmerge = globalmap["types"][tname]["rand"]["average"][0] 
                fsplit = globalmap["types"][tname]["rand"]["average"][1]

                fmerge = (num_subvolumes*fmerge + self.fmerge)/float(num_subvolumes+1)
                fsplit = (num_subvolumes*fsplit + self.fsplit)/float(num_subvolumes+1)

                globalmap["types"][tname]["rand"]["average"] = [fmerge, fsplit]


class EditDistanceStat(StatType):
    def __init__(self, comptype, gt_overlap, seg_overlap):
        StatType.__init__(self, comptype)
        self.results = []

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
            seg2bestgt[body] = max_id

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
            if len(temp_merge) > 0:
                current_accum += temp_merge[0]
            if len(temp_merge) > 1:
                sorted_mergers.extend(temp_merge[1:])

        # actually sort lists for optimal traversal
        sorted_splits.sort()
        sorted_splits.reverse()
        sorted_mergers.sort()
        sorted_mergers.reverse()

        # 4
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
                elif (sorted_splits[sidx] / float(ratio)) > sorted_mergers[midx]:
                    take_split = True

                if take_split:
                    current_accum_rat += sorted_splits[sidx]
                    sidx += 1
                else:
                    current_accum_rat += sorted_mergers[midx]
                    midx += 1

            self.results.append((ratio, midx, sidx))

    def write_to_dict(self, localmap, globalmap = None, sid=0):
        tname = self.comptype.get_name()
        if tname not in localmap:
            localmap[tname] = {}
        localmap[tname]["edit-distance"] = {} 
        
        # comparisontype -> edit-distance -> thres -> [merge, split]
        for (ratio, midx, sidx) in self.results:
            localmap[tname]["edit-distance"][ratio] = [midx, sidx]

class OverlapTable(object):
    def __init__(self, overlaps, comparison_type):
        self.comparison_type = comparison_type
        self.overlap_map = {}

        for item in overlaps:
            body1, body2, overlap = item
            if body1 not in self.overlap_map:
                self.overlap_map[body1] = set()
            self.overlap_map[body1].add((body2, overlap))

    def get_size(self):
        size = 0
        for key, valset in self.overlap_map.items():
            for (key2, overlap) in valset:
                size += overlap
        return size

    def merge_row(self, row1, row2):
        """Merge row2 to row1, update overlap.

        Args:
            row1 (set): set of (body, overlap)
            row2 (set): set of (body, overlap)
        
        """

        duprow = list(row1)
        duprow.extend(list(row2))
        row1.clear()
        overlap_map = {}

        for body, overlap in duprow:
            if body not in overlap_map:
                overlap_map[body] = 0
            overlap_map[body] += overlap

        for body, overlap in overlap_map.items():
            row1.add((body, overlap))

    def partial_remap(self, mapping1, mapping2):
        """Partial remap the overlap table.

        Use mapping1 to remap a subset of the table inputs
        and mapping2 to remap the overlapped outputs for
        each input.

        Note:
            remapped labels must be disjoint from original labels.
        
        Args:
            mapping1 (dict): label -> new label (for inputs)
            mapping2 (dict): label -> new label (for outputs)

        Returns:
            Internal structures are updated.

        """
        
        del_keys = {}
        for key, val in self.overlap_map.items():
            if key in mapping1:
                # find inputs that need to be remapped
                if mapping1[key] not in del_keys:
                    del_keys[mapping1[key]] = []
                del_keys[mapping1[key]].append(key)

            new_overlap = {}

            # handle new overlaps since mapping could cause merge
            for (key2, val2) in val:
                new_key = key2
                if key2 in mapping2:
                    new_key = mapping2[key2]
                if new_key not in new_overlap:
                    new_overlap[new_key] = 0
                new_overlap[new_key] += val2
            
            # update overlap list
            new_overlap_set = set()
            for body, overlap in new_overlap.items():
                new_overlap_set.add((body, overlap)) 
            self.overlap_map[key] = new_overlap_set
    
        # merge rows mapping to same body, remove old body
        for newbody, bodylist in del_keys.items():
            self.overlap_map[newbody] = set()
            for bodyold in bodylist:
                self.merge_row(self.overlap_map[newbody], self.overlap_map[bodyold])
                del self.overlap_map[bodyold]

    def num_bodies(self):
        return len(self.overlap_map)

    def combine_tables(self, overlap2):
        if self.comparison_type != overlap2.comparison_type:
            raise Exception("incomparable overlap stat types")

        for body, overlapset in overlap2.overlap_map.items():
            if body not in self.overlap_map:
                self.overlap_map[body] = overlapset
            else:
                self.merge_row(self.overlap_map[body], overlap2.overlap_map[body])

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

def get_body_volume(overlapset):
    total = 0
    for body, overlap in overlapset:
        total += overlap
    return total

# calculate Variation of Information metric
def calculate_vi(gtoverlap, segoverlap, body_threshold = 0):
    fsplit_bodies = {}
    fmerge_bodies = {}

    # determine how bad a given body is
    perbody = {}

    glb_total = 0
    fmerge_vi = 0
    fsplit_vi = 0

    ignore_bodies = set()

    # examine fragmentation of gt (fsplit=oversegmentation)
    for (gtbody, overlapset) in gtoverlap.overlap_map.items():
        if get_body_volume(overlapset) < body_threshold:
            # ignore as if it never existed
            ignore_bodies.add(gtbody)
            continue

        vi_unnorm, total, dummy = body_vi(overlapset)
        fsplit_bodies[gtbody] = vi_unnorm
        perbody[gtbody] = vi_unnorm
        glb_total += total
        fmerge_vi += vi_unnorm

    # examine fragmentation of seg (fmerge=undersegmentation)
    for (segbody, overlapset) in segoverlap.overlap_map.items():
        # filter small bodies
        filtered_overlapset = set()
        for (gtbody, overlap) in overlapset:
            if gtbody not in ignore_bodies:
                filtered_overlapset.add((gtbody, overlap))

        vi_unnorm, total, gtcontribs = body_vi(filtered_overlapset)
        fmerge_bodies[segbody] = vi_unnorm
        fsplit_vi += vi_unnorm

        for key, val in gtcontribs.items():
            perbody[key] += val

    for key, val in fsplit_bodies.items():
        fsplit_bodies[key] = val/float(glb_total)
    
    for key, val in fmerge_bodies.items():
        fmerge_bodies[key] = val/float(glb_total)
    
    for key, val in perbody.items():
        perbody[key] = val/float(glb_total)

    # TODO !! Add per body
    if glb_total == 0:
        return 0, 0, fmerge_bodies, fsplit_bodies, perbody

    return fmerge_vi/float(glb_total), fsplit_vi/float(glb_total), fmerge_bodies, fsplit_bodies, perbody 

# calculate Rand Index
def calculate_rand(gtoverlap, segoverlap, body_threshold=0):
    fsplit_total = 0
    overlap_total = 0
    ignore_bodies = set()
    
    # examine fragmentation of gt (fsplit=oversegmentation)
    for (gtbody, overlapset) in gtoverlap.overlap_map.items():
        if get_body_volume(overlapset) < body_threshold:
            # ignore as if it never existed
            ignore_bodies.add(gtbody)
            continue
        total = 0
        for (segid, overlap) in overlapset:
            total += overlap
            overlap_total += (overlap*(overlap-1)/2)

        fsplit_total += (total*(total-1)/2)

    fmerge_total = 0
    # examine fragmentation of seg (fmerge=undersegmentation)
    for (gtbody, overlapset) in segoverlap.overlap_map.items():
        # filter small bodies
        filtered_overlapset = set()
        for (gtbody, overlap) in overlapset:
            if gtbody not in ignore_bodies:
                filtered_overlapset.add((gtbody, overlap))
        
        total = 0
        for (segid, overlap) in filtered_overlapset:
            total += overlap

        fmerge_total += (total*(total-1)/2)

    merge = 1
    split = 1

    if fmerge_total != 0:
        merge  = overlap_total / float(fmerge_total)
    if  fsplit_total != 0:
        split = overlap_total / float(fsplit_total)
    return merge, split

class SubvolumeStats(object):
    def __init__(self, subvolume):
        self.subvolume = subvolume 

        # contains "rand", "vi", etc for substack
        self.subvolume_stats = []

        # contains overlaps over the various sets
        # of overlap computed
        self.gt_overlaps = []
        self.seg_overlaps = []

        # support multiple synapse points but probably
        # not needed
        self.gt_syn_connections = []
        self.seg_syn_connections = []


    def merge_syn_connections(self, connections1, connections2):
        for iter1 in range(0, len(connections1)):
            table1, (leftover1, prop1) = connections1[iter1]
            table2, (leftover2, prop2) = connections2[iter1]
        
            table1.combine_tables(table2)
            prop1.update(prop2)

            for body, indexset in leftover2.items():
                if body not in leftovers1:
                    leftover1[body] = indexset
                else:
                    leftover1[body].union(indexset)

            new_leftovers = {} 
            # try to resolve more unknown values
            for body, indexset in leftover1.items():
                for index in indexset:
                    if index in prop1:
                        if body not in table1.overlap_map:
                            # add one point
                            table1.overlap_map[body] = set((prop1[index],1))
                        else:
                            rm_overlap = None
                            for (body2, overlap) in table1.overlap_map[body]:
                                if body2 == prop1[index]:
                                    rm_overlap = (body2, overlap)
                                    break
                            overlap = 0
                            if rm_overlap is not None:
                                table1.overlap_map[body].remove(rm_overlap)
                                overlap = rm_overlap[1]
                            table1.overlap_map[body].add((prop1[index], overlap+1))
                    else:
                        # repopulate
                        if body not in new_leftovers:
                            new_leftovers[body] = set()
                        new_leftovers[body].add(index)
            
            # update list
            connections1[iter1] = (table1, (new_leftovers, prop1))



    # drops subvolume stats and subvolume
    def merge_stats(self, subvolume):
        self.subvolume = None
        self.subvolume_stats = None

        if len(self.gt_overlaps) != len(subvolume.gt_overlaps):
            raise Exception("incomparable subvolumes")
        if len(self.seg_overlaps) != len(subvolume.seg_overlaps):
            raise Exception("incomparable subvolumes")
        if len(self.gt_syn_connections) != len(subvolume.gt_syn_connections):
            raise Exception("incomparable subvolumes")
        if len(self.seg_syn_connections) != len(subvolume.seg_syn_connections):
            raise Exception("incomparable subvolumes")

        for iter1 in range(0, len(self.gt_overlaps)):
            self.gt_overlaps[iter1].combine_tables(subvolume.gt_overlaps[iter1])           
        for iter1 in range(0, len(self.seg_overlaps)):
            self.seg_overlaps[iter1].combine_tables(subvolume.seg_overlaps[iter1])           
        self.merge_syn_connections(self.gt_syn_connections,
                subvolume.gt_syn_connections) 
        
        self.merge_syn_connections(self.seg_syn_connections,
                subvolume.seg_syn_connections) 

    def add_gt_syn_connections(self, stats, leftover):
        self.gt_syn_connections.append((stats, leftover))
    
    def add_seg_syn_connections(self, stats, leftover):
        self.seg_syn_connections.append((stats, leftover))

    def add_gt_overlap(self, table):
        self.gt_overlaps.append(table)
    
    def add_seg_overlap(self, table):
        self.seg_overlaps.append(table)

    def add_stat(self, value):
        self.subvolume_stats.append(value)


