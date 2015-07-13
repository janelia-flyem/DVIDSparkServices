from math import log

class ComparisonType(object):
    def __init__(self, typename="voxels", instance_name="voxels", sparse=False):
        self.typename = typename
        self.instance_name = instance_name
        self.sparse = sparse

    def get_name(self):
        return self.typename + ":" + self.instance_name

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

    def write_to_dict(self):
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
        localmap[tname]["num-gt-bodies"] = n1
        localmap[tname]["num-seg-bodies"] = n2

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
                num_subvolumes = len(globalmap)
                fmerge = globalmap["types"][tname]["VI"]["average"][0] 
                fsplit = globalmap["types"][tname]["VI"]["average"][1]

                fmerge = (num_subvolumes*fmerge + self.fmerge)/(num_subvolumes+1)
                fsplit = (num_subvolumes*fmerge + self.fmerge)/(num_subvolumes+1)

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
                num_subvolumes = len(globalmap)
                fmerge = globalmap["types"][tname]["rand"]["average"][0] 
                fsplit = globalmap["types"][tname]["rand"]["average"][1]

                fmerge = (num_subvolumes*fmerge + self.fmerge)/(num_subvolumes+1)
                fsplit = (num_subvolumes*fmerge + self.fmerge)/(num_subvolumes+1)

                globalmap["types"][tname]["rand"]["average"] = [fmerge, fsplit]


class OverlapStats(object):
    def __init__(self, overlaps, comarison_type):
        self.comparison_type = comparison_type
        self.overlap_map = {}

        for item in overlaps:
            body1, body2, overlap = item
            if body1 not in self.overlap_map:
                self.overlap_map[body1] = set()
            self.overlap_map[body1].add((body2, overlap))

    def combine_stats(self, overlap2):
        if self.comparison_type != overlap2.comparison_type:
            raise Exception("incomparable overlap stat types")

        for body, overlapset in overlap2.overlap_map.items():
            if body not in self.overlap_map:
                self.overlap_map[body] = overlapset
            else:
                newo = {}
                for (body2, overlap) in overlapset:
                    newo[body2] = overlap
                for (body2, overlap) in self.overlap_map[body]:
                    if body2 in newo:
                        newo[body2] += overlap
                    else:
                        newo[body2] = overlap
                self.overlap_map[body] = set()
                for body2, overlap in newo.items():
                    self.overlap_map[body].add((body2, overlap))


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

    # determine how bad a given body is
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

    # TODO !! Add per body
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

        # support multiple synapse points but probably
        # not needed
        self.gt_syn_connections = []
        self.seg_syn_connections = []

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
            self.gt_overlaps[iter1].combine_stats(subvolume.gt_overlaps[iter1])           
        for iter1 in range(0, len(self.seg_overlaps)):
            self.seg_overlaps[iter1].combine_stats(subvolume.seg_overlaps[iter1])           
        for iter1 in range(0, len(self.gt_syn_connections):
            stats1, (leftover1, prop1) = self.gt_syn_connections[iter1]
            stats2, (leftover2, prop2) = subvolume.gt_syn_connections[iter1]
        
            stats1.combine_stats(stats2)
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
                        if body not in stats1.overlap_map:
                            # add one point
                            stats1.overlap_map[body] = set((prop1[index],1))
                        else:
                            rm_overlap = None
                            for (body2, overlap) in stats1.overlap_map[body]:
                                if body2 == prop1[index]:
                                    rm_overlap = (body2, overlap)
                                    break
                            overlap = 0
                            if rm_overlap is not None:
                                stats1.overlap_map[body].remove(rm_overlap)
                                overlap = rm_overlap[1]
                            stats1.overlap_map[body].add((prop1[index], overlap+1))
                    else:
                        # repopulate
                        if body not in new_leftovers:
                            new_leftovers[body] = set()
                        new_leftovers[body].add(index)
            
            # update list
            self.gt_syn_connections[iter1] = (stats1, (new_leftovers, prop1))

        for iter1 in range(0, len(self.gt_syn_connections):
            stats1, (leftover1, prop1) = self.gt_syn_connections[iter1]
            stats2, (leftover2, prop2) = subvolume.gt_syn_connections[iter1]
        
            stats1.combine_stats(stats2)
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
                        if body not in stats1.overlap_map:
                            # add one point
                            stats1.overlap_map[body] = set((prop1[index],1))
                        else:
                            rm_overlap = None
                            for (body2, overlap) in stats1.overlap_map[body]:
                                if body2 == prop1[index]:
                                    rm_overlap = (body2, overlap)
                                    break
                            overlap = 0
                            if rm_overlap is not None:
                                stats1.overlap_map[body].remove(rm_overlap)
                                overlap = rm_overlap[1]
                            stats1.overlap_map[body].add((prop1[index], overlap+1))
                    else:
                        # repopulate
                        if body not in new_leftovers:
                            new_leftovers[body] = set()
                        new_leftovers[body].add(index)
            
            # update list
            self.gt_syn_connections[iter1] = (stats1, (new_leftovers, prop1))




    def add_gt_syn_connections(self, stats, leftover):
        self.gt_syn_connections.append((stats, leftover))
    
    def add_seg_syn_connections(self, stats, leftover):
        self.seg_syn_connections.append((stats, leftover))

    def add_gt_overlap(self, stats):
        self.gt_overlaps.append(stats)
    
    def add_seg_overlap(self, stats):
        self.seg_overlaps.append(stats)

    def add_stat(self, value)
        self.subvolume_stats.append(value)



        for subvolume.gt_overlaps
        

    def add_gt_syn_connections(self, stats, leftover):
        self.gt_syn_connections.append((stats, leftover))
    
    def add_seg_syn_connections(self, stats, leftover):
        self.seg_syn_connections.append((stats, leftover))

    def add_gt_overlap(self, stats):
        self.gt_overlaps.append(stats)
    
    def add_seg_overlap(self, stats):
        self.seg_overlaps.append(stats)

    def add_stat(self, value)
        self.subvolume_stats.append(value)


