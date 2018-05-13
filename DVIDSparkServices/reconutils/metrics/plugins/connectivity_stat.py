from __future__ import division
from scipy.optimize import linear_sum_assignment
import numpy
from .stat import *

"""Class provides connectivity stats based on synapse annotations.

Note: assume that there is only one "synapse" type
"""
class connectivity_stat(StatType):
    def __init__(self, thresholds = [1, 5, 10]):
        super(connectivity_stat, self).__init__()
        self.thresholds = thresholds
        
    def write_summary_stats(self):
        """Write stats for the volume.
        """

        # disable if non-gt mode
        if self.segstats.nogt:
            return []
        
        # disable if self compare
        if self.segstats.selfcompare:
            return []

        gotable, gtseg = self._retrieve_overlap_tables()

        bodymatches = self._compute_bodymatch(gotable.overlap_map)
        sumstatsbm, bodystatsbm, tablestatsbm = self._compute_tablestats(bodymatches, gtseg.overlap_map, gotable.get_name(), self.thresholds)
        
        return sumstatsbm
        
    def write_body_stats(self):
        """Write stats per body.
        """
        # disable if non-gt mode
        if self.segstats.nogt:
            return []
        
        # disable if self compare
        if self.segstats.selfcompare:
            return []
        
        gotable, gtseg = self._retrieve_overlap_tables()
        typename = gotable.get_name()

        bodymatches = self._compute_bodymatch(gotable.overlap_map)   
        sumstatsbm, bodystatsbm, tablestatsbm = self._compute_tablestats(bodymatches, gtseg.overlap_map, typename, self.thresholds)
        
        return bodystatsbm
    
    def write_bodydebug(self):
        """Write connectivity table in the debug info.
        """
        # disable if non-gt mode
        if self.segstats.nogt:
            return []
        
        # disable if self compare
        if self.segstats.selfcompare:
            return []
        
        gotable, gtseg = self._retrieve_overlap_tables()
        typename = gotable.get_name()

        bodymatches = self._compute_bodymatch(gotable.overlap_map)   
        sumstatsbm, bodystatsbm, tablestatsbm = self._compute_tablestats(bodymatches, gtseg.overlap_map, typename, self.thresholds)
        
        return tablestatsbm 

    def _retrieve_overlap_tables(self):
        gotable_main = None
        gtseg_table = None

        # grab overlap tables
        for onum, gotable in enumerate(self.segstats.gt_overlaps):
            if gotable.get_comparison_type() == "synapse":
                # only one synapse-graph allowed
                assert gotable_main is None
                gotable_main = gotable

            if gotable.get_comparison_type() == "synapse-graph-gtseg":
                # only one synapse-graph allowed
                assert gtseg_table is None
                gtseg_table = gotable

        assert gotable_main is not None
        assert gtseg_table is not None

        return gotable_main, gtseg_table


    def _compute_bodymatch(self, table):
        """Compute the strongest body matches ignoring bodies < threshold.
       
        Note: threshold is applied to both segmentations.  If a body id is matched
        to a 0 body id, there was no match found.

        Returns:
            list of body mathces [(body1, body2, overlap, body1 size, body2 size)]
        """

        # always will be a point type
        threshold = self.segstats.ptfilter

        # keep track of important bodies 
        body1size = {}
        body2size = {}

        # adjacency list for overlap
        overlapflat = {}
        
        body2partners = {}

        # find important bodies
        for b1, overlapset in table.items():
            # filter smalll bodies (do not consider)
            if self.segstats.enable_sparse:
                if b1 not in self.segstats.important_bodies:
                    continue
            total = self._get_body_volume(overlapset)
            if total < threshold:
                continue 
            body1size[b1] = total
          
            bodysizelist = []
            for b2, overlap in overlapset:
                if b2 not in body2size:
                    body2size[b2] = overlap
                else:
                    body2size[b2] += overlap
                overlapflat[(b1,b2)] = overlap 
                bodysizelist.append((overlap, b2))

            # only grab the top 3 candidates for each body
            bodysizelist.sort()
            bodysizelist.reverse()
            for (overlap, b2) in bodysizelist[0:3]:
                if b2 not in body2partners:
                    body2partners[b2] = [b1]
                else:
                    body2partners[b2].append(b1)

        # create match overlap list: b1,b2,overlap,b1size,b2size
        match_overlap = []
      
        # find disjoint partitions
        body1partmap = {}
        repmap = {}
        partitions = {}
        for b2, b1list in body2partners.items():
            b1list.sort()
            repelement = b1list[0]
            if repelement in body1partmap:
                repelement = body1partmap[repelement]
            for b1 in b1list:
                if b1 in body1partmap:
                    rep2 = body1partmap[b1]
                    if repelement < rep2:
                        for body in repmap[rep2]:
                            body1partmap[body] = repelement
                        if repelement not in repmap:
                            repmap[repelement] = []
                        repmap[repelement].extend(repmap[rep2])
                        del repmap[rep2]
                    elif repelement > rep2:
                        if repelement in repmap:
                            for body in repmap[repelement]:
                                body1partmap[body] = rep2
                            repmap[rep2].extend(repmap[repelement])
                            del repmap[repelement]
                        repelement = rep2
                else:
                    body1partmap[b1] = repelement
                    if repelement not in repmap:
                        repmap[repelement] = []
                    repmap[repelement].append(b1)


        for b2, b1list in body2partners.items():
            repelement = 0
            for b1 in b1list:
                repelement = body1partmap[b1]
                if repelement not in partitions:
                    partitions[repelement] = (set(),set())
                partitions[repelement][0].add(b1)
            partitions[repelement][1].add(b2)

        # iterate through different disjoint partitions of the table
        # (should be pretty sparse)
        for rep, partition in partitions.items():
            # create table
            body1list = partition[0]
            body2list = partition[1]
            
            bodies1 = {}
            bodies2 = {}
            indextobodies1 = {}
            indextobodies2 = {}

            for b1 in body1list:
                indextobodies1[len(bodies1)] = b1
                bodies1[b1] = len(bodies1)
            for b2 in body2list:
                indextobodies2[len(bodies2)] = b2
                bodies2[b2] = len(bodies2)

            # create overlap table
            tablewidth = len(bodies2)
            tableheight = len(bodies1)

            # algorithm run on NxN table -- pad with 0
            maxdim = max(tablewidth, tableheight)
            table = numpy.zeros((maxdim, maxdim), dtype=numpy.uint32)

            # populate table
            for (b1,b2), overlap in overlapflat.items():
                # both bodies must be present for the overlap to be considered
                if b1 in bodies1 and b2 in bodies2:
                    table[bodies1[b1],bodies2[b2]] = overlap

            # create profit matrix and run hungarian match
            tableflip = table.max() - table
           
            # munkres hungarian math algorithm
            row_ind, col_ind = linear_sum_assignment(tableflip)  

            for b1 in row_ind:
                b2 = col_ind[b1]
                body1 = 0
                if b1 in indextobodies1:
                    body1 = indextobodies1[b1]
                body2 = 0
                if b2 in indextobodies2:
                    body2 = indextobodies2[b2]

                overlap = table[b1,b2]

                b1size = 0
                if body1 != 0:
                    b1size = body1size[body1]
                b2size = 0
                if body2 != 0:
                    b2size = body2size[body2]
                
                match_overlap.append((body1,body2,overlap,b1size,b2size))

        return match_overlap

    def _compute_tablestats(self, match_overlap, tableseg1seg2, typename, thresholds):
        """Generates summary and connectivity stats into a dict.

        The input table encodes all the connections between seg1 and seg2 intersection.

        Note: provides stats only in one direction.

        Args:
            match_overlap: list of seg1, seg2 matches
            tableseg1seg2: connectivity table showing pairs of seg1seg2 matches
            typename: type name for this stat
            thresholds: list of thresholds for accepting a connection

        Returns:
            dict for summary stats, dict for body stats, dict for connectivity table

            Format:
                summary stats: [ 
                    {"val": "percent matched", "name": "conn.match", "description": "<<amount matched> connection out of <amount total> matched}", "higher-better": true, "typename": <typename>}",
                    {"val": "percent matched", "name": "connpairs.matched-%d", "description": "%d body pairs matched out of %d with >=%d connections", "higher-better": true, "typename": <typename>}" ...
                ]
                body stats: a body stat type where bodies are [connections, [connections total, bodymatch]]

                connectivity table: list of [pre, prematch, post1, post1 match, overlap, total, post2, ...]
        """

        seg2toseg1 = {}
        seg1toseg2 = {}
        seg1stats = {}

        seg1conns = {}
        seg2conns_mapped = {}

        # hack to remove '-graph' from synapse graph
        #typename = "synapse:" + typename.split(":")[1]

        # ignore overlaps if no matching gt
        for (b1, b2, overlap, b1size, b2size) in match_overlap:
            if b1 == 0:
                continue
            seg2toseg1[b2] = b1
            seg1toseg2[b1] = b2
            seg1stats[b1] = (b2, overlap, b1size, b2size)

        # grab subset of tableseg1seg2 that has large bodies
        thresholded_falsepositives = [0]*len(thresholds)
        for pre, overlapset in tableseg1seg2.items():
            # ignore small bodies from segmentation 1
            
            # get encoded seg1 and seg2
            pre1 = int(pre >> 64)
            pre2 = int(pre & 0xffffffffffffffff)

            if pre1 not in seg1stats:
                continue
           
            # show all other bodies even if match is small or 0
            for (post, overlap) in overlapset:
                # get encoded seg1 and seg2
                post1 = int(post >> 64)
                post2 = int(post & 0xffffffffffffffff)
                
                if post1 not in seg1stats:
                    continue
                
                if pre1 not in seg1conns:
                    seg1conns[pre1] = {}
                if post1 not in seg1conns[pre1]:
                    seg1conns[pre1][post1] = 0
                seg1conns[pre1][post1] += overlap
        
                # find matching overlap
                if (pre2 in seg2toseg1 and seg2toseg1[pre2] == pre1) and (post2 in seg2toseg1 and seg2toseg1[post2] == post1):
                    # probably should only be called once by construction
                    assert (pre1, post1) not in seg2conns_mapped
                    seg2conns_mapped[(pre1,post1)] = overlap
                else: 
                    for count, threshold in enumerate(thresholds):
                        if overlap >= threshold:
                            thresholded_falsepositives[count] += 1


                   
        # generate stats
        conntable_stats = []
        bodystats = {}
        bodystatstemp = []

        overall_tot = 0
        overall_match = 0
        num_pairs = 0

        # find number of matches at different thresholds
        thresholded_match = [0]*len(thresholds)
        thresholded_match2 = [0]*len(thresholds)

        # iterate all connections and compile stats
        for pre, connections in seg1conns.items():
            # generate stats per body
            totconn = 0
            totmatch = 0
            pre2 = 0
            if pre in seg1toseg2:
                pre2 = seg1toseg2[pre]

            for post, overlap in connections.items():
                num_pairs += 1
                post2 = 0
                if post in seg1toseg2:
                    post2 = seg1toseg2[post]
                overlap2 = 0
                if (pre, post) in seg2conns_mapped:
                    overlap2 = seg2conns_mapped[(pre,post)]
                
                connrow = [pre, pre2, post, post2, overlap2, overlap]  
                conntable_stats.append(connrow)
                
                # accumulate body stats
                totconn += overlap
                totmatch += overlap2

                # find threshold matches
                for count, threshold in enumerate(thresholds):
                    if overlap >= threshold:
                        thresholded_match[count] += 1
                        if overlap2 >= threshold:
                            thresholded_match2[count] += 1
                    else: # check for false positives
                        if overlap2 >= threshold:
                            thresholded_falsepositives[count] += 1

            # accumulate global stats
            overall_tot += totconn
            overall_match += totmatch
            bodystatstemp.append((totmatch, [totconn, pre2], pre))

        sum_stats = [{"name": "conn.match", "description": "%d matched out of %d" % (overall_match, overall_tot), "higher-better": True, "typename": typename}]
        if overall_tot == 0:
            sum_stats[0]["val"] = 0
        else:
            sum_stats[0]["val"] = round(overall_match / float(overall_tot), 4)

        for pos, threshold in enumerate(thresholds):
            thresstat = {}
            thresstat["name"] = "connpairs.recall-%d" % threshold
            thresstat["higher-better"] = True
            thresstat["typename"] = typename
            if thresholded_match[pos] == 0:
                thresstat["val"] = 0 
            else:
                thresstat["val"] = round(thresholded_match2[pos] / float(thresholded_match[pos]), 4)
            thresstat["description"] = "%d body pairs matched out of %d with >=%d connections" % (thresholded_match2[pos], thresholded_match[pos], threshold)

            sum_stats.append(thresstat)

            thresstat = {}
            thresstat["name"] = "connpairs.precision-%d" % threshold
            thresstat["higher-better"] = True
            thresstat["typename"] = typename
            if thresholded_match[pos] == 0:
                thresstat["val"] = 0 
            else:
                thresstat["val"] = round(thresholded_match2[pos] / float(thresholded_match[pos]+thresholded_falsepositives[pos]), 4)
            thresstat["description"] = "precision: true positives %d and false positives %d with >=%d connections" % (thresholded_match2[pos], thresholded_falsepositives[pos], threshold)

            sum_stats.append(thresstat)
 


        # configure body stats
        bodytype = {"typename": typename, "name": "GTConn", "largest2smallest": True, "isgt": True}
        
        # sort and restrict number of bodies displayed
        bodystatstemp.sort()
        bodystatstemp.reverse()
        bodystatstemp = bodystatstemp[0:self.segstats.num_displaybodies]
        for (val, payload, pre) in bodystatstemp:
            bodystats[pre] = [val, payload]
        bodytype["bodies"] = bodystats

        debuginfo = {"typename": typename, "name": "connectivity-table"}
        debuginfo["info"] = conntable_stats
        return sum_stats, [bodytype], [debuginfo]
