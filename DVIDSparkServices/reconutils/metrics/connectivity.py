from munkres import Munkres
import numpy

# TODO: make generic class

def compute_bodymatch(overlapmap, threshold=0):
    """Compute the strongest body matches ignoring bodies < threshold.
   
    Note: threshold is applied to both segmentations.  If a body id is matched
    to a 0 body id, there was no match found.

    Returns:
        list of body mathces [(body1, body2, overlap, body1 size, body2 size)]
    """

    # keep track of important bodies 
    bodies1 = {}
    bodies2 = {}
    indextobodies1 = {}
    indextobodies2 = {}
    body1size = {}
    body2size = {}

    # adjacency list for overlap
    overlapflat = {}

    # find important bodies
    for b1, overlapset in overlapmap.items():
        total = 0
        for b2, overlap in overlapset:
            total += overlap
            overlapflat[(b1,b2)] = overlap 
            if b2 not in bodies2:
                body2size[b2] = overlap
            else:
                body2size[b2] += overlap
            if body2size[b2] >= threshold and b2 not in bodies2:
                indextobodies2[len(bodies2)] = b2
                bodies2[b2] = len(bodies2)
                
        if total >= threshold:
            indextobodies1[len(bodies1)] = b1
            bodies1[b1] = len(bodies1)
            body1size[b1] = total

    # create overlap table
    tablewidth = len(bodies2)
    tableheight = len(bodies1)

    # algorithm run on NxN table -- pad with 0
    maxdim = max(tablewidth, tableheight)
    table = numpy.zeros((maxdim, maxdim))

    # populate table
    for (b1,b2), overlap in overlapflat.items():
        # both bodies must be present for the overlap to be considered
        if b1 in bodies1 and b2 in bodies2:
            table[bodies1[b1],bodies2[b2]] = overlap

    # create profit matrix and run hungarian match
    table = table.max() - table
    m = Munkres()
    res = m.compute(table)
    
    # flip table back to reprsent overlap
    table = table.max() - table

    # create match overlap list: b1,b2,overlap,b1size,b2size
    match_overlap = []
    for (b1,b2) in res:
        # ?!
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


def compute_tablestats(match_overlap, tableseg1seg2, thresholds):
    """Generates summary and connectivity stats into a dict.

    The input table encodes all the connections between seg1 and seg2 intersection.

    Note: provides stats only in one direction.

    Args:
        match_overlap: list of seg1, seg2 matches
        tableseg1seg2: connectivity table showing pairs of seg1seg2 matches 
        thresholds: list of thresholds for accepting a connection

    Returns:
        dict for summary stats, dict for body stats, dict for connectivity table

        Format:
            summary stats: [ amount matched, amount total, num_pairs, [thresholds], [threshold match], threshold totals]]

            body stats: list of [body, bodymatch, connections, connections total]

            connectivity table: list of [pre, prematch, post1, post1 match, overlap, total, post2, ...]
    """

    seg2toseg1 = {}
    seg1toseg2 = {}
    seg1stats = {}

    seg1conns = {}
    seg2conns_mapped = {}

    # ignore overlaps if no matching gt
    for (b1, b2, overlap, b1size, b2size) in match_overlap:
        if b1 == 0:
            continue
        seg2toseg1[b2] = b1
        seg1toseg2[b1] = b2
        seg1stats[b1] = (b2, overlap, b1size, b2size)

    # grab subset of tableseg1seg2 that has large bodies
    for pre, overlapset in tableseg1seg2.items():
        # ignore small bodies from segmentation 1
        
        # get encoded seg1 and seg2
        pre1 = pre >> 64
        pre2 = pre & 0xffffffffffffffff

        if pre1 not in seg1stats:
            continue
       
        # show all other bodies even if match is small or 0
        for (post, overlap) in overlapset:
            # get encoded seg1 and seg2
            post1 = post >> 64
            post2 = post & 0xffffffffffffffff
            
            if post1 not in seg1stats:
                continue
            
            if pre1 not in seg1conns:
                seg1conns[pre1] = {}
            if post1 not in seg1conns[pre1]:
                seg1conns[pre1][post1] = 0
            seg1conns[pre1][post1] += overlap
    
            # find matching overlap
            if seg2toseg1[pre2] == pre1 and seg2toseg1[post2] == post1:
                # probably should only be called once by construction
                assert (pre1, post1) not in seg2conns_mapped
                seg2conns_mapped[(pre1,post1)] = overlap
               
    # generate stats
    conntable_stats = []
    bodystats = []

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

        # accumulate global stats
        overall_tot += totconn
        overall_match += totmatch
        bodystats.append([pre, pre2, totmatch, totconn])

    sum_stats = [overall_match, overall_tot, num_pairs, thresholds, thresholded_match2, thresholded_match]

    return sum_stats, bodystats, conntable_stats
