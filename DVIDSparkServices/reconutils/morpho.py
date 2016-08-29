"""Morphological operations used by recon utilities.

This module contains helper functions for various morphological
operations used throughout reconutils.

"""

import libNeuroProofMetrics as np
from skimage.measure import label
from segstats import *
import numpy


def split_disconnected_bodies(label2):
    """Produces 3D volume split into connected components.

    This function identifies bodies that are the same label
    but are not connected.  It splits these bodies and
    produces a dict that maps these newly split bodies to
    the original body label.

    Args:
        label2 (numpy.array): 3D array of labels

    Returns:
        (
            partially relabeled array (numpy.array),
            new labels -> old labels (dict)
        )
    """

    # run connected components
    label2_split = label(label2)

    # make sure first label starts at 1
    label2_split = label2_split + 1

    # remove 0 label from consideration
    label2_split[label2 == 0] = 0

    # find maps and remap partition with rest
    stack2 = np.Stack(label2.astype(numpy.float64), 0)
    stack2_split = np.Stack(label2_split.astype(numpy.float64), 0)
    overlap_2_split = stack2.find_overlaps(stack2_split)
    stats2 = OverlapTable(overlap_2_split, ComparisonType())

    # needed to remap split labels into original coordinates
    remapping = {}
    label2_map = {}

    # 0=>0
    remapping[0] = 0
    remap_id = int(label2.max() + 1)
    
    for orig, newset in stats2.overlap_map.items():
        if len(newset) == 1:
            remapping[next(iter(newset))[0]] = orig
        else:
            for newbody, overlap in newset:
                remapping[newbody] = remap_id
                label2_map[remap_id] = orig
                remap_id += 1

    # relabel volume (minimize size of map that needs to be communicated)
    # TODO: !! refactor into morpho
    vectorized_relabel = numpy.frompyfunc(remapping.__getitem__, 1, 1)
    label2_split = vectorized_relabel(label2_split).astype(numpy.uint64)

    return label2_split, label2_map


def seeded_watershed(boundary, seed_threshold = 0, seed_size = 5, mask=None):
    """Extract seeds from boundary prediction and runs seeded watershed.

    Args:
        boundary (3D numpy array) = boundary predictions
        seed_threshold (int) = Add seeds where boundary prob is <= threshold
        seed_size (int) = seeds must be >= seed size
        mask (3D numpy array) = true to watershed, false to ignore
    Returns:
        3d watershed
    """
    
    from skimage import morphology as skmorph
    from numpy import bincount 

    # get seeds
    from scipy.ndimage import label as label2
    seeds = label2(boundary <= seed_threshold, output=numpy.uint32)[0]

    # remove small seeds
    if seed_size > 0:
        component_sizes = bincount(seeds.ravel())
        small_components = component_sizes < seed_size 
        small_locations = small_components[seeds]
        seeds[small_locations] = 0

    # mask out background (don't have to 0 out seeds since)
    supervoxels = skmorph.watershed(boundary, seeds,
            None, None, mask)

    return supervoxels



# stitch label chunks together (input: (s_id, (substack, array_c)))
def stitch(sc, label_chunks):
    def example(stuff):
        return stuff[1][0]
    subvolumes = label_chunks.map(example).collect()

    # return all subvolumes back to the driver
    # create offset map (substack id => offset) and broadcast
    offsets = {}
    offset = 0 

    for subvolume in subvolumes:
        offsets[subvolume.roi_id] = offset
        offset += subvolume.max_id
    subvolume_offsets = sc.broadcast(offsets)

    # (key, subvolume, label chunk)=> (new key, (subvolume, boundary))
    def extract_boundaries(key_labels):
        # compute overlap -- assume first point is less than second
        def intersects(pt1, pt2, pt1_2, pt2_2):
            assert pt1 <= pt2, "point 1 greater than point 2: {} > {}".format( pt1, pt2 )
            assert pt1_2 <= pt2_2, "point 1_2 greater than point 2_2: {} > {}".format( pt1_2, pt2_2 )

            val1 = max(pt1, pt1_2)
            val2 = min(pt2, pt2_2)
            size = val2-val1
            npt1 = val1 - pt1 
            npt1_2 = val1 - pt1_2

            return npt1, npt1+size, npt1_2, npt1_2+size

        import numpy

        oldkey, (subvolume, labels) = key_labels

        boundary_array = []
        
        # iterate through all ROI partners
        for partner in subvolume.local_regions:
            key1 = subvolume.roi_id
            key2 = partner[0]
            roi2 = partner[1]
            if key2 < key1:
                key1, key2 = key2, key1
            
            # create key for boundary pair
            newkey = (key1, key2)

            # crop volume to overlap
            offx1, offx2, offx1_2, offx2_2 = intersects(
                            subvolume.roi.x1-subvolume.border,
                            subvolume.roi.x2+subvolume.border,
                            roi2.x1-subvolume.border,
                            roi2.x2+subvolume.border
                        )
            offy1, offy2, offy1_2, offy2_2 = intersects(
                            subvolume.roi.y1-subvolume.border,
                            subvolume.roi.y2+subvolume.border,
                            roi2.y1-subvolume.border,
                            roi2.y2+subvolume.border
                        )
            offz1, offz2, offz1_2, offz2_2 = intersects(
                            subvolume.roi.z1-subvolume.border,
                            subvolume.roi.z2+subvolume.border,
                            roi2.z1-subvolume.border,
                            roi2.z2+subvolume.border
                        )
                        
            labels_cropped = numpy.copy(labels[offz1:offz2, offy1:offy2, offx1:offx2])

            # add to flat map
            boundary_array.append((newkey, (subvolume, labels_cropped)))

        return boundary_array


    # return compressed boundaries (id1-id2, boundary)
    mapped_boundaries = label_chunks.flatMap(extract_boundaries) 

    # shuffle the hopefully smallish boundaries into their proper spot
    # groupby is not a big deal here since same keys will not be in the same partition
    grouped_boundaries = mapped_boundaries.groupByKey()

    # mappings to one partition (larger/second id keeps orig labels)
    # (new key, list<2>(subvolume, boundary compressed)) =>
    # (key, (subvolume, mappings))
    def stitcher(key_boundary):
        import numpy
        key, (boundary_list) = key_boundary

        # should be only two values
        if len(boundary_list) != 2:
            raise Exception("Expects exactly two subvolumes per boundary")
        # extract iterables
        boundary_list_list = []
        for item1 in boundary_list:
            boundary_list_list.append(item1)

        # order subvolume regions (they should be the same shape)
        subvolume1, boundary1 = boundary_list_list[0] 
        subvolume2, boundary2 = boundary_list_list[1] 

        if subvolume1.roi_id > subvolume2.roi_id:
            subvolume1, subvolume2 = subvolume2, subvolume1
            boundary1, boundary2 = boundary2, boundary1

        if boundary1.shape != boundary2.shape:
            raise Exception("Extracted boundaries are different shapes")
        
        # determine list of bodies in play
        z2, y2, x2 = boundary1.shape
        z1 = y1 = x1 = 0 

        # determine which interface there is touching between subvolumes 
        if subvolume1.touches(subvolume1.roi.x1, subvolume1.roi.x2,
                            subvolume2.roi.x1, subvolume2.roi.x2):
            x1 = x2/2 
            x2 = x1 + 1
        if subvolume1.touches(subvolume1.roi.y1, subvolume1.roi.y2,
                            subvolume2.roi.y1, subvolume2.roi.y2):
            y1 = y2/2 
            y2 = y1 + 1
        
        if subvolume1.touches(subvolume1.roi.z1, subvolume1.roi.z2,
                            subvolume2.roi.z1, subvolume2.roi.z2):
            z1 = z2/2 
            z2 = z1 + 1

        eligible_bodies = set(numpy.unique(boundary2[z1:z2, y1:y2, x1:x2]))
        body2body = {}

        label2_bodies = numpy.unique(boundary2)

        for body in label2_bodies:
            if body == 0:
                continue
            body2body[body] = {}

        # traverse volume to find maximum overlap
        for (z,y,x), body1 in numpy.ndenumerate(boundary1):
            body2 = boundary2[z,y,x]
            if body2 == 0 or body1 == 0:
                continue
            
            if body1 not in body2body[body2]:
                body2body[body2][body1] = 0
            body2body[body2][body1] += 1


        # create merge list 
        merge_list = []

        # merge if any overlap
        for body2, bodydict in body2body.items():
            if body2 in eligible_bodies:
                for body1, val in bodydict.items():
                    if val > 0:
                        merge_list.append([int(body1), int(body2)])
                   

        # handle offsets in mergelist
        offset1 = subvolume_offsets.value[subvolume1.roi_id] 
        offset2 = subvolume_offsets.value[subvolume2.roi_id] 
        for merger in merge_list:
            merger[0] = merger[0]+offset1
            merger[1] = merger[1]+offset2

        # return id and mappings, only relevant for stack one
        return (subvolume1.roi_id, merge_list)

    # key, mapping1; key mapping2 => key, mapping1+mapping2
    def reduce_mappings(b1, b2):
        b1.extend(b2)
        return b1

    # map from grouped boundary to substack id, mappings
    subvolume_mappings = grouped_boundaries.map(stitcher).reduceByKey(reduce_mappings)

    # reconcile all the mappings by sending them to the driver
    # (not a lot of data and compression will help but not sure if there is a better way)
    merge_list = []
    all_mappings = subvolume_mappings.collect()
    for (substack_id, mapping) in all_mappings:
        merge_list.extend(mapping)

    # make a body2body map
    body1body2 = {}
    body2body1 = {}

    for merger in merge_list:
        # body1 -> body2
        body1 = merger[0]
        if merger[0] in body1body2:
            body1 = body1body2[merger[0]]
        body2 = merger[1]
        if merger[1] in body1body2:
            body2 = body1body2[merger[1]]

        if body2 not in body2body1:
            body2body1[body2] = set()
        
        # add body1 to body2 map
        body2body1[body2].add(body1)
        # add body1 -> body2 mapping
        body1body2[body1] = body2

        if body1 in body2body1:
            for tbody in body2body1[body1]:
                body2body1[body2].add(tbody)
                body1body2[tbody] = body2

    body2body = zip(body1body2.keys(), body1body2.values())
   
    # potentially costly broadcast
    # (possible to split into substack to make more efficient but compression should help)
    master_merge_list = sc.broadcast(body2body)

    # use offset and mappings to relabel volume
    def relabel(key_label_mapping):
        import numpy

        (subvolume, labels) = key_label_mapping

        # grab broadcast offset
        offset = subvolume_offsets.value[subvolume.roi_id]

        # check for body mask labels and protect from renumber
        fix_bodies = []
        
        labels = labels + offset 
        
        # make sure 0 is 0
        labels[labels == offset] = 0

        # create default map 
        mapping_col = numpy.unique(labels)
        label_mappings = dict(zip(mapping_col, mapping_col))
       
        # create maps from merge list
        for mapping in master_merge_list.value:
            if mapping[0] in label_mappings:
                label_mappings[mapping[0]] = mapping[1]

        # apply maps
        vectorized_relabel = numpy.frompyfunc(label_mappings.__getitem__, 1, 1)
        labels = vectorized_relabel(labels).astype(numpy.uint64)
   
        return (subvolume, labels)

    # just map values with broadcast map
    # Potential TODO: consider fast join with partitioned map (not broadcast)
    return label_chunks.mapValues(relabel)


