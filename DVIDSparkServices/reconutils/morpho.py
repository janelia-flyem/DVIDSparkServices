"""Morphological operations used by recon utilities.

This module contains helper functions for various morphological
operations used throughout reconutils.

"""

import numpy
import numpy as np
import pandas as pd
import vigra

from dvidutils import LabelMapper

import skimage.measure as skm

from DVIDSparkServices.util import select_item, bb_to_slicing, bb_as_tuple
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
from DVIDSparkServices.reconutils.downsample import downsample_binary_3d, downsample_binary_3d_suppress_zero, downsample_box


def split_disconnected_bodies(labels_orig):
    """
    Produces 3D volume split into connected components.

    This function identifies bodies that are the same label
    but are not connected.  It splits these bodies and
    produces a dict that maps these newly split bodies to
    the original body label.

    Special exception: Segments with label 0 are not relabeled.

    Args:
        labels_orig (numpy.array): 3D array of labels

    Returns:
        (labels_new, new_to_orig)

        labels_new:
            The partially relabeled array.
            Segments that were not split will keep their original IDs.
            Among split segments, the largest 'child' of a split segment retains the original ID.
            The smaller segments are assigned new labels in the range (N+1)..(N+1+S) where N is
            highest original label and S is the number of new segments after splitting.
        
        new_to_orig:
            A pseudo-minimal (but not quite minimal) mapping of labels
            (N+1)..(N+1+S) -> some subset of (1..N),
            which maps new segment IDs to the segments they came from.
            Segments that were not split at all are not mentioned in this mapping,
            for split segments, every mapping pair for the split is returned, including the k->k (identity) pair.
    """
    # Compute connected components and cast back to original dtype
    labels_cc = skm.label(labels_orig, background=0, connectivity=1)
    assert labels_cc.dtype == np.int64
    if labels_orig.dtype == np.uint64:
        labels_cc = labels_cc.view(np.uint64)
    else:
        labels_cc = labels_cc.astype(labels_orig.dtype, copy=False)

    # Find overlapping segments between orig and CC volumes
    overlap_table_df = contingency_table(labels_orig, labels_cc)
    overlap_table_df.columns = ['orig', 'cc', 'voxels']
    overlap_table_df.sort_values('voxels', ascending=False, inplace=True)
    
    # If a label in 'orig' is duplicated, it has multiple components in labels_cc.
    # The largest component gets to keep the original ID;
    # the other components must take on new values.
    # (The new values must not conflict with any of the IDs in the original, so start at orig_max+1)
    new_cc_pos = overlap_table_df['orig'].duplicated()
    orig_max = overlap_table_df['orig'].max()
    new_cc_values = np.arange(orig_max+1, orig_max+1+new_cc_pos.sum(), dtype=labels_orig.dtype)

    overlap_table_df['final_cc'] = overlap_table_df['orig'].copy()
    overlap_table_df.loc[new_cc_pos, 'final_cc'] = new_cc_values
    
    # Relabel the CC volume to use the 'final_cc' labels
    mapper = LabelMapper(overlap_table_df['cc'].values, overlap_table_df['final_cc'].values)
    mapper.apply_inplace(labels_cc)

    # Generate the mapping that could (if desired) convert the new volume into the original one,
    # as described in the docstring above.
    emitted_mapping_rows = overlap_table_df['orig'].duplicated(keep=False)
    new_to_orig = dict(overlap_table_df.loc[emitted_mapping_rows, ['final_cc', 'orig']].values)
    
    return labels_cc, new_to_orig


def contingency_table(left_vol, right_vol):
    """
    Compute the overlap table ("contingency table")
    for the given label volumes.
    
    Returns:
        DataFrame with columns ['left', 'right', 'overlap_size'],
        indicating the unique pairs of overlapping IDs and the number
        of overlapping voxels.
    """
    assert left_vol.dtype == right_vol.dtype
    df = pd.DataFrame( {'left': left_vol.reshape(-1),
                       'right': right_vol.reshape(-1)} )

    table = df.groupby(['left', 'right']).agg('size')
    table = table.reset_index()
    table.columns = ['left', 'right', 'overlap_size']
    return table


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


def object_masks_for_labels( segmentation, box=None, minimum_object_size=1, always_keep_border_objects=True, compress_masks=False ):
    """
    Given a segmentation containing N unique label values (excluding label 0),
    return N binary masks and their bounding boxes.

    Note: Result is *not* sorted by label ID.
    
    segmentation:
        label image, any dtype
    
    box:
        Bounding box of the segmentation in global coordiantes.
        If the segmentation was extracted from a larger (global) coordinate space,
        this parameter can be used to ensure that the returned mask bounding boxes use global coordinates.
    
    minimum_object_size:
        Extracted objects with fewer pixels than the minimum size are discarded.
    
    always_keep_border_objects:
        Ignore the `minimum_object_size` constraint for objects that touch edge of the segmentation volume.
        (Useful if you plan to merge the object masks with neighboring segmentation blocks.)

    compress_masks:
        Return masks as a CompressedNumpyArray instead of an ordinary np.ndarray
    
    Returns:
        List of tuples: [(label_id, (mask_bounding_box, mask, count)),
                         (label_id, (mask_bounding_box, mask, count)), ...]
        
        ...where: `mask_bounding_box` is of the form ((z0, y0, x0), (z1, y1, x1)),
                  `mask` is either a np.ndarray or CompressedNumpyArray, depending on the compress_masks argument, and
                   `count` is the count of nonzero pixels in the mask        

        Note: Result is *not* sorted by label ID.
    """
    if box is None:
        box = [ (0,)*segmentation.ndim, segmentation.shape ]
    sv_start, sv_stop = box

    segmentation = vigra.taggedView(segmentation, 'zyx')
    consecutive_seg = np.empty_like(segmentation, dtype=np.uint32)
    _, maxlabel, bodies_to_consecutive = vigra.analysis.relabelConsecutive(segmentation, out=consecutive_seg) # preserves zeros by default
    consecutive_to_bodies = { v:k for k,v in bodies_to_consecutive.items() }
    del segmentation
    
    # We don't care what the 'image' parameter is, but we have to give something
    image = consecutive_seg.view(np.float32)
    acc = vigra.analysis.extractRegionFeatures(image, consecutive_seg, features=['Coord<Minimum >', 'Coord<Maximum >', 'Count'])

    body_ids_and_masks = []
    for label in range(1, maxlabel+1): # Skip 0
        count = acc['Count'][label]
        min_coord = acc['Coord<Minimum >'][label].astype(int)
        max_coord = acc['Coord<Maximum >'][label].astype(int)
        box_local = np.array((min_coord, 1+max_coord))
        
        mask = (consecutive_seg[bb_to_slicing(*box_local)] == label)
        if compress_masks:
            assert mask.dtype == np.bool # CompressedNumpyArray has special support for boolean masks.
            mask = CompressedNumpyArray(mask)

        body_id = consecutive_to_bodies[label]
        box_global = box_local + sv_start

        # Only keep segments that are big enough OR touch the subvolume border.
        if count >= minimum_object_size \
        or (always_keep_border_objects and (   (box_global[0] == sv_start).any()
                                            or (box_global[1] == sv_stop).any())):
            body_ids_and_masks.append( (body_id, (bb_as_tuple(box_global), mask, count)) )
    
    return body_ids_and_masks

def assemble_masks( boxes, masks, downsample_factor=0, minimum_object_size=1, max_combined_mask_size=1e9, suppress_zero=True, pad=0 ):
    """
    Given a list of bounding boxes and corresponding binary mask arrays,
    assemble the superset of those masks in a larger array.
    To save RAM, the entire result can be optionally downsampled.
    
    boxes:
        List of bounding box tuples [(z0,y0,x0), (z1,y1,x1), ...]
    
    masks:
        Iterable of binary mask arrays.
    
    downsample_factor:
        How much to downsample the result:
            1 - no downsampling (if possible, considering max_combined_mask_size)
            2+ - Downsample the result by at least 2x,3x, etc.

    minimum_object_size:
        If the final result is smaller than this number (as measured in NON-downsampled pixels),
        return 'None' instead of an actual mask.
    
    max_combined_mask_size:
        The maximum allowed size for the combined downsampled array.
        If the given downsample_factor would result in an array that exceeds max_combined_mask_size,
        then a new downsample_factor is automatically chosen.
    
    suppress_zero:
        ("Maximal value downsampling") In the downsampled mask result, output voxels
        will be 1 if *any* of their input voxels were 1 (even of they were outnumbered by 0s).
    
    pad:
        If non-zero, leave some padding (a halo of blank voxels) on all sides of the final volume.
        (This is useful for mesh-generation algorithms, which require a boundary between on/off pixels.)
        
        Note: The padding is applied AFTER downsampling, so the returned combined_bounding_box and combined_mask
              will be expanded by pad*downsample_factor before and after each axis.
    
    Returns: (combined_bounding_box, combined_mask, downsample_factor)

        where:
            combined_bounding_box:
                the bounding box of the returned mask,
                in NON-downsampled coordinates: ((z0,y0,x0), (z1,y1,x1)).
                Note: If you specified a 'pad', then this will be
                      reflected in the combined_bounding_box.
            
            combined_mask:
                the full downsampled combined mask, including any padding.
            
            downsample_factor:
                The chosen downsampling factor if using 'auto' downsampling,
                otherwise equal to the downsample_factor you passed in.
    """
    boxes = np.asarray(boxes)
    
    combined_box = np.zeros((2,3), dtype=np.int64)
    combined_box[0] = boxes[:, 0, :].min(axis=0)
    combined_box[1] = boxes[:, 1, :].max(axis=0)
    
    # Auto-choose a downsample factor that will result in a
    # combined downsampled array no larger than max_combined_mask_size
    full_size = np.prod(combined_box[1] - combined_box[0])
    auto_downsample_factor = 1 + int(np.power(full_size / max_combined_mask_size, (1./3)))
    chosen_downsample_factor = max(downsample_factor, auto_downsample_factor)

    # Leave room for padding.
    combined_box[:] += chosen_downsample_factor * np.array((-pad, pad))[:,None]

    block_shape = np.array((chosen_downsample_factor,)*3)
    combined_downsampled_box = downsample_box( combined_box, block_shape )
    combined_downsampled_box_shape = combined_downsampled_box[1] - combined_downsampled_box[0]

    combined_mask_downsampled = np.zeros( combined_downsampled_box_shape, dtype=np.bool )

    if suppress_zero:
        downsample_func = downsample_binary_3d_suppress_zero
    else:
        downsample_func = downsample_binary_3d

    for box_global, mask in zip(boxes, masks):
        box_global = np.asarray(box_global)
        mask_downsampled, downsampled_box = downsample_func(mask, chosen_downsample_factor, box_global)
        downsampled_box[:] -= combined_downsampled_box[0]
        combined_mask_downsampled[ bb_to_slicing(*downsampled_box) ] |= mask_downsampled

    if combined_mask_downsampled.sum() * chosen_downsample_factor**3 < minimum_object_size:
        # 'None' results will be filtered out. See below.
        combined_mask_downsampled = None

    return ( combined_box, combined_mask_downsampled, chosen_downsample_factor )


def stitch(sc, label_chunks):
    """
    label_chunks (RDD): [ (subvol, (seg_vol, max_id)),
                          (subvol, (seg_vol, max_id)),
                          ... ]

    Note: This function requires that label_chunks is already persist()ed in memory.
    """    
    assert label_chunks.is_cached, "You must persist() label_chunks before calling this function."
    subvolumes_rdd = select_item(label_chunks, 0)
    subvolumes = subvolumes_rdd.collect()
    max_ids = select_item(label_chunks, 1, 1).collect()

    # return all subvolumes back to the driver
    # create offset map (substack id => offset) and broadcast
    offsets = {}
    offset = numpy.uint64(0)

    for subvolume, max_id in zip(subvolumes, max_ids):
        offsets[subvolume.sv_index] = offset
        offset += max_id
    subvolume_offsets = sc.broadcast(offsets)

    # (subvol, label_vol) => [ (sv_index_1, sv_index_2), (subvol, boundary_labels)), 
    #                          (sv_index_1, sv_index_2), (subvol, boundary_labels)), ...] 
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

        subvolume, labels = key_labels

        boundary_array = []
        
        # iterate through all ROI partners
        for partner in subvolume.local_regions:
            key1 = subvolume.sv_index
            key2 = partner[0]
            box2 = partner[1]
            if key2 < key1:
                key1, key2 = key2, key1
            
            # crop volume to overlap
            offx1, offx2, offx1_2, offx2_2 = intersects(
                            subvolume.box.x1-subvolume.border,
                            subvolume.box.x2+subvolume.border,
                            box2.x1-subvolume.border,
                            box2.x2+subvolume.border
                        )
            offy1, offy2, offy1_2, offy2_2 = intersects(
                            subvolume.box.y1-subvolume.border,
                            subvolume.box.y2+subvolume.border,
                            box2.y1-subvolume.border,
                            box2.y2+subvolume.border
                        )
            offz1, offz2, offz1_2, offz2_2 = intersects(
                            subvolume.box.z1-subvolume.border,
                            subvolume.box.z2+subvolume.border,
                            box2.z1-subvolume.border,
                            box2.z2+subvolume.border
                        )
                        
            labels_cropped = numpy.copy(labels[offz1:offz2, offy1:offy2, offx1:offx2])

            # create key for boundary pair
            newkey = (key1, key2)

            # add to flat map
            boundary_array.append((newkey, (subvolume, labels_cropped)))

        return boundary_array


    # return compressed boundaries (id1-id2, boundary)
    # (subvol, labels) -> [ ( (k1, k2), (subvol, boundary_labels_1) ),
    #                       ( (k1, k2), (subvol, boundary_labels_1) ),
    #                       ( (k1, k2), (subvol, boundary_labels_1) ), ... ]
    label_vols_rdd = select_item(label_chunks, 1, 0)
    mapped_boundaries = subvolumes_rdd.zip(label_vols_rdd).flatMap(extract_boundaries) 

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

        if subvolume1.sv_index > subvolume2.sv_index:
            subvolume1, subvolume2 = subvolume2, subvolume1
            boundary1, boundary2 = boundary2, boundary1

        if boundary1.shape != boundary2.shape:
            raise Exception("Extracted boundaries are different shapes")
        
        # determine list of bodies in play
        z2, y2, x2 = boundary1.shape
        z1 = y1 = x1 = 0 

        # determine which interface there is touching between subvolumes 
        if subvolume1.touches(subvolume1.box.x1, subvolume1.box.x2,
                              subvolume2.box.x1, subvolume2.box.x2):
            x1 = x2 // 2
            x2 = x1 + 1

        if subvolume1.touches(subvolume1.box.y1, subvolume1.box.y2,
                              subvolume2.box.y1, subvolume2.box.y2):
            y1 = y2 // 2
            y2 = y1 + 1
        
        if subvolume1.touches(subvolume1.box.z1, subvolume1.box.z2,
                              subvolume2.box.z1, subvolume2.box.z2):
            z1 = z2 // 2
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
        offset1 = subvolume_offsets.value[subvolume1.sv_index] 
        offset2 = subvolume_offsets.value[subvolume2.sv_index] 
        for merger in merge_list:
            merger[0] = merger[0]+offset1
            merger[1] = merger[1]+offset2

        # return id and mappings, only relevant for stack one
        return (subvolume1.sv_index, merge_list)

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

    body2body = list(zip(body1body2.keys(), body1body2.values()))
   
    # potentially costly broadcast
    # (possible to split into substack to make more efficient but compression should help)
    master_merge_list = sc.broadcast(body2body)

    # use offset and mappings to relabel volume
    def relabel(key_label_mapping):
        import numpy

        (subvolume, labels) = key_label_mapping

        # grab broadcast offset
        offset = numpy.uint64( subvolume_offsets.value[subvolume.sv_index] )

        # check for body mask labels and protect from renumber
        fix_bodies = []
        
        labels = labels + offset 
        
        # make sure 0 is 0
        labels[labels == offset] = 0

        # create default map 
        labels_view = vigra.taggedView(labels.astype(numpy.uint64), 'zyx')
        mapping_col = numpy.sort( vigra.analysis.unique(labels_view) )
        label_mappings = dict(zip(mapping_col, mapping_col))
       
        # create maps from merge list
        for mapping in master_merge_list.value:
            if mapping[0] in label_mappings:
                label_mappings[mapping[0]] = mapping[1]

        # apply maps
        new_labels = numpy.empty_like( labels, dtype=numpy.uint64 )
        new_labels_view = vigra.taggedView(new_labels, 'zyx')
        vigra.analysis.applyMapping(labels_view, label_mappings, allow_incomplete_mapping=True, out=new_labels_view)
        return (subvolume, new_labels)

    # just map values with broadcast map
    # Potential TODO: consider fast join with partitioned map (not broadcast)
    return subvolumes_rdd.zip(label_vols_rdd).map(relabel)


