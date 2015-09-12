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
    seeds = label2(boundary <= seed_threshold)[0]

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

