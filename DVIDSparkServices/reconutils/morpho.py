"""Morphological operations used by recon utilities.

This module contains helper functions for various morphological
operations used throughout reconutils.

"""

import libNeuroProofMetrics as np
from skimage.measure import label
from segstats import *

def connected_components(label2):
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

    # find maps and remap partition with rest
    stack2 = np.Stack(label2, 0)
    stack2_split = np.Stack(label2_split, 0)
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
    label2_split = vectorized_relabel(label2_split).astype(numpy.uint64)

    return label2_split, label2_map
