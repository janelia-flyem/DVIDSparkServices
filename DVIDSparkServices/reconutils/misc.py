import numpy
import scipy.ndimage
import DVIDSparkServices.reconutils.morpho

def find_large_empty_regions(grayscale_vol, min_background_voxel_count=100):
    """
    Returns mask that excludes large background (0-valued) regions, if any exist.
    """    
    if not (grayscale_vol == 0).any():
        # No background pixels.
        # We could return all ones, but we are also allowed
        # by convention to return 'None', which is faster.
        return None

    # Produce a mask that excludes 'background' pixels
    # (typically zeros around the volume edges)
    background_mask = numpy.zeros(grayscale_vol.shape, dtype=numpy.uint8)
    background_mask[grayscale_vol == 0] = 1

    # Compute connected components (cc) and toss out the small components
    cc = scipy.ndimage.label(background_mask)[0]
    cc_sizes = numpy.bincount(cc.ravel())
    small_cc_selections = cc_sizes < min_background_voxel_count
    small_cc_locations = small_cc_selections[cc]
    background_mask[small_cc_locations] = 0

    if not background_mask.any():
        # No background pixels.
        # We could return all ones, but we are also allowed
        # by convention to return 'None', which is faster.
        return None
    
    # Now background_mask == 1 for background and 0 elsewhere, so invert.
    numpy.logical_not(background_mask, out=background_mask)
    return background_mask.view(numpy.bool_)

def naive_membrane_predictions(grayscale_vol, mask_vol=None ):
    """
    Stand-in for membrane prediction, for testing purposes.
    Simply returns the inverted grayscale as our 'predictions'
    """
    low = grayscale_vol.min()
    high = grayscale_vol.max()

    # Predictions should be in range 0.0-1.0
    grayscale_vol = grayscale_vol.astype(numpy.float32)
    grayscale_vol[:] = (grayscale_vol-low)/(high-low)

    # Low intensity means high probability of membrane.
    inverted = (1.0-grayscale_vol)
    return inverted[..., None] # Segmentor wants 4D predictions, so append channel axis

def seeded_watershed(boundary_volume, mask, seed_threshold=0.2, seed_size=5):
    """
    Perform a seeded watershed on the given volume.
    Seeds are generated using a seed-threshold and minimum seed-size.
    """
    ws = DVIDSparkServices.reconutils.morpho.seeded_watershed
    supervoxels = ws( boundary_volume.sum(axis=-1), seed_threshold, seed_size, mask )
    return supervoxels

def noop_aggolmeration(bounary_volume, supervoxels):
    """
    Stand-in for an agglomeration function.
    This function returns the supervoxels unchanged.
    """
    return supervoxels

def compute_vi(seg1, seg2):
    """ Compute VI between seg1 and seg2

    Note:
        Consider seg1 as the "ground truth"

    Args:
        seg1 (numpy): 3D array of seg1 labels
        seg2 (numpy): 3D array of seg2 labels

    Returns:
        false merge vi, false split v

    """
    
    # TODO !! avoid conversion
    seg1 = seg1.astype(numpy.float64)
    seg2 = seg2.astype(numpy.float64)

    import libNeuroProofMetrics as np
    from segstats import *

    # creates stack and adds boundary padding
    seg1 = np.Stack(seg1, 0)
    seg2 = np.Stack(seg2, 0)

    # returns list of (body1, body2, overlap)
    overlaps12 = seg1.find_overlaps(seg2)

    # reverse overlaps for mappings
    overlaps21 = []
    for (body1, body2, overlap) in overlaps12:
        overlaps21.append((body2, body1, overlap))

    # make mappings for overlaps
    seg1_overlap = OverlapTable(overlaps12, None)
    seg2_overlap = OverlapTable(overlaps21, None)


    fmerge_vi, fsplit_vi, fmerge_bodies, fsplit_bodies, dummy = \
                                calculate_vi(seg1_overlap, seg2_overlap)
    return fmerge_vi, fsplit_vi


def select_channels( predictions, selected_channels ):
    """
    selected_channels: A list of channel indexes to select and return from the prediction results.
                       'None' can also be given, which means "return all prediction channels".
                       You may also return a *nested* list, in which case groups of channels can be
                       combined (summed) into their respective output channels.
                       For example: selected_channels=[0,3,[2,4],7] means the output will have 4 channels:
                                    0,3,2+4,7 (channels 5 and 6 are simply dropped).
    """
    import numpy as np

    if selected_channels is None:
        return predictions
    
    # If a list of ints, then this is fast -- just select the channels we want.
    assert isinstance(selected_channels, list)
    if all(np.issubdtype(type(x), np.integer) for x in selected_channels):
        return predictions[..., selected_channels]
    
    # The user gave us a nested list.
    # We have to compute it channel-by-channel (the slow way).
    output_shape = predictions.shape[:-1] + (len(selected_channels),)
    combined_predictions = np.ndarray(shape=output_shape, dtype=predictions.dtype )
    for output_channel, selection in enumerate(selected_channels):
        if np.issubdtype(type(selection), np.integer):
            combined_predictions[..., output_channel] = predictions[..., selection]
        else:
            # Selected channel is a list of channels to combine
            assert isinstance(selection, list)
            assert all(np.issubdtype(type(x), np.integer) for x in selection)
            combined_predictions[..., output_channel] = predictions[..., selection].sum(axis=-1)
    return combined_predictions

def normalize_channels_in_place(predictions):
    """
    Renormalize all pixels so the channels sum to 1 everywhere.
    That is, (predictions.sum(axis=-1) == 1.0).all()

    Note: Pixels with 0.0 in all channels will be simply given a value of 1/N in all channels.
    """
    import numpy as np

    channel_totals = predictions.sum(axis=-1)
    
    # Avoid divide-by-zero: Replace all-zero pixels with 1/N for all channels
    num_channels = predictions.shape[-1]
    predictions[:] = np.where( channel_totals[...,None] == 0.0, 1.0/num_channels, predictions )
    channel_totals[:] = np.where( channel_totals == 0.0, 1.0, channel_totals )

    # Normalize
    predictions[:] /= channel_totals[...,None]

