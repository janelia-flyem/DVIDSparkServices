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
    return background_mask

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
    return inverted

def seeded_watershed(boundary_volume, mask, seed_threshold=0.2, seed_size=5):
    """
    Perform a seeded watershed on the given volume.
    Seeds are generated using a seed-threshold and minimum seed-size.
    """
    ws = DVIDSparkServices.reconutils.morpho.seeded_watershed
    supervoxels = ws( boundary_volume, seed_threshold, seed_size, mask )
    return supervoxels

def noop_aggolmeration(bounary_volume, supervoxels):
    """
    Stand-in for an agglomeration function.
    This function returns the supervoxels unchanged.
    """
    return supervoxels
