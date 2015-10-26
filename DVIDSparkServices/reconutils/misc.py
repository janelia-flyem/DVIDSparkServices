import numpy
import scipy.ndimage
import DVIDSparkServices.reconutils.morpho

def find_large_empty_regions(grayscale_vol, parameters={}):
    """
    Returns mask that excludes large background (0-valued) regions, if any exist.
    """
    # Start with defaults, then update with user's
    all_parameters = { 'min-background-voxel-count' : 100 }
    all_parameters.update(parameters)
    
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
    small_cc_selections = cc_sizes < all_parameters['min-background-voxel-count'] 
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

def naive_membrane_predictions(grayscale_vol, mask_vol=None, parameters={} ):
    """
    Stand-in for membrane prediction, for testing purposes.
    Simply returns the inverted grayscale as our 'predictions'
    """
    # Just use inverted intensity as a proxy for membrane probability
    inverted = (255-grayscale_vol).astype(numpy.float32)
    inverted[:] /= 255 # Scale: Watershed step expects range 0.0-1.0
    return inverted

def seeded_watershed(boundary_volume, mask, parameters={}):
    """
    Perform a seeded watershed on the given volume.
    Seeds are generated using a seed-threshold and minimum seed-size.
    """
    # Start with defaults, then update with user's
    all_parameters = { 'seed-threshold' : 0.5,
                       'seed-size' : 5 }
    all_parameters.update(parameters)

    ws = DVIDSparkServices.reconutils.morpho.seeded_watershed
    supervoxels = ws( boundary_volume,
                      all_parameters['seed-threshold'],
                      all_parameters['seed-size'],
                      mask )
    return supervoxels

def noop_aggolmeration(bounary_volume, supervoxels, parameters={}):
    """
    Stand-in for an agglomeration function.
    This function returns the supervoxels unchanged.
    """
    return supervoxels
