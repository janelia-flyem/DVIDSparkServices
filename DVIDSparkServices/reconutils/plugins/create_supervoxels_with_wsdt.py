import numpy as np
import wsdt

def create_supervoxels_with_wsdt( boundary_volume,
                                  mask,
                                  boundary_channel=0,
                                  pmin=0.5,
                                  minMembraneSize=10000,
                                  minSegmentSize=300,
                                  sigmaMinima=3,
                                  sigmaWeights=1.6,
                                  cleanCloseSeeds=False ):
    """
    Generate supervoxels using Timo's watershed of the distance-transform method.
    """
    assert boundary_volume.ndim == 4, "Expected a 4D volume."
    boundary_volume = boundary_volume[..., boundary_channel]

    if mask is not None:
        # Forbid the watershed from bleeding into the masked area prematurely
        mask = mask.astype(np.bool, copy=False)
        # Mask is now inverted
        inverted_mask = np.logical_not(mask, out=mask)
        boundary_volume[inverted_mask] = 2.0

    watershed = wsdt.wsDtSegmentation( boundary_volume,
                                       pmin,
                                       minMembraneSize,
                                       minSegmentSize,
                                       sigmaMinima,
                                       sigmaWeights,
                                       cleanCloseSeeds=False,
                                       returnSeedsOnly=False )

    if mask is not None:
        watershed[inverted_mask] = 0
    return watershed
    