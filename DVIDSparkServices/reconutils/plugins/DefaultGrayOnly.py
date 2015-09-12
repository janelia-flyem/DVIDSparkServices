"""Simple algorithm that re-implements the Segmentor segment function

This module is a placeholder to indicate how to use the segmentation
plugin architecture.


"""

from DVIDSparkServices.reconutils.Segmentor import Segmentor

class DefaultGrayOnly(Segmentor):
    def __init__(self, context, config, options):
        super(DefaultGrayOnly, self).__init__(context, config, options)
    
    def segment(self, gray_chunks):
        """Simple, default seeded watershed off of grayscale

        """
        
        from scipy.ndimage import label
        from scipy.ndimage.morphology import binary_closing
        from numpy import bincount 
        from skimage import morphology as skmorph
        seed_threshold = self.SEED_SIZE_THRES
        seed_cytothreshold = self.SEED_THRES
        background_size = self.BACKGROUND_SIZE
        from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray

        # only looks at gray values
        # TODO: should mask out based on ROI ?!
        # (either mask gray or provide mask separately)
        def _segment(gray_chunks):
            import numpy
            (subvolume, gray) = gray_chunks

            # extract seed mask from grayscale
            seedmask = gray >= seed_cytothreshold

            # closing operation to clean up seed mask
            seedmask_closed = binary_closing(seedmask, iterations=2)

            # split into connected components
            seeds = label(seedmask_closed)[0]

            # filter small connected components
            component_sizes = bincount(seeds.ravel())
            small_components = component_sizes < seed_threshold 
            small_locations = small_components[seeds]
            seeds[small_locations] = 0

            # mask out background (don't have to 0 out seeds since
            mask_mask = numpy.zeros(gray.shape).astype(numpy.uint8)
            mask_mask[gray == 0] = 1
            mask_mask = label(mask_mask)[0]
            mask_sizes = bincount(mask_mask.ravel())
            mask_components = mask_sizes < background_size 
            small_mask_locations = mask_components[mask_mask]
            mask_mask[small_mask_locations] = 0
            mask = mask_mask == 0 

            # invert and convert to 0 to 1 range for watershed
            new_gray = (255.0-gray)/255.0

            # compute watershed (mask any boundary)
            supervoxels = skmorph.watershed(new_gray, seeds,
                    None, None, mask)
            
            max_id = supervoxels.max()
            supervoxels_compressed = CompressedNumpyArray(supervoxels)
            subvolume.set_max_id(max_id)

            return (subvolume, supervoxels_compressed)

        # preserver partitioner
        return gray_chunks.mapValues(_segment)



