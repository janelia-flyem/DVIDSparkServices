"""
Simple algorithm that re-implements the Segmentor segment function

This module is a placeholder to indicate how to use the segmentation
plugin architecture.
"""
from DVIDSparkServices.reconutils import misc
from DVIDSparkServices.reconutils.Segmentor import Segmentor
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray

class DefaultGrayOnly(Segmentor):

    def segment(self, gray_chunks):
        """
        Simple, default seeded watershed off of grayscale, using nothing
        but the (inverted) grayscale intensities as membrane probabilities.
        """
        # only looks at gray values
        # TODO: should mask out based on ROI ?!
        # (either mask gray or provide mask separately)
        def _segment(gray_chunks):
            (subvolume, gray) = gray_chunks

            mask = misc.find_large_empty_regions(gray)
            predictions = misc.naive_membrane_predictions(gray, mask)
            supervoxels = misc.seeded_watershed(predictions, mask, seed_threshold=0.2, seed_size=5 )
            agglomerated_sp = misc.noop_aggolmeration(predictions, supervoxels)
            
            max_id = agglomerated_sp.max()
            subvolume.set_max_id(max_id)

            agglomerated_sp_compressed = CompressedNumpyArray(agglomerated_sp)
            return (subvolume, agglomerated_sp_compressed)

        # preserver partitioner
        return gray_chunks.mapValues(_segment)
