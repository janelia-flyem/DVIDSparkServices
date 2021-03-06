"""
Simple algorithm that re-implements the Segmentor segment function

This module is a placeholder to indicate how to use the segmentation
plugin architecture.
"""
from DVIDSparkServices.reconutils import misc
from DVIDSparkServices.reconutils.Segmentor import Segmentor

class DefaultGrayOnly(Segmentor):

    def segment(self, subvols, gray_vols):
        """
        Simple, default seeded watershed off of grayscale, using nothing
        but the (inverted) grayscale intensities as membrane probabilities.
        """
        # only looks at gray values
        # TODO: should mask out based on ROI ?!
        # (either mask gray or provide mask separately)
        def _segment(gray_chunks):
            (_subvolume, gray) = gray_chunks

            mask = misc.find_large_empty_regions(gray)
            predictions = misc.naive_membrane_predictions(gray, mask)
            supervoxels = misc.seeded_watershed(predictions, mask, seed_threshold=0.2, seed_size=5 )
            agglomerated_sp = misc.noop_agglomeration(gray, predictions, supervoxels)
            return agglomerated_sp

        # preserver partitioner
        return subvols.zip(gray_vols).map(_segment, True)
