"""
Simple algorithm that re-implements the Segmentor segment function

This module is a placeholder to indicate how to use the segmentation
plugin architecture.
"""
from __future__ import print_function, absolute_import
from DVIDSparkServices.reconutils.Segmentor import Segmentor

class precomputedpipeline(Segmentor):

    def segment(self, subvols, gray_vols=None):
        """
        Does not require the gray vols
        """
        # read pre-computed segmentation

        pathloc = self.segmentor_config["segpath"]
        
        def _segment(subvolume):
            z1 = subvolume.box.z1
            y1 = subvolume.box.y1
            x1 = subvolume.box.x1

            fileloc = (pathloc + "/%d_%d_%d.h5") % (z1,y1,x1)
            import h5py
            import numpy
            print("!!", fileloc)
            try:
                hfile = h5py.File(fileloc, 'r')
                seg = numpy.array(hfile["segmentation"])
                print("!! good")
                return seg.astype(numpy.uint32)
            except:
                print("!! bad")
                return numpy.zeros((552,552,552), numpy.uint32)

        # preserver partitioner
        return subvols.map(_segment, True)
