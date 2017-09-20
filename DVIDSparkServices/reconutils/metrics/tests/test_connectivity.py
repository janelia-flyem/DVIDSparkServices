from __future__ import division
import unittest
from DVIDSparkServices.reconutils.metrics.plugins.connectivity_stat import *
from DVIDSparkServices.reconutils.metrics.segstats import SubvolumeStats

class Testconnectivity(unittest.TestCase):
    def test_bodymatch(self):
        """Verify output from hungarian matching of bodies.
        """

        # create a sample overlap map between two segmentations
        overlap = {}
        overlap[3] = set([(100, 3210), (101, 242), (50, 1), (30, 1)])
        overlap[30] = set([(100, 300), (101, 100), (50, 1), (30, 20)])
        overlap[10] = set([(101, 50), (30, 20)])


        # use default threshold of 0
        stats = SubvolumeStats(0, ptfilter=0)
        cs = connectivity_stat(thresholds = [1,5])
        cs.set_segstats(stats)
        matches = cs._compute_bodymatch(overlap)
       
        # check output
        self.assertTrue(len(matches) == 4)
        self.assertTrue((3, 100, 3210, 3454, 3510) in matches)
        self.assertTrue((0, 50, 0, 0, 2) in matches)
        self.assertTrue((30, 101, 100, 421, 392) in matches)
        self.assertTrue((10, 30, 20, 70, 41) in matches)

        # ignore bodies smaller than 3 (will effectively eliminate a row)
        stats = SubvolumeStats(0, ptfilter=3)
        cs = connectivity_stat(thresholds = [1,5])
        cs.set_segstats(stats)
        matches = cs._compute_bodymatch(overlap)
        self.assertTrue(len(matches) == 3)
        self.assertTrue((0, 50, 0, 0, 2) not in matches)

        # add a body1 with no matches
        overlap[45] = set([(100, 5)])
        stats = SubvolumeStats(0, ptfilter=0)
        cs = connectivity_stat(thresholds = [1,5])
        cs.set_segstats(stats)
        matches = cs._compute_bodymatch(overlap)
        self.assertTrue(len(matches) == 4)
        self.assertTrue((45, 50, 0, 5, 2) in matches)

        # removes empty match for body1
        stats = SubvolumeStats(0, ptfilter=3)
        cs = connectivity_stat(thresholds = [1,5])
        cs.set_segstats(stats)
        matches = cs._compute_bodymatch(overlap)
        self.assertTrue(len(matches) == 4)
        self.assertTrue((45, 0, 0, 5, 0) in matches)

        # more body1 than body2
        overlap[15] = set([(100, 5)])
        stats = SubvolumeStats(0, ptfilter=3)
        cs = connectivity_stat(thresholds = [1,5])
        cs.set_segstats(stats)
        matches = cs._compute_bodymatch(overlap)
        self.assertTrue(len(matches) == 5)

    def test_computetablestats(self):
        """Verify stats extracted from connectivity overlap.
        """
        stats = SubvolumeStats(0, ptfilter=0)
        cs = connectivity_stat(thresholds = [1,5])
        cs.set_segstats(stats)
    
        # create overlap map where 15 is below the threshold and body2=13 will get spurious match
        overlap = {}
        overlap[3] = set([(100, 3210), (101, 242), (50, 1), (30, 1)])
        overlap[30] = set([(100, 300), (101, 100), (50, 1), (30, 20), (13,3)])
        overlap[10] = set([(101, 50), (30, 20)])
        overlap[15] = set([(100, 1)])
        stats = SubvolumeStats(0, ptfilter=3)
        cs = connectivity_stat(thresholds = [1,5])
        cs.set_segstats(stats)
        matches = cs._compute_bodymatch(overlap)
 
        tablemap = {}        
        tablemap[((3 << 64) | 100)] = set([(((3 << 64) | 100), 50), (((30 << 64) | 100), 10), (((30 << 64) | 101), 4),  (((3 << 64) | 101), 7)])
        tablemap[((3 << 64) | 101)] = set([(((3 << 64) | 100), 3), (((30 << 64) | 101), 2), (((15 << 64) | 100), 1) ])
        tablemap[((30 << 64) | 101)] = set([(((30 << 64) | 101), 1)])

        # extract stats for at least 1 and 5 connections
        sumstats, bodystats, conntablestats = cs._compute_tablestats(matches, tablemap, "blah", [1, 5])  

        #self.assertTrue(sumstats == [55, 77, 3, [1, 5], [3, 1], [3, 2]])
        self.assertTrue(len(sumstats) == 3)
        self.assertTrue(sumstats[0]["val"] == round(55/float(77),4))
        self.assertTrue(sumstats[1]["val"] == round(3/float(3),4))
        self.assertTrue(sumstats[2]["val"] == round(1/float(2),4))
        
        self.assertTrue(len(bodystats[0]["bodies"]) == 2)
        self.assertTrue(3 in bodystats[0]["bodies"])
        self.assertEqual([54, [76, 100]], bodystats[0]["bodies"][3])
        self.assertTrue(30 in bodystats[0]["bodies"])
        self.assertEqual([1, [1, 101]], bodystats[0]["bodies"][30])
        self.assertTrue([3, 100, 3, 100, 50, 60] in conntablestats[0]["info"])
    
        # create overlap map where a body1 has no match
        overlap = {}
        overlap[3] = set([(100, 3210), (101, 242)])
        overlap[30] = set([(100, 300), (101, 100)])
        overlap[15] = set([(100, 5), (23, 3)])

        stats = SubvolumeStats(0, ptfilter=4)
        cs = connectivity_stat(thresholds = [1,5])
        cs.set_segstats(stats)
        matches = cs._compute_bodymatch(overlap)
        tablemap = {}
        # create edge with removed node
        tablemap[((15 << 64) | 23)] = set([(((3 << 64) | 100), 4)])
       
        sumstats, bodystats, conntablestats = cs._compute_tablestats(matches, tablemap, "blah", [1,5])
        self.assertTrue(len(bodystats[0]["bodies"]) == 1)
        #self.assertTrue(sumstats == [0, 4, 1, [1, 5], [0, 0], [1, 0]])
        self.assertTrue(len(sumstats) == 3)
        self.assertTrue(sumstats[0]["val"] == round(0/float(4),4))
        self.assertTrue(sumstats[1]["val"] == round(0/float(1),4))
        self.assertTrue(sumstats[2]["val"] == 0)
        
 
if __name__ == "__main__":
    unittest.main()
