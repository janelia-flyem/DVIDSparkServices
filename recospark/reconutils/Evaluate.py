import libNeuroProofMetrics as np
from math import log
import numpy

class Evaluate(object):
    def __init__(self, config):
        pass

    def calcoverlap(self, label_pairs):
        pt1, pt2, label1, label2 = label_pairs

        # ?! avoid conversion
        label1 = label1.astype(numpy.float64)
        label2 = label2.astype(numpy.float64)
        stack1 = np.Stack(label1)
        stack2 = np.Stack(label2)

        overlaps12 = stack1.find_overlaps(stack2)
        overlaps21 = stack2.find_overlaps(stack1)

        results = []

        for overlap in overlaps12:
            results.append((True, overlap))

        for overlap in overlaps21:
            results.append((False, overlap))

        return results

    def body_vi(self, bodies_overlap):
        # input: (body1id, list[(body2id, count)])
        # output: (body1id, VI_unnorm, count)

        body1, overlap_list = bodies_overlap

        # accumulate values
        body2counts = {}
        total = 0
        for (body2, val)  in overlap_list:
            if (body1, body2) in body2counts:
                body2counts[(body1, body2)] += val
            else:
                body2counts[(body1, body2)] = val
            total += val

        vi_unnorm = 0
        for key, val in body2counts.items():
            # compute VI
            vi_unnorm += val*log(total/val)/log(2.0)

        return (body1, vi_unnorm, total)


    def is_vol1(self, overlap_pair):
        vol1, overlap = overlap_pair
        return vol1

    def is_vol2(self, overlap_pair):
        vol1, overlap = overlap_pair
        return not vol1
