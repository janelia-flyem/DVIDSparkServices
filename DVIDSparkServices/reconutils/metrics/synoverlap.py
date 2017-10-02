from .overlap import *

"""Maintains overlap information .
"""
class SynOverlapTable(OverlapTable):
    def __init__(self, overlaps, comparison_type, leftovers):
        super(SynOverlapTable, self).__init__(overlaps, comparison_type)
        self.leftovers = leftovers
    
    def combine_tables(self, connections2):
        (leftover1, prop1) = self.leftovers
        (leftover2, prop2) = connections2.leftovers

        super(SynOverlapTable, self).combine_tables(connections2)
        prop1.update(prop2)

        for body, indexset in leftover2.items():
            if body not in leftover1:
                leftover1[body] = indexset
            else:
                leftover1[body] = leftover1[body].union(indexset)

        new_leftovers = {} 
        # try to resolve more unknown values
        for body, indexset in leftover1.items():
            for index in indexset:
                if index in prop1:
                    if prop1[index] != -1:
                        if body not in self.overlap_map:
                            # add one point
                            self.overlap_map[body] = set([(prop1[index],1)])
                        else:
                            rm_overlap = None
                            for (body2, overlap) in self.overlap_map[body]:
                                if body2 == prop1[index]:
                                    rm_overlap = (body2, overlap)
                                    break
                            overlap = 0
                            if rm_overlap is not None:
                                self.overlap_map[body].remove(rm_overlap)
                                overlap = rm_overlap[1]
                            self.overlap_map[body].add((prop1[index], overlap+1))
                else:
                    # repopulate
                    if body not in new_leftovers:
                        new_leftovers[body] = set()
                    new_leftovers[body].add(index)
      
        # update list
        self.leftovers = (new_leftovers, prop1)


