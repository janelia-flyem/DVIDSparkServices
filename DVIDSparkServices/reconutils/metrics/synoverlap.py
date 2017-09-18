"""Maintains overlap information .
"""
class SynOverlapTable(OverlapTable):
    def __init__(self, overlaps, comparison_type, leftovers):
        super(SynOverlapTable, self).__init__(overlaps, comparison_type)
        self.leftovers = leftovers
    
    def combine_tables(self, connections1, connections2):
        for iter1 in range(0, len(connections1)):
            table1, (leftover1, prop1) = connections1[iter1]
            table2, (leftover2, prop2) = connections2[iter1]
        
            table1.combine_tables(table2)
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
                            if body not in table1.overlap_map:
                                # add one point
                                table1.overlap_map[body] = set([(prop1[index],1)])
                            else:
                                rm_overlap = None
                                for (body2, overlap) in table1.overlap_map[body]:
                                    if body2 == prop1[index]:
                                        rm_overlap = (body2, overlap)
                                        break
                                overlap = 0
                                if rm_overlap is not None:
                                    table1.overlap_map[body].remove(rm_overlap)
                                    overlap = rm_overlap[1]
                                table1.overlap_map[body].add((prop1[index], overlap+1))
                    else:
                        # repopulate
                        if body not in new_leftovers:
                            new_leftovers[body] = set()
                        new_leftovers[body].add(index)
          
            # update list
            connections1[iter1] = (table1, (new_leftovers, prop1))



