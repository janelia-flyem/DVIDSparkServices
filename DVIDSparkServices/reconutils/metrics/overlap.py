"""Stores overlap information between two sets of labels.
"""
class OverlapTable(object):
    def __init__(self, overlaps, comparison_type, mapping1=None, mapping2=None):
        """Init.

        Args:
            overlaps (list): Overlap data for seg1->seg2
            comparison_type (ComparisonType): type of comparison
            mapping1 (dict): label -> new label (for inputs)
            mapping2 (dict): label -> new label (for outputs)
        """
        self.comparison_type = comparison_type
        self.overlap_map = {}

        for item in overlaps:
            body1, body2, overlap = item
            if body1 not in self.overlap_map:
                self.overlap_map[body1] = set()
            self.overlap_map[body1].add((body2, int(overlap)))

    def has_remap():
        """Indicates whether overlap has remap tables.
        
        A subvolume where CC is applied will have unique
        body ids for disjoint commponents.  Applying remap
        will undo the connected component computation.
        """
        return mapping1 != None or mapping2 != None

    def get_name():
        """Unique name of overlap table.
        """
        return str(self.comparison_type)

    def get_comparison_type():
        """Returns comparison type (string).
        """
        return self.comparison_type.get_type() 


    def apply_remap():
        """Applies existing body to body mappings to the overlap tables.

        Returns:
            self or a copy of self that is remapped

        """

        if not has_remap():
            return self

        newdata = self.copy()
        newdata_partial_remap()
        return newdata

    def _partial_remap():
        """Partial remap the overlap table.

        Use mapping1 to remap a subset of the table inputs
        and mapping2 to remap the overlapped outputs for
        each input.

        Note:
            remapped labels must be disjoint from original labels.
        
        Returns:
            Internal structures are updated.

        """
        
        del_keys = {}
        keep_ids = set()

        for key, val in self.overlap_map.items():
            if key in self.mapping1:
                # find inputs that need to be remapped
                if self.mapping1[key] not in del_keys:
                    del_keys[self.mapping1[key]] = set()
                del_keys[self.mapping1[key]].add(key)
                keep_ids.add(self.mapping1[key])
            else:
                keep_ids.add(key)
            new_overlap = {}

            # handle new overlaps since mapping could cause merge
            for (key2, val2) in val:
                new_key = key2
                if key2 in self.mapping2:
                    new_key = self.mapping2[key2]
                if new_key not in new_overlap:
                    new_overlap[new_key] = 0
                new_overlap[new_key] += val2
            
            # update overlap list
            new_overlap_set = set()
            for body, overlap in new_overlap.items():
                new_overlap_set.add((body, overlap)) 
            self.overlap_map[key] = new_overlap_set
        
        temp_overlap = self.overlap_map.copy()
    
        # merge rows mapping to same body, remove old body
        for newbody, bodylist in del_keys.items():
            self.overlap_map[newbody] = set()
            for bodyold in bodylist:
                self._merge_row(self.overlap_map[newbody], temp_overlap[bodyold])
                if bodyold not in keep_ids:
                    del self.overlap_map[bodyold]

    def combine_tables(self, overlap2):
        if self.comparison_type != overlap2.comparison_type:
            raise Exception("incomparable overlap stat types")

        # if remap tables exist, perform the remap here
        self._partial_remap()
        overlap._partial_remap()

        for body, overlapset in overlap2.overlap_map.items():
            if body not in self.overlap_map:
                self.overlap_map[body] = overlapset
            else:
                self._merge_row(self.overlap_map[body], overlap2.overlap_map[body])

    def _merge_row(self, row1, row2):
        """Merge row2 to row1, update overlap.

        Args:
            row1 (set): set of (body, overlap)
            row2 (set): set of (body, overlap)
        
        """

        duprow = list(row1)
        duprow.extend(list(row2))
        row1.clear()
        overlap_map = {}

        for body, overlap in duprow:
            if body not in overlap_map:
                overlap_map[body] = 0
            overlap_map[body] += overlap

        for body, overlap in overlap_map.items():
            row1.add((body, overlap))


