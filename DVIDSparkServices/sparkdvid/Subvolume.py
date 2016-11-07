"""Defines Subvolume class

Many DVID operations involve working on a large 3D dataset.
The subvolumes frequently define the RDD partitioning for
different transformations in sparkdvid.

"""

import collections
    
SubvolumeNamedTuple = collections.namedtuple('SubvolumeNamedTuple',
            'x1 y1 z1 x2 y2 z2')

class Subvolume(object):
    """Define subvolume datatype.

    The subvolume provides X,Y,Z locations in DVID coordinates
    and has other information like neighboring substacks
    (if this infor is computed).  It has several functions for
    helping to determine overlap between substacks.
    
    """

    def __init__(self, roi_id, roi, chunk_size, border):
        """Initializes subvolume.

        Args:
            roi_id (int): identifier key for subvolume (must be unique)
            roi ([]): x,y,z array
            chunk_size (int): dimension of subvolume (assume isotropic)
            border (int): border size surrounding core subvolume    
        """

        self.roi_id = roi_id
        self.roi = SubvolumeNamedTuple(roi[0],
                    roi[1], roi[2],
                    roi[0] + chunk_size,
                    roi[1] + chunk_size,
                    roi[2] + chunk_size)
        self.border = border
        self.local_regions = []

        # ROI is always in 32x32x32 blocks for now
        self.roi_blocksize = 32

        # index off of (z,y,x) block indices
        # TODO: Instead of listing block coordinates, it would be ~24x
        #       more RAM-efficient to maintain a bool array of the ROI mask
        #       (at block resolution)
        self.intersecting_blocks = []

    def __eq__(self, other):
        return (self.roi_id == other.roi_id and
                self.roi == other.roi and
                self.border == other.border)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        # TODO: We still assume unique roi_ids, and only use that in the hash,
        #       so that partitioning with roi_id is equivalent to partitioning on the Subvolume itself.
        #       If sparkdvid functions (e.g. map_grayscale8) are ever changed not to partition over roi_id,
        #       then we can change this hash function to include the other members, such as border, etc.
        #return hash( (self.roi_id, self.roi, self.border) )
        return hash(self.roi_id)

    @property
    def roi_with_border(self):
        """
        Read-only property.
        Same as self.roi, but expanded to include the border.
        """
        x1, y1, z1, x2, y2, z2 = self.roi
        return SubvolumeNamedTuple(x1 - self.border, y1 - self.border, z1 - self.border,
                                   x2 + self.border, y2 + self.border, z2 + self.border)


    def __str__(self):
        return "x{x1}-y{y1}-z{z1}--x{x2}-y{y2}-z{z2}"\
               .format(**self.roi.__dict__)

    # assume line[0] < line[1] and add border in calculation 
    def intersects(self, line1, line2):
        pt1, pt2 = line1[0], line1[1]
        pt1_2, pt2_2 = line2[0], line2[1]
        
        if (pt1_2 < pt2 and pt1_2 >= pt1) or (pt2_2 <= pt2 and pt2_2 > pt1):
            return True
        return False 

    # check if one of the dims touch
    def touches(self, p1, p2, p1_2, p2_2):
        if p1 == p2_2 or p2 == p1_2:
            return True
        return False

    # returns true if two rois overlap
    def recordborder(self, roi2):
        linex1 = [self.roi.x1, self.roi.x2]
        linex2 = [roi2.roi.x1, roi2.roi.x2]
        liney1 = [self.roi.y1, self.roi.y2]
        liney2 = [roi2.roi.y1, roi2.roi.y2]
        linez1 = [self.roi.z1, self.roi.z2]
        linez2 = [roi2.roi.z1, roi2.roi.z2]
       
        # check intersection
        if (self.touches(linex1[0], linex1[1], linex2[0], linex2[1]) and self.intersects(liney1, liney2) and self.intersects(linez1, linez2)) \
        or (self.touches(liney1[0], liney1[1], liney2[0], liney2[1]) and self.intersects(linex1, linex2) and self.intersects(linez1, linez2)) \
        or (self.touches(linez1[0], linez1[1], linez2[0], linez2[1]) and self.intersects(liney1, liney2) and self.intersects(linex1, linex2)):
            # save overlapping substacks
            self.local_regions.append((roi2.roi_id, roi2.roi))
            roi2.local_regions.append((self.roi_id, self.roi))



