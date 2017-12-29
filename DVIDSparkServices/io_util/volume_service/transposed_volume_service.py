import numpy as np

from . import VolumeServiceReader

class TransposedVolumeService(VolumeServiceReader):
    """
    Wraps an existing VolumeServiceReader and presents
    a transposed or rotated view of it.
    
    (Techinically, this is an example of the so-called
    "decorator" GoF pattern.)
    """
    
    # Rotations in the XY-plane, about the Z axis
    XY_CLOCKWISE_90 = ['z', 'x', '1-y']
    XY_COUNTERCLOCKWISE_90 = ['z', '1-x', 'y']
    XY_ROTATE_180 = ['z', '1-y', '1-x']

    # Rotations in the XZ-plane, about the Y axis
    XZ_CLOCKWISE_90 = ['x', 'y', '1-z']
    XZ_COUNTERCLOCKWISE_90 = ['1-x', 'y', 'z']
    XZ_ROTATE_180 = ['1-z' 'y', '1-x']

    # Rotations in the YZ-plane, about the X axis
    YZ_CLOCKWISE_90 = ['y', '1-z', 'x']
    YZ_COUNTERCLOCKWISE_90 = ['1-y', 'z', 'x']
    YZ_ROTATE_180 = ['1-z', '1-y', 'x']

    # No-op transpose; identity
    NO_TRANSPOSE = ['z', 'y', 'x']

    def __init__(self, original_volume_service, new_axis_order=NO_TRANSPOSE):
        self.original_volume_service = original_volume_service
        self.new_axis_order = new_axis_order

        assert len(new_axis_order) == 3
        assert not (set(new_axis_order) - set(['z', 'y', 'x', '1-z', '1-y', '1-x'])), \
            f"Invalid axis order items: {new_axis_order}"
        
        self.axis_names = [ a[-1] for a in new_axis_order ]
        assert set(self.axis_names) == set(['z', 'y', 'x'])
        self.transpose_order = ['zyx'.index(a) for a in self.axis_names] # where to find the new axis in the old order
        self.rev_transpose_order = [self.axis_names.index(a) for a in 'zyx'] # where to find the original axis in the new order
        self.axis_inversions = [a.startswith('1-') for a in new_axis_order]

        for i, (new, orig) in enumerate( zip(new_axis_order, 'zyx') ):
            if new != orig:
                assert self.bounding_box_zyx[0, i] == 0, \
                    "Bounding box must start at the origin for transposed axes."
    
    @property
    def dtype(self):
        return self.original_volume_service.dtype

    @property
    def block_width(self):
        return self.original_volume_service.block_width

    @property
    def preferred_message_shape(self):
        orig = self.original_volume_service.preferred_message_shape
        return np.array([orig[i] for i in self.transpose_order])

    @property
    def bounding_box_zyx(self):
        orig_start = self.original_volume_service.bounding_box_zyx[0]
        orig_stop = self.original_volume_service.bounding_box_zyx[1]

        start = np.array([orig_start[i] for i in self.transpose_order])
        stop = np.array([orig_stop[i] for i in self.transpose_order])

        return np.array([start, stop])

    def get_subvolume(self, new_box_zyx, scale=0):
        """
        Extract the subvolume, specified in new (transposed) coordinates from the
        original volume service, then transpose the result accordingly before returning it.
        """
        new_start = new_box_zyx[0]
        new_stop = new_box_zyx[1]
        
        orig_start = [new_start[i] for i in self.rev_transpose_order]        
        orig_stop = [new_stop[i] for i in self.rev_transpose_order]
        
        orig_box = np.array([orig_start, orig_stop])
        orig_bb = self.original_volume_service.bounding_box_zyx
        orig_bb //= 2**scale
        
        for i, inverted_name in enumerate(['1-z', '1-y', '1-x']):
            if inverted_name in self.new_axis_order:
                assert orig_bb[0, i] == 0
                Bw = _bounding_box_width = orig_bb[1, i]
                orig_box[:, i] = Bw - orig_box[:, i]
                orig_box[:, i] = orig_box[::-1, i]
                
        inversion_slices = tuple( { False: slice(None), True: slice(None, None, -1) }[inv]
                                  for inv in self.axis_inversions )

        data = self.original_volume_service.get_subvolume(orig_box, scale)
        data = data.transpose(self.transpose_order)
        data = data[inversion_slices]

        # Force contiguous so caller doesn't have to worry about it.
        data = np.asarray(data, order='C')
        return data

