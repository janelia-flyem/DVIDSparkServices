import numpy as np

from . import VolumeServiceReader

NewAxisOrderSchema = \
{
    "description": "How to present the volume, in terms of the source volume axes.",
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": { "enum": ["x", "y", "z", "1-x", "1-y", "1-z"] },
    "default": ["x", "y", "z"] # no transpose
}

class TransposedVolumeService(VolumeServiceReader):
    """
    Wraps an existing VolumeServiceReader and presents
    a transposed or rotated view of it.
    
    (Technically, this is an example of the so-called
    "decorator" GoF pattern.)
    """
    
    ## These constants are expressed using X,Y,Z conventions!
    
    # Rotations in the XY-plane, about the Z axis
    XY_CLOCKWISE_90 = ['1-y', 'x', 'z']
    XY_COUNTERCLOCKWISE_90 = ['y', '1-x', 'z']
    XY_ROTATE_180 = ['1-x', '1-y', 'z']

    # Rotations in the XZ-plane, about the Y axis
    XZ_CLOCKWISE_90 = ['1-z', 'y', 'x']
    XZ_COUNTERCLOCKWISE_90 = ['z', 'y', '1-x']
    XZ_ROTATE_180 = ['1-x', 'y', '1-z']

    # Rotations in the YZ-plane, about the X axis
    YZ_CLOCKWISE_90 = ['x', '1-z', 'y']
    YZ_COUNTERCLOCKWISE_90 = ['x', 'z', '1-y']
    YZ_ROTATE_180 = ['x', '1-y', '1-z']

    # No-op transpose; identity
    NO_TRANSPOSE = ['x', 'y', 'z']

    def __init__(self, original_volume_service, new_axis_order_xyz=NO_TRANSPOSE):
        """
        Note: new_axis_order_xyz should be provided in [x.y,z] order,
              exactly as it should appear in config files.
              (e.g. see NO_TRANSPOSE above).
        """
        assert len(new_axis_order_xyz) == 3
        assert not (set(new_axis_order_xyz) - set(['z', 'y', 'x', '1-z', '1-y', '1-x'])), \
            f"Invalid axis order items: {new_axis_order_xyz}"

        new_axis_order_zyx = new_axis_order_xyz[::-1]
        self.new_axis_order_zyx = new_axis_order_zyx
        self.original_volume_service = original_volume_service

        
        self.axis_names = [ a[-1] for a in new_axis_order_zyx ]
        assert set(self.axis_names) == set(['z', 'y', 'x'])
        self.transpose_order = tuple('zyx'.index(a) for a in self.axis_names) # where to find the new axis in the old order
        self.rev_transpose_order = tuple(self.axis_names.index(a) for a in 'zyx') # where to find the original axis in the new order
        self.axis_inversions = [a.startswith('1-') for a in new_axis_order_zyx]

        for i, (new, orig) in enumerate( zip(new_axis_order_zyx, 'zyx') ):
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
        return self.original_volume_service.preferred_message_shape[(self.transpose_order,)]

    @property
    def bounding_box_zyx(self):
        return self.original_volume_service.bounding_box_zyx[:, self.transpose_order]

    @property
    def available_scales(self):
        raise self.original_volume_service.available_scales

    def get_subvolume(self, new_box_zyx, scale=0):
        """
        Extract the subvolume, specified in new (transposed) coordinates from the
        original volume service, then transpose the result accordingly before returning it.
        """
        new_box_zyx = np.asarray(new_box_zyx)
        orig_box = new_box_zyx[:, self.rev_transpose_order]
        orig_bb = self.original_volume_service.bounding_box_zyx // 2**scale
        
        for i, inverted_name in enumerate(['1-z', '1-y', '1-x']):
            if inverted_name in self.new_axis_order_zyx:
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
