import numpy as np
from skimage.util.shape import view_as_blocks

from DVIDSparkServices.util import box_to_slicing
from DVIDSparkServices.reconutils.downsample import downsample_labels_3d, downsample_raw, downsample_box

from . import VolumeServiceReader

RescaleLevelSchema = \
{
    "description": "Level to rescale the original input source when reading.\n"
                   "Presents a resized view of the original volume.\n"
                   "Examples:\n"
                   "  0: no rescaling\n"
                   "  1: downsample by 2x\n"
                   "  2: downsample by 4x\n"
                   " -1: upsample by 2x",
    "type": "integer",
    "default": 0
}

class ScaledVolumeService(VolumeServiceReader):
    """
    Wraps an existing VolumeServiceReader and presents
    a scaled view of it.
    
    (Technically, this is an example of the so-called
    "decorator" GoF pattern.)
    
    Notes:
        - uint64 data is assumed to be label data, and it is downsampled accordingly, and precisely
        - All other data is assumed to be grayscale. Downsampling is IMPRECISE because this class
          does not fetch a halo before computing the downsample.
        - In both cases, upsampling is performed without any interpolation.
    """
    def __init__(self, original_volume_service, scale_delta=0):
        self.original_volume_service = original_volume_service
        self.scale_delta = scale_delta

    @property
    def base_service(self):
        return self.original_volume_service.base_service

    @property
    def dtype(self):
        return self.original_volume_service.dtype

    @property
    def block_width(self):
        bw = int(self.original_volume_service.block_width // 2.**self.scale_delta)
        bw = max(1, bw) # Never shrink below width of 1
        return bw

    @property
    def preferred_message_shape(self):
        # We return the preferred_message_shape for scale 0.
        # If we're able to make a direct call to the original service (i.e. we don't have to downample it ourselves),
        # then just return the native preferred_message_shape
        #
        # This is a bit ugly.  See TODO in get_subvolume().
        if self.scale_delta in self.original_volume_service.available_scales:
            return self.original_volume_service.preferred_message_shape
        else:
            ms = (self.original_volume_service.preferred_message_shape // 2**self.scale_delta).astype(np.uint32)
            ms = np.maximum(ms, 1) # Never shrink below 1 pixel in each dimension
            return ms

    @property
    def bounding_box_zyx(self):
        bb = (self.original_volume_service.bounding_box_zyx // 2**self.scale_delta).astype(np.uint32)
        
        # Avoid shrinking the bounding box to 0 pixels in any dimension.
        bb[1] = np.maximum(bb[1], bb[0]+1)
        return bb

    @property
    def available_scales(self):
        return self.original_volume_service.available_scales

    def get_subvolume(self, box_zyx, scale=0):
        """
        Extract the subvolume, specified in new (scaled) coordinates from the
        original volume service, then scale result accordingly before returning it.
        
        TODO: It would be better to request the scale (from among the available-scales)
              that is closest to the final adjusted_scale.
              In the current implementation, it's just assumed that the requested scale exists,
              and then we downsample/upsample according to self.scale_delta.
        """
        adjusted_scale = scale + self.scale_delta
        if adjusted_scale in self.original_volume_service.available_scales:
            # The original source already has the data at the necessary scale.
            return self.original_volume_service.get_subvolume( box_zyx, adjusted_scale )

        if self.scale_delta > 0:
            orig_box_zyx = box_zyx * 2**self.scale_delta
            orig_data = self.original_volume_service.get_subvolume(orig_box_zyx, scale)

            if self.dtype == np.uint64:
                # Assume that uint64 means labels.
                downsampled_data, _ = downsample_labels_3d( orig_data, 2**self.scale_delta )
            else:
                downsampled_data = downsample_raw( orig_data, self.scale_delta )[-1]
            return downsampled_data
        else:
            upsample_factor = int(2**-self.scale_delta)
            orig_box_zyx = downsample_box(box_zyx, np.array(3*(upsample_factor,)))
            orig_data = self.original_volume_service.get_subvolume(orig_box_zyx, scale)

            orig_shape = np.array(orig_data.shape)
            upsampled_data = np.empty( orig_shape * upsample_factor, dtype=self.dtype )
            v = view_as_blocks(upsampled_data, 3*(upsample_factor,))
            v[:] = orig_data[:,:,:,None, None, None]

            relative_box = box_zyx - upsample_factor*orig_box_zyx[0]
            requested_data = upsampled_data[box_to_slicing(*relative_box)]

            # Force contiguous so caller doesn't have to worry about it.
            return np.asarray(requested_data, order='C')
