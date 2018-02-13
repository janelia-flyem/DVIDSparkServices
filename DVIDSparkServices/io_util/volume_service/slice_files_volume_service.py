import os
import re
import glob

import numpy as np
from PIL import Image
from jsonschema import validate

from DVIDSparkServices.util import replace_default_entries, box_to_slicing
from . import VolumeServiceReader, VolumeServiceWriter, GeometrySchema


SliceFilesServiceSchema = \
{
    "description": "Parameters specify a source of grayscale data from image slices.",
    "type": "object",

    "required": ["slice-path-format"],

    "default": {},
    "properties": {
        "slice-path-format": {
            "description": 'String format for image Z-slice paths, using python format-string syntax, \n'
                           'e.g. "/path/to/slice{:05}.png"  \n'
                           'Some workflows may also support the prefix "gs://" for gbucket data.',
            "type": "string",
            "minLength": 1
        },
        "slice-xy-offset": {
            "description": "The XY-offset indicating where the slices reside within the global input coordinates, \n"
                           "That is, which global (X,Y) coordinate does pixel (0,0) of each slice correspond to? \n"
                           "(The Z-offset is presumed to be already encoded within the slice-path-format)",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 2,
            "maxItems": 2,
            "default": [0,0],
        }
    }
}

SliceFilesVolumeSchema = \
{
    "description": "Describes a volume from slice files on disk.",
    "type": "object",
    "default": {},
    "properties": {
        "slice-files": SliceFilesServiceSchema,
        "geometry": GeometrySchema
    }
}


class SliceFilesVolumeServiceReader(VolumeServiceReader):

    class NoSlicesFoundError(RuntimeError): pass

    def __init__(self, volume_config, config_dir):
        validate(volume_config, SliceFilesVolumeSchema)

        # Convert path to absolute if necessary (and write back to the config)
        slice_fmt = volume_config["slice-files"]["slice-path-format"]
        assert not slice_fmt.startswith('gs://'), "FIXME: Support gbuckets"
        if not slice_fmt.startswith('/'):
            slice_fmt = os.path.normpath( os.path.join(config_dir, slice_fmt) )

        # Determine complete bounding box
        default_bounding_box_zyx, dtype = determine_stack_attributes(slice_fmt)
        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        replace_default_entries(bounding_box_zyx, default_bounding_box_zyx)

        # Determine complete preferred "message shape" - one full output slice.
        output_slice_shape = bounding_box_zyx[1] - bounding_box_zyx[0]
        output_slice_shape[0] = 1
        preferred_message_shape_zyx = np.array(volume_config["geometry"]["message-block-shape"][::-1])
        replace_default_entries(preferred_message_shape_zyx, output_slice_shape)
        assert (preferred_message_shape_zyx == output_slice_shape).all(), \
            f"Preferred message shape for slice files must be a single Z-slice, and a complete XY output plane ({output_slice_shape}), "\
            f"not {preferred_message_shape_zyx}"

        available_scales = volume_config["geometry"]["available-scales"]
        assert available_scales == [0], \
            "Bad config: slice-files reader supports only scale zero."

        # Store members
        self._slice_fmt = slice_fmt
        self._dtype = dtype
        self._dtype_nbytes = np.dtype(dtype).type().nbytes
        self._bounding_box_zyx = bounding_box_zyx
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._available_scales = available_scales

        # Overwrite config entries that we might have modified
        volume_config["slice-files"]["slice-path-format"] = slice_fmt
        volume_config["geometry"]["bounding-box"] = bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = preferred_message_shape_zyx[::-1].tolist()

        # Forbid unsupported config entries
        assert volume_config["slice-files"]["slice-xy-offset"] == [0,0], \
            "Non-zero slice-xy-offset is not yet supported"
        assert volume_config["geometry"]["block-width"] == -1, \
            "Slice files have no concept of a native block width. Please leave it set to the default (-1)"

    @property
    def dtype(self):
        return self._dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def block_width(self):
        return -1
    
    @property
    def bounding_box_zyx(self):
        return self._bounding_box_zyx

    @property
    def available_scales(self):
        return self._available_scales

    def get_subvolume(self, box_zyx, scale=0):
        assert scale == 0, "Slice File reader only supports scale 0"
        z_offset = box_zyx[0,0]
        yx_box = box_zyx[:,1:]
        output = np.ndarray(shape=(box_zyx[1] - box_zyx[0]), dtype=self.dtype)
        for z in range(*box_zyx[:,0]):
            slice_path = self._slice_fmt.format(z)
            slice_data = np.array( Image.open(slice_path).convert("L") )
            output[z-z_offset] = slice_data[box_to_slicing(*yx_box)]
        return output

class SliceFilesVolumeServiceWriter(VolumeServiceWriter):

    def __init__(self, volume_config, config_dir):
        validate(volume_config, SliceFilesVolumeSchema)

        # Convert path to absolute if necessary (and write back to the config)
        slice_fmt = volume_config["slice-files"]["slice-path-format"]
        assert not slice_fmt.startswith('gs://'), "FIXME: Support gbuckets"
        if not slice_fmt.startswith('/'):
            slice_fmt = os.path.normpath( os.path.join(config_dir, slice_fmt) )

        slice_dir = os.path.dirname(slice_fmt)
        os.makedirs(slice_dir, exist_ok=True)

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        assert -1 not in bounding_box_zyx.flat[:], "Output bounding box must be completely specified."

        # Determine complete preferred "message shape" - one full output slice.
        output_slice_shape = bounding_box_zyx[1] - bounding_box_zyx[0]
        output_slice_shape[0] = 1
        preferred_message_shape_zyx = np.array(volume_config["geometry"]["message-block-shape"][::-1])
        replace_default_entries(preferred_message_shape_zyx, output_slice_shape)
        assert (preferred_message_shape_zyx == output_slice_shape).all(), \
            f"Preferred message shape for slice files must be a single Z-slice, and a complete XY output plane ({output_slice_shape}), "\
            f"not {preferred_message_shape_zyx}"

        # Store members
        self._slice_fmt = slice_fmt
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._bounding_box_zyx = bounding_box_zyx
        
        # Overwrite config entries that we might have modified
        volume_config["slice-files"]["slice-path-format"] = slice_fmt
        volume_config["geometry"]["bounding-box"] = bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = preferred_message_shape_zyx[::-1].tolist()

        # Forbid unsupported config entries
        assert volume_config["slice-files"]["slice-xy-offset"] == [0,0], \
            "Non-zero slice-xy-offset is not yet supported"
        assert volume_config["geometry"]["block-width"] == -1, \
            "Slice files have no concept of a native block width. Please leave it set to the default (-1)"


    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        offset_zyx = np.array(offset_zyx)
        assert scale == 0, "Currently, only writing scale 0 is supported."
        assert (offset_zyx[1:] == [0,0]).all(), \
            "Subvolumes must be written in complete slices. Writing partial slices is not supported."
        
        for sv_z, z_slice in enumerate(subvolume):
            z = sv_z + offset_zyx[0]
            slice_path = self._slice_fmt.format(z)
            Image.fromarray(z_slice).save(slice_path)

    @property
    def bounding_box_zyx(self):
        return self._bounding_box_zyx

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape

    @property
    def dtype(self):
        raise NotImplementedError
    
    @property
    def block_width(self):
        return -1

def determine_stack_attributes(slice_fmt):
    """
    Determine the shape and dtype of a stack of slices that already reside on disk.
    
    slice_fmt:
        Example: '/path/to/slices/z{:05d}-iso.png'
    
    Returns:
        maximal_bounding_box_zyx, dtype
    """
    prefix, _index_format, suffix = split_slice_fmt(slice_fmt)

    matching_paths = sorted( glob.glob(f"{prefix}*{suffix}") )
    if not matching_paths:
        raise SliceFilesVolumeServiceReader.NoSlicesFoundError(f"No slice files found to match pattern: {slice_fmt}")

    if (np.array(list(map(len, matching_paths))) != len(matching_paths[0])).all():
        raise RuntimeError("Image file paths are not all the same length. "
                           "Slice paths must use 0-padding for all slice indexes, e.g. zcorr.00123.png")

    min_available_index = int( matching_paths[0][len(prefix):-len(suffix)] )
    max_available_index = int( matching_paths[-1][len(prefix):-len(suffix)] )

    # Note: For simplicity, we read the slice shape from the first *available* slice,
    # regardless of the first slice we'll actually use. Should be fine.
    first_slice_path = slice_fmt.format(min_available_index)
    first_slice = np.array( Image.open(first_slice_path).convert("L") )
    first_height, first_width = first_slice.shape

    maximal_bounding_box_zyx = [[min_available_index, 0, 0],
                                [max_available_index+1, first_height, first_width]]
    
    return np.array(maximal_bounding_box_zyx), first_slice.dtype


def split_slice_fmt(slice_fmt):
    """
    Break up the slice_fmt into a prefix, index_format, and suffix.
    
    Example:
        prefix, index_format, suffix = split_slice_fmt('/path/to/slices/z-{:05d}-iso.png')
        assert prefix == '/path/to/slices/z-'
        assert index_format == '{:05d}'
        assert suffix == '-iso.png'    
    """
    if '%' in slice_fmt:
        raise RuntimeError("Please use python-style string formatting for 'basename': (e.g. zcorr.{:05d}.png)")

    match = re.match('^(.*)({[^}]*})(.*)$', slice_fmt)
    if not match:
        raise RuntimeError(f"Unrecognized format string for image basename: {slice_fmt}")

    prefix, index_format, suffix = match.groups()
    return prefix, index_format, suffix
