import os

import numpy as np
from jsonschema import validate

import z5py

from DVIDSparkServices.util import box_to_slicing, replace_default_entries

from . import VolumeServiceReader, GeometrySchema

N5ServiceSchema = \
{
    "description": "Parameters specify a DVID node",
    "type": "object",
    "required": ["path", "dataset-name"],
    "default": {},
    
    "properties": {
        "path": {
            "description": "Path to the n5 parent directory, which may contain multiple datasets",
            "type": "string",
            "minLength": 1
        },
        "dataset-name": {
            "description": "Name of the volume",
            "type": "string",
            "minLength": 1
        }
    }
}

N5VolumeSchema = \
{
    "description": "Describes a volume from N5.",
    "type": "object",
    "default": {},
    "properties": {
        "n5": N5ServiceSchema,
        "geometry": GeometrySchema
    }
}

class N5VolumeServiceReader(VolumeServiceReader):

    def __init__(self, volume_config, config_dir):
        validate(volume_config, N5VolumeSchema)
        
        # Convert path to absolute if necessary (and write back to the config)
        path = volume_config["n5"]["path"]
        if not path.startswith('/'):
            path = os.path.normpath( os.path.join(config_dir, path) )
            volume_config["n5"]["path"] = path

        self._path = path
        self._dataset_name = volume_config["n5"]["dataset-name"]

        self._n5_file = None
        self._n5_datasets = {}
        
        chunk_shape = np.array(self.n5_dataset(0).chunks)
        assert len(chunk_shape) == 3

        # Replace -1's in the message-block-shape with the corresponding chunk_shape dimensions.
        preferred_message_shape_zyx = np.array(volume_config["geometry"]["message-block-shape"][::-1])
        replace_default_entries(preferred_message_shape_zyx, chunk_shape)
        missing_shape_dims = (preferred_message_shape_zyx == -1)
        preferred_message_shape_zyx[missing_shape_dims] = chunk_shape[missing_shape_dims]
        assert not (preferred_message_shape_zyx % chunk_shape).any(), \
            f"Expected message-block-shape ({preferred_message_shape_zyx}) to be a multiple of the chunk shape ({preferred_message_shape_zyx})"

        if chunk_shape[0] == chunk_shape[1] == chunk_shape[2]:
            block_width = int(chunk_shape[0])
        else:
            # The notion of 'block-width' doesn't really make sense if the chunks aren't cubes.
            block_width = -1
        
        auto_bb = np.array([(0,0,0), self.n5_dataset(0).shape])

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        assert (auto_bb[1] >= bounding_box_zyx[1]).all(), \
            f"Volume config bounding box ({bounding_box_zyx}) exceeds the bounding box of the data ({auto_bb})."

        # Replace -1 bounds with auto
        missing_bounds = (bounding_box_zyx == -1)
        bounding_box_zyx[missing_bounds] = auto_bb[missing_bounds]

        # Store members
        self._bounding_box_zyx = bounding_box_zyx
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._block_width = block_width

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

    @property
    def dtype(self):
        return self.n5_dataset(0).dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def block_width(self):
        return self._block_width
    
    @property
    def bounding_box_zyx(self):
        return self._bounding_box_zyx

    def get_subvolume(self, box_zyx, scale=0):
        box_zyx = np.asarray(box_zyx)
        return self.n5_dataset(scale)[box_to_slicing(*box_zyx.tolist())]

    def n5_dataset(self, scale):
        # This member is memoized because that makes it
        # easier to support pickling/unpickling.
        if self._n5_file is None:
            self._n5_file = z5py.File(self._path)
        
        if scale not in self._n5_datasets:
            if scale == 0:
                name = self._dataset_name
            else:
                assert self._dataset_name[-1] == '0', "The N5 dataset does not appear to be a multi-resolution dataset."
                name = self._dataset_name[:-1] + f'{scale}'
            self._n5_datasets[scale] = self._n5_file[name]

        return self._n5_datasets[scale]

    def __getstate__(self):
        """
        Pickle representation.
        """
        d = self.__dict__.copy()
        # Don't attempt to pickle the underlying C++ objects
        # Instead, set them to None so it will be lazily regenerated after unpickling.
        d['_n5_file'] = None
        d['_n5_datasets'] = {}
        return d
