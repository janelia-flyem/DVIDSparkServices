import os

import numpy as np

import z5py

from DVIDSparkServices.util import box_to_slicing

from .volume_service import VolumeServiceReader

class N5VolumeServiceReader(VolumeServiceReader):

    def __init__(self, volume_config, config_dir):
        # Convert path to absolute if necessary (and write back to the config)
        path = volume_config["n5"]["path"]
        if not path.startswith('/'):
            path = os.path.normpath( os.path.join(config_dir, path) )
            volume_config["n5"]["path"] = path

        self._path = path
        self._dataset_name = volume_config["n5"]["dataset-name"]

        self._n5_file = None
        self._n5_dataset = None
        
        chunk_shape = np.array(self.n5_dataset.chunks)
        assert len(chunk_shape) == 3

        # Replace -1's in the message-block-shape with the corresponding chunk_shape dimensions.
        preferred_message_shape_zyx = np.array(volume_config["geometry"]["message-block-shape"][::-1])
        missing_shape_dims = (preferred_message_shape_zyx == -1)
        preferred_message_shape_zyx[missing_shape_dims] = chunk_shape[missing_shape_dims]
        assert not (preferred_message_shape_zyx % chunk_shape).any(), \
            f"Expected message-block-shape ({preferred_message_shape_zyx}) to be a multiple of the chunk shape ({preferred_message_shape_zyx})"

        if chunk_shape[0] == chunk_shape[1] == chunk_shape[2]:
            self._block_width = chunk_shape[0]
        else:
            # The notion of 'block-width' doesn't really make sense if the chunks aren't cubes.
            self._block_width = None
        
        auto_bb = np.array([(0,0,0), self.n5_dataset.shape])

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        assert (auto_bb[1] >= bounding_box_zyx[1]).all(), \
            f"Volume config bounding box ({bounding_box_zyx}) exceeds the bounding box of the data ({auto_bb})."

        # Replace -1 bounds with auto
        missing_bounds = (bounding_box_zyx == -1)
        bounding_box_zyx[missing_bounds] = auto_bb[missing_bounds]

        self._bounding_box_zyx = bounding_box_zyx
        self._preferred_message_shape_zyx = preferred_message_shape_zyx

    @property
    def dtype(self):
        return self._dtype

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
        assert scale == 0, "For now, only scale 0 is supported."
        
        # FIXME: z5py reverses the slicing order (but returns the data in the correct order).
        #return self.n5_dataset[box_to_slicing(*box_zyx.tolist())]
        return self.n5_dataset[box_to_slicing(*box_zyx[:,::-1].tolist())]


    @property
    def n5_dataset(self):
        # This member is memoized because that makes it
        # easier to support pickling/unpickling.
        if self._n5_dataset is None:
            self._n5_file = z5py.File(self._path)
            self._n5_dataset = self._n5_file[self._dataset_name]
        return self._n5_dataset

    def __getstate__(self):
        """
        Pickle representation.
        """
        d = self.__dict__.copy()
        # Don't attempt to pickle the underlying C++ objects
        # Instead, set them to None so it will be lazily regenerated after unpickling.
        d['_n5_file'] = None
        d['_n5_dataset'] = None
        return d
