import os
import lz4

import numpy as np

from jsonschema import validate

from dvidutils import LabelMapper

from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
from DVIDSparkServices.io_util.labelmap_utils import LabelMapSchema, load_labelmap

from . import VolumeServiceWriter


class LabelmappedVolumeService(VolumeServiceWriter):
    """
    Wraps an existing VolumeServiceReader/Writer for label data
    and presents a view of it in which all values are remapped
    according to a given mapping.
    
    (Technically, this is an example of the so-called
    "decorator" GoF pattern.)
    
    Note: This class uses only one mapping. It is valid to apply the same mapping
          for both reading and writing, provided that the mapping is idempotent
          (in which case one of the operations isn't doing any remapping anyway).
          That is, applying the mapping twice is equivalent to applying it only once.
       
          A mapping is idempotent IFF:

              For a mapping from set A to set B, B is a superset of A and all items
              of B map to themselves.
              This is typically true of FlyEM supervoxels (A) and body IDs (B),
              since bodies always contain a supervoxel with matching ID.
    """
    def __init__(self, original_volume_service, labelmap_config, config_dir):
        self.original_volume_service = original_volume_service
        validate(labelmap_config, LabelMapSchema)

        # Convert relative path to absolute
        if not labelmap_config["file"].startswith('gs://') and not labelmap_config["file"].startswith("/"):
            abspath = os.path.normpath( os.path.join(config_dir, labelmap_config["file"]) )
            labelmap_config["file"] = abspath
        
        self.labelmap_config = labelmap_config
        self.config_dir = config_dir
        
        # These are computed on-demand and memoized for the sake of pickling support.
        # See __getstate__()
        self._mapper = None
        self._mapping_pairs = None
        self._compressed_mapping_pairs = None
        
        assert np.issubdtype(self.dtype, np.integer)
        
        self.apply_when_reading = labelmap_config["apply-when"] in ("reading", "reading-and-writing")
        self.apply_when_writing = labelmap_config["apply-when"] in ("writing", "reading-and-writing")

    def __getstate__(self):
        if self._compressed_mapping_pairs is None:
            # Load the labelmapping and then compress 
            mapping_pairs = self.mapping_pairs
            self._compressed_mapping_pairs = CompressedNumpyArray(mapping_pairs)

        d = self.__dict__.copy()
        
        # Discard mapping pairs (will be reconstructed from compressed)
        d['_mapping_pairs'] = None
        
        # Discard mapper. It isn't pickleable
        d['_mapper'] = None
        return d

    @property
    def mapping_pairs(self):
        if self._mapping_pairs is None:
            if self._compressed_mapping_pairs is not None:
                self._mapping_pairs = self._compressed_mapping_pairs.deserialize()
            else:
                self._mapping_pairs = load_labelmap(self.labelmap_config, self.config_dir)
                
                # Save RAM by converting to uint32 if possible (usually possible)
                if self._mapping_pairs.max() <= np.iinfo(np.uint32).max:
                    self._mapping_pairs = self._mapping_pairs.astype(np.uint32)
        return self._mapping_pairs

    @property
    def mapper(self):
        if not self._mapper:
            domain, codomain = self.mapping_pairs.transpose()
            self._mapper = LabelMapper(domain, codomain)
        return self._mapper

    @property
    def base_service(self):
        return self.original_volume_service.base_service

    @property
    def dtype(self):
        return self.original_volume_service.dtype

    @property
    def block_width(self):
        return self.original_volume_service.block_width

    @property
    def preferred_message_shape(self):
        return self.original_volume_service.preferred_message_shape

    @property
    def bounding_box_zyx(self):
        return self.original_volume_service.bounding_box_zyx

    @property
    def available_scales(self):
        return self.original_volume_service.available_scales

    def get_subvolume(self, box_zyx, scale=0):
        volume = self.original_volume_service.get_subvolume(box_zyx, scale)

        if self.apply_when_reading:
            # TODO: Apparently LabelMapper can't handle non-contiguous arrays right now.
            #       (It yields incorrect results)
            #       Check to see if this is still a problem in the latest version of xtensor-python.
            volume = np.asarray(volume, order='C')
            self.mapper.apply_inplace(volume, allow_unmapped=True)

        return volume

    def write_subvolume(self, subvolume, offset_zyx, scale):
        if self.apply_when_writing:
            # Copy first to avoid remapping user's input volume
            # (which they might want to reuse)
            subvolume = subvolume.copy(order='C')
            self.mapper.apply_inplace(subvolume, allow_unmapped=True)        

        self.original_volume_service.write_subvolume(subvolume, offset_zyx, scale)
