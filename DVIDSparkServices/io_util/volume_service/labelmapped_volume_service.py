import os

import numpy as np

from dvidutils import LabelMapper

from DVIDSparkServices.io_util.labelmap_utils import load_labelmap
from . import VolumeServiceWriter

LabelMapSchema = \
{
    "description": "A label mapping file to apply to segmentation after reading or before writing.",
    "type": "object",
    "default": {},
    "properties": {
        "file": {
            "description": "Path to a file of labelmap data",
            "type": "string" ,
            "default": ""
        },
        "file-type": {
            "type": "string",
            "enum": ["label-to-body",       # CSV file containing the direct mapping.  Rows are orig,new 
                     "equivalence-edges",   # CSV file containing a list of label merges.
                                            # (A label-to-body mapping is derived from this, via connected components analysis.)
                     "__invalid__"],
            "default": "__invalid__"
        },
        "apply-when": {
            "type": "string",
            "enum": ["reading", "writing", "reading-and-writing"],
            "default": "reading-and-writing"
        }
    }
}

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
        
        # Convert relative path to absolute
        if not labelmap_config["file"].startswith('gs://') and not labelmap_config["file"].startswith("/"):
            abspath = os.path.normpath( os.path.join(config_dir, labelmap_config["file"]) )
            labelmap_config["file"] = abspath
        
        self.mapping_pairs = load_labelmap(labelmap_config, config_dir)
        
        # This is computed on-demand and memoized for the sake of pickling support
        self._mapper = None
        
        assert np.issubdtype(self.dtype, np.integer)
        
        self.apply_when_reading = labelmap_config["apply-when"] in ("reading", "reading-and-writing")
        self.apply_when_writing = labelmap_config["apply-when"] in ("writing", "reading-and-writing")

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
