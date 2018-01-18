from .. import ( DvidGrayscaleServiceSchema, SliceFilesServiceSchema, N5ServiceSchema, 
                 DvidSegmentationServiceSchema, BrainMapsSegmentationServiceSchema, 
                 NewAxisOrderSchema, RescaleLevelSchema )

from .geometry import GeometrySchema

# TERMINOLOGY:
#
# - Service: Arbitrary source (or sink) of voxels, with no defined bounding box or access pattern
# - Geometry: Bounding box, access pattern, scale
# - Volume: Combines both Service with a Geometry (and possibly other properties, such as a labelmap)


#
# Generic grayscale volume (one service + geometry)
#
GrayscaleVolumeSchema = \
{
    "description": "Describes a grayscale volume (service and geometry)",
    "type": "object",
    "default": {},
    "oneOf": [
        { "properties": { "dvid": DvidGrayscaleServiceSchema } },
        { "properties": { "slice-files": SliceFilesServiceSchema } },
        { "properties": { "n5": N5ServiceSchema } }
    ],
    "properties": {
        "geometry": GeometrySchema,
        "transpose-axes": NewAxisOrderSchema,
        "rescale-level": RescaleLevelSchema
    }
}


LabelMapSchema = \
{
    "description": "A label mapping file to apply to segmentation after reading or before writing.",
    "type": "object",
    "required": ["file", "file-type"],
    "default": {"file": "", "file-type": "__invalid__"},
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
                     "__invalid__"]
        }
    }
}


#
# Generic segmentation volume (one service + geometry)
#
SegmentationVolumeSchema = \
{
    "description": "Describes a segmentation volume source (or destination), \n"
                   "extents, and preferred access pattern",
    "type": "object",
    "required": ["geometry"],
    "default": {},
    "oneOf": [
        { "properties": { "dvid": DvidSegmentationServiceSchema }, "required": ["dvid"] },
        { "properties": { "brainmaps": BrainMapsSegmentationServiceSchema }, "required": ["brainmaps"] }
    ],
    "properties": {
        "apply-labelmap": LabelMapSchema,
        "geometry": GeometrySchema,
        "transpose-axes": NewAxisOrderSchema,
        "rescale-level": RescaleLevelSchema
    }
}


#
# List of segmentation volumes
#
SegmentationVolumeListSchema = \
{
    "description": "A list of segmentation volume sources (or destinations).",
    "type": "array",
    "items": SegmentationVolumeSchema,
    "minItems": 1,
    "default": [{}] # One item by default (will be filled in during yaml dump)
}
