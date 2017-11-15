# Terms:
# - Source: Arbitrary source of voxels, with no defined bounding box or access pattern
# - Geometry: Bounding box, access pattern
# - Volume: Combines both Source and Geometry

BoundingBoxSchema = \
{
    "description": "The bounding box [[x0,y0,z0],[x1,y1,z1]], "
                   "where [x1,y1,z1] == maxcoord+1 (i.e. Python conventions)",
    "type": "array",
    "minItems": 2,
    "maxItems": 2,
    "items": {
        "type": "array",
        "items": { "type": "integer" },
        "minItems": 3,
        "maxItems": 3
    }
}

GeometrySchema = \
{
    "description": "Describes a volume's geometric and access pattern properties",
    "type": "object",
    "default": {},
    "properties": {
        "scale": {
            "description": "The pyramid scale at which the data should be accessed: "
                           "0: full-res; "
                           "1: downsample-by-2; "
                           "2: downsample-by-4; etc.",
            "type": "integer",
            "minimum": 0,
            "default": 0
        },

        "bounding-box": BoundingBoxSchema,

        "message-block-shape": {
            "description": "The preferred access pattern block shape. A value of -1 for any axis means 'auto'",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
            "default": [6400,64,64],
        },

        "block-width": {
            "description": "The block size of the underlying volume storage, if applicable.",
            "type": "integer",
            "default": 64
        } 
    }
}

SourceBaseSchema = \
{
    "description": "Base fields for source schemas",
    "type": "object",
    "required": ["service-type", "semantic-type"],
    "default": {},
    "properties": {
        "service-type": { "type": "string", "enum": ["dvid", "brainmaps", "slice-files"] },
        "semantic-type": { "type": "string", "enum": ["grayscale", "segmentation"] }
    }    
}

SliceFilesSourceSchema = \
{
    "description": "Parameters specify a source of grayscale data from image slices.",
    "type": "object",

    "required": ["slice-path-format"],

    "default": {},
    "properties": {
        "service-type": { "type": "string", "enum": ["slice-files"], "default": "slice-files" },
        "semantic-type": { "type": "string", "enum": ["grayscale"], "default": "grayscale" },
        
        "slice-path-format": {
            "description": 'String format for image Z-slice paths, using python format-string syntax, '
                           'e.g. "/path/to/slice{:05}.png" '
                           'Some workflows may also support the prefix "gs://" for gbucket data.',
            "type": "string",
            "minLength": 1
        },
        "slice-xy-offset": {
            "description": "The XY-offset indicating where the slices reside within the global input coordinates, "
                           "That is, which global (X,Y) coordinate does pixel (0,0) of each slice correspond to? "
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
        "source": SliceFilesSourceSchema,
        "geometry": GeometrySchema
    }
}

DvidSourceBaseSchema = \
{
    "description": "Parameters specify a DVID node",
    "type": "object",
    "required": ["service-type", "server", "uuid"],

    "allOf": [SourceBaseSchema],

    "default": {},
    "properties": {
        "service-type": { "type": "string", "enum": ["dvid"] },

        "server": {
            "description": "location of DVID server to READ.",
            "type": "string",
        },
        "uuid": {
            "description": "version node for READING segmentation",
            "type": "string"
        }
    }
}

DvidGrayscaleSourceSchema = \
{
    "description": "Parameters specify a source of grayscale data from DVID",
    "type": "object",

    "allOf": [DvidSourceBaseSchema],

    "required": ["grayscale-name"],
    "default": {},
    "properties": {
        "grayscale-name": {
            "description": "The grayscale instance to read/write from/to. Instance must be grayscale (uint8blk).",
            "type": "string",
            "minLength": 1
        }
    }
}

DvidSegmentationSourceSchema = \
{
    "description": "Parameters specify a source of segmentation data from DVID",
    "type": "object",

    "allOf": [DvidSourceBaseSchema],

    "required": ["segmentation-name"],
    "default": {},
    "properties": {
        "segmentation-name": {
            "description": "The labels instance to read/write from. Instance may be either googlevoxels, labelblk, or labelarray.",
            "type": "string",
            "minLength": 1
        }
    }
}

GrayscaleVolumeSchema = \
{
    "description": "Describes a segmentation volume source (or destination), extents, and preferred access pattern",
    "type": "object",
    "required": ["source", "geometry"],
    "default": {},
    "properties": {
        "source": {
            "type": "object",
            "semantic-type": { "enum": ["grayscale"] }, # Force as grayscale
            "oneOf": [ DvidGrayscaleSourceSchema,
                       SliceFilesSourceSchema ],
            "default": {}
        },
        "geometry": GeometrySchema
    }
}


BrainMapsSegmentationSourceSchema = \
{
    "description": "Parameters to use Google BrainMaps as a source of voxel data",
    "type": "object",
    "required": ["service-type", "project", "dataset", "volume-id", "change-stack-id"],
    "default": {},
    "properties": {
        "service-type": { "type": "string",
                          "enum": ["brainmaps"] },
        "project": {
            "description": "Project ID",
            "type": "string",
        },
        "dataset": {
            "description": "Dataset identifier",
            "type": "string"
        },
        "volume-id": {
            "description": "Volume ID",
            "type": "string"
        },
        "change-stack-id": {
            "description": "Change Stack ID, specifies a set of changes to apple on top of the volume, (e.g. a set of agglomeration steps).",
            "type": "string",
            "default": ""
        }
    }
}

SkippedSegmentationSourceSchema = \
{
    "description": "A quick way to specify that a given segmentation source/destination should be SKIPPED.",
    "type": "object",
    "required": ["service-type"],
    "additionalProperties": True,
    "default": {},
    "properties": {
        "service-type": { "type": "string", "enum": ["SKIP"] }
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

SegmentationVolumeSchema = \
{
    "description": "Describes a segmentation volume source (or destination), extents, and preferred access pattern",
    "type": "object",
    "required": ["source", "geometry"],
    "default": {},
    "properties": {
        "source": {
            "type": "object",
            "oneOf": [ DvidSegmentationSourceSchema,
                       BrainMapsSegmentationSourceSchema,
                       SkippedSegmentationSourceSchema ],
            
            "default": {},
            "properties": {
                "semantic-type": { "enum": ["segmentation"], "default": "segmentation" }, # force as segmentation
                # Add extra field for labelmap
                "apply-labelmap": LabelMapSchema
            }
        },
        "geometry": GeometrySchema
    }
}

SegmentationVolumeListSchema = \
{
    "description": "A list of segmentation volume sources (or destinations).",
    "type": "array",
    "items": SegmentationVolumeSchema,
    "minItems": 1
}
