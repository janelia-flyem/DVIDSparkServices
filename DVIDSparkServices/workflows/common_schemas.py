from DVIDSparkServices.json_util import flow_style

# Terms:
# - Service: Arbitrary source (or sink) of voxels, with no defined bounding box or access pattern
# - Geometry: Bounding box, access pattern, scale
# - Volume: Combines both Service with a Geometry (and possibly other properties, such as a labelmap)

BoundingBoxSchema = \
{
    "description": "The bounding box [[x0,y0,z0],[x1,y1,z1]], \n"
                   "where [x1,y1,z1] == maxcoord+1 (i.e. Python conventions)",
    "type": "array",
    "minItems": 2,
    "maxItems": 2,
    "items": {
        "type": "array",
        "items": { "type": "integer" },
        "minItems": 3,
        "maxItems": 3
    },
    "default": flow_style( [[-1,-1,-1], [-1,-1,-1]] )
}

GeometrySchema = \
{
    "description": "Describes a volume's geometric and access pattern properties",
    "type": "object",
    "default": {},
    "properties": {
        "scale": {
            "description": "The pyramid scale at which the data should be accessed: \n"
                           "0: full-res; \n"
                           "1: downsample-by-2; \n"
                           "2: downsample-by-4; etc. \n",
            "type": "integer",
            "minimum": 0,
            "default": 0
        },

        "bounding-box": BoundingBoxSchema,

        "message-block-shape": {
            "description": "The preferred access pattern block shape.\n"
                           "A value of -1 for any axis means 'auto'",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
            "default": flow_style( [6400,64,64] )
        },

        "block-width": {
            "description": "The block size of the underlying volume storage, if applicable.",
            "type": "integer",
            "default": 64
        } 
    }
}

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
            "default": flow_style( [0,0] ),
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

DvidServiceSchema = \
{
    "description": "Parameters specify a DVID node",
    "type": "object",
    "required": ["server", "uuid"],

    "default": {},
    "properties": {
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

DvidGrayscaleServiceSchema = \
{
    "description": "Parameters specify a source of grayscale data from DVID",
    "type": "object",

    "allOf": [DvidServiceSchema],

    "required": DvidServiceSchema["required"] + ["grayscale-name"],
    "default": {},
    "properties": {
        "grayscale-name": {
            "description": "The grayscale instance to read/write from/to.\n"
                           "Instance must be grayscale (uint8blk).",
            "type": "string",
            "minLength": 1
        }
    }
}

GrayscaleVolumeSchema = \
{
    "description": "Describes a grayscale volume (service and geometry)",
    "type": "object",
    "default": {},
    "oneOf": [
        { "properties": { "dvid": DvidGrayscaleServiceSchema } },
        { "properties": { "slice-files": SliceFilesServiceSchema } }
    ],
    "properties": {
        "geometry": GeometrySchema
    }
}

DvidSegmentationServiceSchema = \
{
    "description": "Parameters specify a source of segmentation data from DVID",
    "type": "object",

    "allOf": [DvidServiceSchema],

    "required": DvidServiceSchema["required"] + ["segmentation-name"],
    "default": {},
    "properties": {
        "segmentation-name": {
            "description": "The labels instance to read/write from. \n"
                           "Instance may be either googlevoxels, labelblk, or labelarray.",
            "type": "string",
            "minLength": 1
        }
    }
}

BrainMapsSegmentationServiceSchema = \
{
    "description": "Parameters to use Google BrainMaps as a source of voxel data",
    "type": "object",
    "required": ["project", "dataset", "volume-id", "change-stack-id"],
    "default": {},
    "properties": {
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
            "description": "Change Stack ID. Specifies a set of changes to apple on top of the volume\n"
                           "(e.g. a set of agglomeration steps).",
            "type": "string",
            "default": ""
        }
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
    "description": "Describes a segmentation volume source (or destination), \n"
                   "extents, and preferred access pattern",
    "type": "object",
    "required": ["source", "geometry"],
    "default": {},
    "oneOf": [
        { "properties": { "dvid": DvidSegmentationServiceSchema } },
        { "properties": { "brainmaps": BrainMapsSegmentationServiceSchema } }
    ],
    "properties": {
        "apply-labelmap": LabelMapSchema,
        "geometry": GeometrySchema
    }
}

SegmentationVolumeListSchema = \
{
    "description": "A list of segmentation volume sources (or destinations).",
    "type": "array",
    "items": SegmentationVolumeSchema,
    "minItems": 1,
    "default": [{}] # One item by default (will be filled in during yaml dump)
}
