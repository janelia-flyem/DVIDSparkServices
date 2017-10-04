DvidNodeSchema = \
{
    "description": "Parameters specify a DVID node",
    "type": "object",
    "required": ["service-type", "server", "uuid"],
    "properties": {
        "service-type": { "type": "string",
                          "enum": ["dvid"] },
        "server": {
            "description": "location of DVID server to READ.  Either IP:PORT or the special word 'node-local'.",
            "type": "string",
        },
        "uuid": {
            "description": "version node for READING segmentation",
            "type": "string"
        }
    }
}

DvidSegmentationSourceSchema = \
{
    "description": "Parameters specify a DVID node",
    "type": "object",

    "allOf": [
        DvidNodeSchema
    ],

    "required": ["segmentation-name"],
    "properties": {
        "segmentation-name": {
            "description": "The labels instance to READ from. Instance may be either googlevoxels, labelblk, or labelarray.",
            "type": "string",
            "minLength": 1
        }
    }
}

BrainMapsSegmentationSourceSchema = \
{
    "description": "Parameters to use Google BrainMaps as a source of voxel data",
    "type": "object",
    "required": ["service-type", "project", "dataset", "volume-id", "change-stack-id"],
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

SegmentationVolumeSchema = \
{
    "description": "Describes a segmentation volume source, extents, and preferred access pattern",
    "type": "object",
    "required": ["bounding-box", "message-block-shape"],
    "oneOf": [
        DvidSegmentationSourceSchema,
        BrainMapsSegmentationSourceSchema
    ],
    "properties": {
        "bounding-box": BoundingBoxSchema,
        "message-block-shape": {
            "description": "The block shape (XYZ) for the initial tasks that fetch segmentation from DVID.",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
            "default": [6400,64,64],
        },
        "block-width": {
            "description": "The block size of the underlying volume storage.",
            "type": "integer",
            "default": 64
        }
    }
}
