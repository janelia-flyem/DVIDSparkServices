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
    "default": [[-1,-1,-1], [-1,-1,-1]]
}

GeometrySchema = \
{
    "description": "Describes a volume's geometric and access pattern properties",
    "type": "object",
    "default": {},
    "properties": {
        "bounding-box": BoundingBoxSchema,

        "message-block-shape": {
            "description": "The preferred access pattern block shape.\n"
                           "A value of -1 for any axis means 'auto'",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
            "default": [-1,-1,-1]
        },

        "block-width": {
            "description": "The block size of the underlying volume storage, if applicable.",
            "type": "integer",
            "default": -1
        },
        
        "available-scales": {
            "description": "The list of available scales for the volume source.",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 1,
            "maxItems": 10,
            "default": [0]
        }
    }
}
