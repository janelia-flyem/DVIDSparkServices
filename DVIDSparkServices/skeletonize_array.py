from __future__ import print_function, absolute_import
import ctypes

import numpy as np
from neutube import ZStack, ZStackSkeletonizer, ZSwcTree

BYTE_PTR = ctypes.POINTER(ctypes.c_byte)

class ZStackKind(object):
    # Constants from neurolabi/lib/genelib/src/image_lib.h
    GREY      = 1 # 1-byte grey-level image or stack
    GREY16    = 2 # 2-byte grey-level image or stack
    COLOR     = 3 # 3-byte RGB image or stack
    FLOAT32   = 4 # 4-byte floating values

    # Kind -> dtype
    ToDtype = { GREY    : np.uint8,
                GREY16  : np.uint16,
                #COLOR  : np.uint8, # What is correct here?
                FLOAT32 : np.float32, }

    # dtype -> Kind
    FromDtype = { v:k for k,v in ToDtype.items() }
    FromDtype.update( { np.dtype(v):k for k,v in ToDtype.items() } )


def view_zstack_as_ndarray(zstack):
    """
    Returns a C-order view of the given ZStack, with axes order (c,z,y,x).
    """
    assert isinstance(zstack, ZStack)
    if zstack.kind() == ZStackKind.COLOR:
        # At the moment, I don't understand how a 'color' image can have only 1 channel,
        # and how that should map to numpy arrays.
        # And in general, I have no need for RGB numpy arrays anyway...
        raise NotImplementedError( "Don't know how to view color ZStacks as numpy arrays." )
    
    shape_czyx = (zstack.channelNumber(), zstack.depth(), zstack.height(), zstack.width())
    dtype = ZStackKind.ToDtype[ zstack.kind() ]
    dtype_bytes = np.dtype(dtype).itemsize
    total_stack_bytes = np.prod(shape_czyx) * dtype_bytes
    
    # Start by viewing the raw data as a flat uint8 array
    swig_ptr = zstack.array8()
    ctypes_ptr = ctypes.cast(int(swig_ptr), BYTE_PTR)
    flat_bytes = np.ctypeslib.as_array(ctypes_ptr, shape=(total_stack_bytes,))

    # Now cast and reshape
    view_czyx = flat_bytes.view(dtype).reshape(shape_czyx)
    return view_czyx


def copy_ndarray_to_zstack(data_zyx):
    assert data_zyx.ndim in (3,4), \
        "Bad shape: {}. Must be either zyx or czyx".format(data_zyx.shape)

    if data_zyx.ndim == 4:
        data_czyx = data_zyx
    elif data_zyx.ndim == 3:
        data_czyx = data_zyx[None,...]

    kind = ZStackKind.FromDtype[data_czyx.dtype]
    C, Z, Y, X = data_czyx.shape

    zstack = ZStack( kind, X, Y, Z, C )
    view = view_zstack_as_ndarray( zstack )
    view[:] = data_czyx
    return zstack

# These default values match the ones Ting uses for his Skeletonizer service,
# not the defaults in the C++ constructor.
DefaultConfig = \
{
    "downsampleInterval": [0,0,0],
    "minimalLength": 40.0,
    "maximalDistance": 100,
    "keepingSingleObject": True,
    "rebase": True,
    "fillingHole": True,
    "minimalObjectSize": 0
}

SkeletonConfigSchema = \
{
  #"$schema": "http://json-schema.org/schema#",
  #"title": "Skeletonization config file",
  "type": "object",
  "default": {},
  "properties": {
    "downsampleInterval": {
      "type": "array",
      "items": { "type": "integer" },
      "minItems": 3,
      "maxItems": 3
    },
    "minimalLength": {
      "description": "",
      "type": "number"
    },
    "maximalDistance": {
      "description": "",
      "type": "number"
    },
    "keepingSingleObject": {
      "description": "",
      "type": "boolean"
    },
    "rebase": {
      "description": "",
      "type": "boolean"
    },
    "fillingHole": {
      "description": "",
      "type": "boolean"
    },
    "minimalObjectSize": {
      "description": "",
      "type": "integer"
    }    
  }
}

# Augment the schema with defaults.
for key, default in DefaultConfig.items():
    SkeletonConfigSchema["properties"][key]["default"] = default

def make_skeletonizer(config={}):
    try:
        full_config = config.copy()
        from DVIDSparkServices.json_util import validate_and_inject_defaults
        # Validate and replace missing values with their defaults
        validate_and_inject_defaults(full_config, SkeletonConfigSchema)
    except ImportError:
        # Skip schema validation
        full_config = DefaultConfig.copy()
        full_config.update(config)

    # Configure
    skeletonizer = ZStackSkeletonizer()
    skeletonizer.setDownsampleInterval( *full_config['downsampleInterval'] )
    skeletonizer.setLengthThreshold( full_config['minimalLength'] )
    skeletonizer.setDistanceThreshold( full_config['maximalDistance'] )
    skeletonizer.setMinObjSize( full_config['minimalObjectSize'] )
    skeletonizer.setKeepingSingleObject( full_config['keepingSingleObject'] )
    skeletonizer.setRebase( full_config['rebase'] )

    return skeletonizer

def skeletonize_array(binary_zyx, config={}):
    assert binary_zyx.ndim == 3
    binary_zyx = binary_zyx.astype(np.uint8, copy=False)
    skeletonizer = make_skeletonizer(config)
    skeletonizer._print()
    
    zstack = copy_ndarray_to_zstack(binary_zyx)
    tree = skeletonizer.makeSkeleton(zstack)
    return tree

if __name__ == "__main__":
    # Create a test object (shaped like an 'X')
    from scipy.ndimage import distance_transform_edt
    center_line_img = np.zeros((100,100,100), dtype=np.uint32)
    for i in range(100):
        center_line_img[i, i, i] = 1
        center_line_img[99-i, i, i] = 1
    
    # Scipy distance_transform_edt conventions are opposite of vigra:
    # it calculates distances of non-zero pixels to the zero pixels.
    center_line_img = 1 - center_line_img
    distance_to_line = distance_transform_edt(center_line_img)
    binary_vol = (distance_to_line <= 10).astype(np.uint8)

    # Generate a skeleton
    tree = skeletonize_array(binary_vol)
    assert isinstance(tree, ZSwcTree)

    # Translate and scale the skeleton
    tree.translate(1000,1000,1000) # X,Y,Z
    tree.rescale(2,2,2) # X,Y,Z

    # Save it, print it.
    tree.save('/tmp/test-skeleton.swc')

    print("...................")
    print(tree.toString())
    print("...................")
    print("DONE.")
