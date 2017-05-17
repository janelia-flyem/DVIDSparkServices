"""This moduel defines routines to query DVID meta data.
"""

import json
import requests
from enum import Enum
import numpy as np

from libdvid import DVIDNodeService
from libdvid import ConnectionMethod
from libdvid import DVIDConnection
from libdvid._dvid_python import DVIDException

"""Defines label and raw array types currently supported.

Format: "typename" : (is_label, numpy type)
"""

supportedArrayTypes = {"uint8blk": (False, np.uint8), "labelblk": (True, np.uint64)}

class Compression(Enum):
    """Defines compression types supported by Google.
    """

    DEFAULT = None
    LZ4 = "lz4"
    JPEG = "jpeg"
    PNG = "png"
    
def is_dvidversion(dvid_server, uuid):
    """Checks if uuid and dvid server exists.

    Args:
        dvid_server (str): location of dvid server
        uuid (str): version id
    """
    try:
        ns = DVIDNodeService(str(dvid_server), str(uuid))
    except DVIDException:
        # returns exception if it does not exist
        return False
    return True

def is_datainstance(dvid_server, uuid, name):
    """Checks if datainstance name exists.

    Args:
        dvid_server (str): location of dvid server
        uuid (str): version id
        name (str): data instance name
    """
    try:
        ns = DVIDNodeService(str(dvid_server), str(uuid))
        info = ns.get_typeinfo(name)
    except DVIDException:
        # returns exception if it does not exist
        return False
    return True

def create_labelarray(dvid_server, uuid, name, blocksize=(64,64,64),
                      compression=Compression.DEFAULT ):
    """Create 64 bit labels data structure.

    Note:
        Currenly using labelblk.  Does not check whether
        the type already exists.  DVIDExceptions are uncaught.
        libdvid-cpp can be used directly but this also supports
        setting the compression type.

    Args:
        dvid_server (str): location of dvid server
        uuid (str): version id
        name (str): data instance name
        blocksize (3 int tuple): block size z,y,x
        compression (Compression enum): compression to be used
        minimal_extents: box [(z0,y0,x0), (z1,y1,x1)].
                        If provided, data extents will be at least this large (possibly larger).
    
    Raises:
        DVIDExceptions are not caught in this function and will be
        transferred to the caller.
    """
    
    conn = DVIDConnection(dvid_server) 

    endpoint = "/repo/" + uuid + "/instance"
    blockstr = "%d,%d,%d" % (blocksize[2], blocksize[1], blocksize[0])
    data = {"typename": "labelblk", "dataname": name, "BlockSize": blockstr}
    if compression != Compression.DEFAULT:
        data["Compression"] = compression.value

    conn.make_request(endpoint, ConnectionMethod.POST, json.dumps(data))

def create_rawarray8(dvid_server, uuid, name, blocksize=(64,64,64),
                     compression=Compression.DEFAULT ):
    """Create 8 bit labels data structure.

    Note:
        Currenly using uint8blk only.  Does not check whether
        the type already exists.  DVIDExceptions are uncaught.
        libdvid-cpp can be used directly but this also supports
        setting the compression type.

    Args:
        dvid_server (str): location of dvid server
        uuid (str): version id
        name (str): data instance name
        blocksize (3 int tuple): block size z,y,x
        compression (Compression enum): compression to be used
        minimal_extents: box [(z0,y0,x0), (z1,y1,x1)].
                        If provided, data extents will be at least this large (possibly larger).
    
    Raises:
        DVIDExceptions are not caught in this function and will be
        transferred to the caller.
    """
    
    conn = DVIDConnection(dvid_server) 

    endpoint = "/repo/" + uuid + "/instance"
    blockstr = "%d,%d,%d" % (blocksize[2], blocksize[1], blocksize[0])
    data = {"typename": "uint8blk", "dataname": name, "BlockSize": blockstr}
    if compression != Compression.DEFAULT:
        data["Compression"] = compression.value

    conn.make_request(endpoint, ConnectionMethod.POST, json.dumps(data))
    

def update_extents(dvid_server, uuid, name, minimal_extents):
    """
    Ensure that the given data instance has at least the given extents.
    
    Args:
        dvid_server (str): location of dvid server
        uuid (str): version id
        name (str): data instance name
        minimal_extents: 3D bounding box [(z0,y0,x0), (z1,y1,x1)].
                         If provided, data extents will be at least this large (possibly larger).
    """    
    new_extents = np.array(minimal_extents, dtype=int)
    assert new_extents.shape == (2,3), \
        "Minimal extents must be provided as a 3D bounding box: [(z0,y0,x0), (z1,y1,x1)]"
    
    # Fetch original extents.
    r = requests.get('{dvid_server}/api/node/{uuid}/{name}/info'.format(**locals()))
    r.raise_for_status()

    info = r.json()
    
    if info["Extended"]["MinPoint"] is not None:
        new_extents = np.minimum(new_extents, info["Extended"]["MinPoint"])

    if info["Extended"]["MaxPoint"] is not None:
        new_extents = np.maximum(new_extents, info["Extended"]["MaxPoint"])

    if (new_extents != minimal_extents).any():
        new_extents = new_extents.tolist()
        extents_json = { "MinPoint": new_extents[0],
                         "MaxPoint": new_extents[1] }

        r = requests.post( '{dvid_server}/api/node/{uuid}/{name}/extents'.format(**locals()),
                           json=extents_json )
        r.raise_for_status()

def extend_list_value(dvid_server, uuid, kv_instance, key, new_list):
    """
    For the list stored at the given keyvalue instance and key, extend it with the given new_list.
    If the keyvalue instance and/or key are missing from the server, create them.
    """
    assert isinstance(new_list, list)
    old_list = []

    r = requests.get('{dvid_server}/api/node/{uuid}/{kv_instance}/keys'.format(**locals()))
    if r.status_code not in (200,400):
        r.raise_for_status()
    
    if r.status_code == 400:
        # Create the keyvalue instance first
        r_post = requests.post('{dvid_server}/api/repo/{uuid}/instance'.format(**locals()),
                               json={ "typename": "keyvalue", 
                                      "dataname": kv_instance } )
        r_post.raise_for_status()

    elif key in r.json():
        # Fetch original value
        r = requests.get('{dvid_server}/api/node/{uuid}/{kv_instance}/key/{key}'.format(**locals()))
        r.raise_for_status()
        old_list = r.json()
        assert isinstance(old_list, list)

    new_list = list(set(old_list + new_list))
    
    r = requests.post('{dvid_server}/api/node/{uuid}/{kv_instance}/key/{key}'.format(**locals()),
                      json=new_list)
    r.raise_for_status()
    

def get_blocksize(dvid_server, uuid, dataname):
    """Gets block size for supplied data name.

    Note:
        Does not check for existence of body and whether
        it is an array type.  The block size is always
        assumes to be isotropic.
   
    Args:
        dvid_server (str): location of dvid server
        uuid (str): version id
        dataname (str): data instance
   
    Returns:
        (z,y,x) blocksize.

    Raises:
        DVIDExceptions are not caught in this function and will be
        transferred to the caller.
    """

    ns = DVIDNodeService(str(dvid_server), str(uuid))
    info = ns.get_typeinfo(dataname)
    x,y,z = info["Extended"]["BlockSize"] # DVID ordered x,y,z
    return (z,y,x)

def set_sync(dvid_server, uuid, srcname, destname):
    """Sets a sync on srcname to point to destname.
    
    Note: only a limited number of syncs are possible.
    libdvid-cpp will throw an error if a sync is not possible.

    Args:
        dvid_server (str): location of dvid server
        uuid (str): version id
        srcname (str): data instance with new sync
        destname (str): data instance pointed to by new sync 
    
    Raises:
        DVIDExceptions are not caught in this function and will be
        transferred to the caller.
    """

    ns = DVIDNodeService(str(dvid_server), str(uuid))
    data = {"sync": destname}
    ns.custom_request(srcname + "/sync", json.dumps(data), ConnectionMethod.POST)

def has_sync(dvid_server, uuid, srcname, destname):
    """Checks whether srcname is synced (listen to changes) on destname.
    
    Args:
        dvid_server (str): location of dvid server
        uuid (str): version id
        srcname (str): data instance with the potential sync
        destname (str): data instance pointed to by the sync
    
    Returns:
        A boolean of value True if the sync exists.

    Raises:
        DVIDExceptions are not caught in this function and will be
        transferred to the caller.
    """

    ns = DVIDNodeService(str(dvid_server), str(uuid))
    info = ns.get_typeinfo(srcname)
    sync_data = info["Base"]["Syncs"]
    return destname in sync_data


class dataInstance(object):
    """Container for DVID data instance meta information.

    Note:
        The instance and server must exist or a ValueError will be thrown.
    """

    def __init__(self, dvidserver, uuid, dataname):
        """Initialization.

        Args:
            dvidserver (string): address for dvid server
            uuid (string): uuid for DVID instance 
            dataname (string): name of data instance
        """

        self.server = dvidserver
        self.uuid = uuid
        self.name = dataname

        # check DVID existence and get meta
        try:  
            node_service = DVIDNodeService(str(dvidserver), str(uuid))
            self.info = node_service.get_typeinfo(dataname)
        except DVIDException:
            raise ValueError("Instance not available")        
        self.datatype = str(self.info["Base"]["TypeName"])

    def is_array(self):
        """Checks if data instance is a raw or label array.
        """

        if self.datatype in supportedArrayTypes:
            return True
        return False

    def is_labels(self):
        """Checks if data instance is label array type.
        """
        
        if self.datatype not in supportedArrayTypes:
            return False
        typeinfo = supportedArrayTypes[self.datatype]
        return typeinfo[0]


