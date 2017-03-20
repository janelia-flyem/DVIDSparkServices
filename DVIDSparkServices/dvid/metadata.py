"""This moduel defines routines to query DVID meta data.
"""

import numpy as np
from libdvid import DVIDNodeService
from libdvid._dvid_python import DVIDException

"""Defines label and raw array types currently supported.

Format: "typename" : (is_label, numpy type)
"""

supportedArrayTypes = {"uint8blk": (False, np.uint8), "labelblk": (True, np.uint64)}

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
            self.node_service = DVIDNodeService(str(dvidserver), str(uuid))
            self.info = self.node_service.get_typeinfo(dataname)
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


