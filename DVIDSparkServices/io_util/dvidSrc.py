"""This module defines routines to load 3D data from DVID.
"""

from volumeSrc import volumeSrc
import partitionSchema
import numpy as np
from DVIDSparkServices.dvid.metadata import DataInstance
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service

class dvidSrc(volumeSrc):
    """Iterator class provides a read interface to DVID raw and label arrays.

    This function can load data in parallel with Spark our as a single-threaded
    library.  The user must specify a DVID address and datatype name.
    A description of how to partition the data is the only way to call
    this interface currently.  The datatype size is determined by the instance.

    If the server or data instance does not exist or is of the wrong type,
    a ValueError is thrown.  Only uint8blk and labelblk is supported currently.
    The resource server should be used if there are many DVID readers and DVID does
    not have a distributed backend.

    TODO:
        Currently, only supports masking pre-existing partitions.
        This src should input defined by ROI and bounding box.  partitionSchema
        will then probably have to be specified.

        Support all array instance types.

        Support image tiles.

        Improve efficieny of DVID masking.  It currently fetches entire partition
        even if there is only a small portion of masked data.

        Implement generic handler for dvid read/write to centralize handling of
        resource server.
    """

    def __init__(self, dvidserver, uuid, dataname, partitions, iteration_size=1,
            maskonly=True, resource_server="", resource_port=1200, spark_context = None):
        """Initialization.    
        
        Args:
            dvidserver (string): address for dvid server
            uuid (string): uuid for DVID instance 
            dataname (string): name of data instance (label or raw array type)
            partitions ([volumePartition, 3D numpy array or None]): this can be an RDD or standard list
            iteration_size (int): defines how large the iteration is, currently a no-op for Spark
            maskonly (boolean): only load data into 0 mask region in input partition
            resource_server (string): set if resource server is used
            resource_port (int): set if resource server is used
            spark_context (sparkconf): determines if spark is used for fetching data
        """

        super(dvidSrc, self).__init__(None)

        # will throw error if not available
        self.instance = DataInstance(dvidserver, uuid, dataname)

        # only supports volume interfaces
        if not self.instance.is_array():
            raise ValueError("not an label or raw array")
    
        # for now labelblk is the only supported label
        self.islabel = self.instance.is_labels()
       
        self.spark_context = spark_context
        self.maskonly = maskonly
        self.resource_server = resource_server
        self.resource_port = resource_port
        self.uuid = uuid
        self.server = dvidserver

        # set iteration size if relevant
        self.partitions = partitions
        self.iteration_size = 1
        self.usespark = True
        if type(partitions) == list:
            self.usespark = False
            self.iteration_size = iteration_size 
        self.current_spot = 0

    def __iter__(self):
        """Defines iterable type.
        """
        return self

    def next(self):
        """Iterates partitions specified in the partitionSchema.

        Node:
            Iteration cannot be done with an RDD pad source.
        """
        if self.usespark:
            raise ValueError("DVID source iteration in Spark not supported")
        
        if self.current_spot >= len(self.partitions):
            raise StopIteration()

        # RDD or array of [(partition, vol)]
        vols = self._retrieve_vol(self.current_spot, self.iteration_size)
        self.current_spot += self.iteration_size
        return vols

    def extract_volume(self):
        """Retrieve entire volume as numpy array or RDD.
        """

        # RDD or array of [(partition, vol)]
        vols = None
        if self.usespark:
            vols = self._retrieve_vol(self.current_spot, None)
        else:
            vols = self._retrieve_vol(self.current_spot, len(self.partitions))
            self.current_spot += len(self.partitions)
        
        return vols

    def _retrieve_vol(self, currentspot, itersize):
        """Calls DVID to fetch data for given partitions.
        """
    
        instance = self.instance
        maskonly = self.maskonly
        resource_server = self.resource_server
        resource_port = self.resource_port
        uuid = self.uuid
        server = self.server
        islabel = self.islabel

        def retrieve_vol(partvol):
            part, volume = partvol

            # check if there is no mask data
            if maskonly and ((part.mask is None) or (0 not in part.mask)):
                    return (part, volume)

            # fetch data
            # TODO: only fetch smaller subset if masked
           
            # grab extents and size (but take subset of data already exists)
            offset = part.get_offset()
            reloffset = part.get_reloffset()
            zoff = offset.z + reloffset.z
            yoff = offset.y + reloffset.y
            xoff = offset.x + reloffset.x
           
            volsize = part.get_volsize()
            volz = volsize.z
            voly = volsize.y
            volx = volsize.x
            if volume is not None:
                z,y,x = volume.shape
                volz = z
                voly = y
                volx = x

            # perform fetch
            node_service = retrieve_node_service(server, uuid,
                    resource_server, resource_port)  
            newvol = None
            offset = (zoff, yoff, xoff)
            shape= (volz, voly, volx)
            if resource_server != "": # throttling unnecessary with resource server
                if islabel:
                    newvol = node_service.get_labels3D(instance.name, shape, offset, throttle=False)
                else:
                    newvol = node_service.get_gray3D(instance.name, shape, offset, throttle=False)
            else: # throttle volume fetches if no resource server
                if islabel:
                    newvol = node_service.get_labels3D(instance.name, shape, offset, throttle=True)
                else:
                    newvol = node_service.get_gray3D(instance.name, shape, offset, throttle=True)

            # mask return data
            if maskonly:
                # 0 out areas not masked (1)
                newvol[part.mask != 0] = 0 

                # 0 out areas that are overwritten by DVID
                volume[part.mask ==  0] = 0

                # combine
                newvol = newvol + volume
            return (part, newvol)

        if self.usespark:
            return self.partitions.map(retrieve_vol)
        else:
            res = []
            for partnum in range(currentspot, currentspot+itersize):
                res.append(retrieve_vol(self.partitions[partnum]))
            return res

