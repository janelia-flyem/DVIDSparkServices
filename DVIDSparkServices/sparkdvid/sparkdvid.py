"""Contains core functionality for interfacing with DVID using Spark.

This module defines a sparkdvid type that allows for basic
reading and writing operations on DVID using RDDs.  The fundamental
unit of many of the functions is the Subvolume.

Helper functions for different workflow algorithms that work
with sparkdvid can be found in DVIDSparkServices.reconutil

Note: the RDDs for many of these transformations use the
unique subvolume key.  The mapping operations map only the values
and perserve the partitioner to make future joins faster.

Note: Access to DVID is done through the python bindings to libdvid-cpp.
For now, all volume GET/POST acceses are throttled (only one at a time)
because DVID is only being run on one server.  This will obviously
greatly reduce scalability but will be changed as soon as DVID
is backed by a clustered DB.

"""
from __future__ import division
import time

import numpy as np
from DVIDSparkServices.sparkdvid.Subvolume import Subvolume

import logging
from networkx.algorithms.shortest_paths import dense
logger = logging.getLogger(__name__)

from libdvid import SubstackZYX, DVIDException
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.util import mask_roi, RoiMap, blockwise_boxes, num_worker_nodes, cpus_per_worker, extract_subvol, runlength_decode_from_lengths, default_dvid_session
from DVIDSparkServices.io_util.partitionSchema import volumePartition
from DVIDSparkServices.io_util.brick import generate_bricks_from_volume_source
from DVIDSparkServices.dvid.metadata import create_label_instance, DataInstance, get_blocksize


def retrieve_node_service(server, uuid, resource_server, resource_port, appname="sparkservices"):
    """Create a DVID node service object"""

    server = str(server)  
   
    # refresh dvid server meta if localhost (since it is exclusive or points to global db)
    """
    if server.startswith("http://127.0.0.1") or  \
            server.startswith("http://localhost") or  \
            server.startswith("127.0.0.1") or server.startswith("localhost"):
        
        import os
        if not os.path.exists("/tmp/reloaded.hack"):
            addr = server + "/api/server/reload-metadata"
            if not server.startswith("http://"):
                addr = "http://" + addr

            session = default_dvid_session()
            session.post(addr)
            open("/tmp/reloaded.hack", 'w').close()
    """

    from libdvid import DVIDNodeService
    import os
    username = os.environ["USER"]

    if resource_server != "":
        node_service = DVIDNodeService(server, str(uuid), username, appname, str(resource_server), resource_port)
    else:
        node_service = DVIDNodeService(server, str(uuid), username, appname)


    return node_service

class sparkdvid(object):
    """Creates a spark dvid context that holds the spark context.

    Note: only the server name, context, and uuid are stored in the
    object to help reduce costs of serializing/deserializing the object.

    """
    
    BLK_SIZE = 32
    
    def __init__(self, context, dvid_server, dvid_uuid, workflow):
        """Initialize object

        Args:
            context: spark context
            dvid_server (str): location of dvid server (e.g. emdata2:8000)
            dvid_uuid (str): DVID dataset unique version identifier
            workflow (workflow): workflow instance
       
        """

        self.sc = context
        self.dvid_server = dvid_server
        self.uuid = dvid_uuid
        self.workflow = workflow

    # Produce RDDs for each subvolume partition (this will replace default implementation)
    # Treats subvolum index as the RDD key and maximizes partition count for now
    # Assumes disjoint subsvolumes in ROI
    def parallelize_roi(self, roi, chunk_size, border=0, find_neighbors=False, partition_method='ask-dvid', partition_filter=None):
        """Creates an RDD from subvolumes found in an ROI.

        This is analogous to the Spark parallelize function.
        It currently defines the number of partitions as the 
        number of subvolumes.

        TODO: implement general partitioner given other
        input such as bounding box coordinates.
        
        Args:
            roi (str): name of DVID ROI at current server and uuid
            chunk_size (int): the desired dimension of the subvolume
            border (int): size of the border surrounding the subvolume
            find_neighbors (bool): whether to identify neighbors

        Returns:
            RDD as [(subvolume id, subvolume)] and # of subvolumes

        """
        subvolumes = self._initialize_subvolumes(roi, chunk_size, border, find_neighbors, partition_method, partition_filter)
        enumerated_subvolumes = [(sv.sv_index, sv) for sv in subvolumes]

        # Potential TODO: custom partitioner for grouping close regions
        return self.sc.parallelize(enumerated_subvolumes, len(enumerated_subvolumes))

    def _initialize_subvolumes(self, roi, chunk_size, border=0, find_neighbors=False, partition_method='ask-dvid', partition_filter=None):
        if partition_method == 'grid-aligned':
            partition_method = 'grid-aligned-32'
        assert partition_method == 'ask-dvid' or partition_method.startswith('grid-aligned-')
        assert partition_filter in (None, "all", 'interior-only')
        if partition_filter == "all":
            partition_filter = None

        # Split ROI into subvolume chunks
        substack_tuples = self.get_roi_partition(roi, chunk_size, partition_method)

        # Create dense representation of ROI
        roi_map = RoiMap( self.get_roi(roi) )

        # Initialize all Subvolumes (sv_index is updated below)
        subvolumes = [Subvolume(None, (ss.z, ss.y, ss.x), chunk_size, border, roi_map) for ss in substack_tuples]

        # Discard empty subvolumes (ones that don't intersect the ROI at all).
        # The 'grid-aligned' partition-method can return such subvolumes;
        # it assumes we'll filter them out, which we're doing right now.
        subvolumes = [sv for sv in subvolumes if len(sv.intersecting_blocks_noborder) != 0]

        # Discard 'interior' subvolumes if the user doesn't want them.
        if partition_filter == 'interior-only':
            subvolumes = [sv for sv in subvolumes if sv.is_interior]

        # Assign sv_index
        for i, sv in enumerate(subvolumes):
            sv.sv_index = i

        # grab all neighbors for each substack
        if find_neighbors:
            # inefficient search for all boundaries
            for i in range(0, len(subvolumes)-1):
                for j in range(i+1, len(subvolumes)):
                    subvolumes[i].recordborder(subvolumes[j])

        return subvolumes

    def get_roi(self, roi):
        """
        An alternate implementation of libdvid.DVIDNodeService.get_roi(),
        since DVID sometimes returns strange 503 errors and DVIDNodeService.get_roi()
        doesn't know how to handle them.
        """
        session = default_dvid_session()

        # grab roi blocks (should use libdvid but there are problems handling 206 status)
        import requests
        addr = self.dvid_server + "/api/node/" + str(self.uuid) + "/" + str(roi) + "/roi"
        if not self.dvid_server.startswith("http://"):
            addr = "http://" + addr
        data = session.get(addr)
        roi_blockruns = data.json()
        
        roi_blocks = []
        for (z,y,x_first, x_last) in roi_blockruns:
            for x in range(x_first, x_last+1):
                roi_blocks.append((z,y,x))
        
        return roi_blocks

    def get_roi_partition(self, roi_name, subvol_size, partition_method):
        """
        Partition the given ROI into a list of 'Substack' tuples (size,z,y,x).
        
        roi_name:
            string
        subvol_size:
            The size of the substack without overlap border
        partition_method:
            Either 'ask-dvid' or 'grid-aligned-<N>', where <N> is the grid width in px, e.g. 'grid-aligned-64'
            Note: If using 'grid-aligned', the set of Substacks may
                  include 'empty' substacks that don't overlap the ROI at all.
            
        """
        assert subvol_size % self.BLK_SIZE == 0, \
            "This function assumes chunk size is a multiple of block size"

        node_service = retrieve_node_service(self.dvid_server, self.uuid, self.workflow.resource_server, self.workflow.resource_port)
        if partition_method == 'ask-dvid':
            subvol_tuples, _ = node_service.get_roi_partition(str(roi_name), subvol_size // self.BLK_SIZE)
            return subvol_tuples

        if partition_method == 'grid-aligned':
            # old default
            partition_method = 'grid-aligned-32'

        if partition_method.startswith('grid-aligned-'):
            grid_spacing_px = int(partition_method.split('-')[-1])
            grid_spacing_blocks = grid_spacing_px // self.BLK_SIZE

            assert subvol_size % grid_spacing_px == 0, \
                "Subvolume partitions won't be aligned to grid unless subvol_size is a multiple of the grid size."
            
            roi_blocks = np.asarray(list(self.get_roi(roi_name)))
            roi_blocks_start = np.min(roi_blocks, axis=0)
            roi_blocks_stop = 1 + np.max(roi_blocks, axis=0)
            
            # Clip start/stop to grid
            roi_blocks_start = (roi_blocks_start // grid_spacing_blocks) * grid_spacing_blocks # round down to grid
            roi_blocks_stop = ((roi_blocks_stop + grid_spacing_blocks - 1) // grid_spacing_blocks) * grid_spacing_blocks # round up to grid
            
            roi_blocks_shape = roi_blocks_stop - roi_blocks_start
    
            sv_size_in_blocks = (subvol_size // self.BLK_SIZE)
            
            # How many subvolumes wide is the ROI in each dimension?
            roi_shape_in_subvols = (roi_blocks_shape + sv_size_in_blocks - 1) // sv_size_in_blocks
    
            subvol_tuples = []
            for subvol_index in np.ndindex(*roi_shape_in_subvols):
                subvol_index = np.array(subvol_index)
                subvol_start = subvol_size*subvol_index + (roi_blocks_start*self.BLK_SIZE)
                z_start, y_start, x_start = subvol_start
                subvol_tuples.append( SubstackZYX(subvol_size, z_start, y_start, x_start) )
            return subvol_tuples

        # Shouldn't get here
        raise RuntimeError('Unknown partition_method: {}'.format( partition_method ))

    def parallelize_bounding_box( self,
                                  instance_name,
                                  bounding_box_zyx,
                                  grid,
                                  target_partition_size_voxels ):
        """
        Create an RDD for the given data instance (of either grayscale, labelblk, labelarray, or labelmap),
        within the given bounding_box (start_zyx, stop_zyx) and split into blocks of the given shape.
        The RDD parallelism will be set to include approximately target_partition_size_voxels in total.
        """
        block_size_voxels = np.prod(grid.block_shape)
        rdd_partition_length = target_partition_size_voxels // block_size_voxels

        bricks = generate_bricks_from_volume_source( bounding_box_zyx,
                                                     grid,
                                                     self.get_volume_accessor(instance_name),
                                                     self.sc,
                                                     rdd_partition_length )
        
        # If we're working with a tiny volume (e.g. testing),
        # make sure we at least parallelize across all cores.
        if bricks.getNumPartitions() < cpus_per_worker() * num_worker_nodes():
            bricks = bricks.repartition( cpus_per_worker() * num_worker_nodes() )

        return bricks

        
    def checkpointRDD(self, rdd, checkpoint_loc, enable_rollback):
        """Defines functionality for checkpointing an RDD.

        Future implementation should be a member function of RDD.

        """
        import os
        from pyspark import StorageLevel
        from pyspark.storagelevel import StorageLevel

        if not enable_rollback or not os.path.exists(checkpoint_loc): 
            if os.path.exists(checkpoint_loc):
                import shutil
                shutil.rmtree(checkpoint_loc)
            rdd.persist(StorageLevel.MEMORY_AND_DISK_SER)
            rdd.saveAsPickleFile(checkpoint_loc)
            return rdd
        else:
            newrdd = self.sc.pickleFile(checkpoint_loc)
            return newrdd


    def map_grayscale8(self, distsubvolumes, gray_name):
        """Creates RDD of grayscale data from subvolumes.

        Note: Since EM grayscale is not highly compressible
        lz4 is not called.

        Args:
            distsubvolumes (RDD): (subvolume id, subvolume)
            gray_name (str): name of grayscale instance

        Returns:
            RDD of grayscale data (partitioner perserved)
    
        """
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid
        resource_server = self.workflow.resource_server
        resource_port = self.workflow.resource_port

        # only grab value
        def mapper(subvolume):
            # extract grayscale x
            # get sizes of subvolume
            size_x = subvolume.box.x2 + 2*subvolume.border - subvolume.box.x1
            size_y = subvolume.box.y2 + 2*subvolume.border - subvolume.box.y1
            size_z = subvolume.box.z2 + 2*subvolume.border - subvolume.box.z1

            #logger = logging.getLogger(__name__)
            #logger.warn("FIXME: As a temporary hack, this introduces a pause before accessing grayscale, to offset accesses to dvid")
            #import time
            #import random
            #time.sleep( random.randint(0,512) )

            # retrieve data from box start position considering border
            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_gray():
                # Note: libdvid uses zyx order for python functions
                node_service = retrieve_node_service(server, uuid,resource_server, resource_port)
                if resource_server != "":
                    return node_service.get_gray3D( str(gray_name),
                                                    (size_z, size_y, size_x),
                                                    (subvolume.box.z1-subvolume.border, subvolume.box.y1-subvolume.border, subvolume.box.x1-subvolume.border), throttle=False )
                else:
                    return node_service.get_gray3D( str(gray_name),
                                                    (size_z, size_y, size_x),
                                                    (subvolume.box.z1-subvolume.border, subvolume.box.y1-subvolume.border, subvolume.box.x1-subvolume.border) )

            gray_volume = get_gray()

            return (subvolume, gray_volume)

        return distsubvolumes.mapValues(mapper)

    def map_labels64(self, distrois, label_name, border, roiname=""):
        """Creates RDD of labelblk data from subvolumes.

        Note: Numpy arrays are compressed which leads to some savings.
        
        Args:
            distrois (RDD): (subvolume id, subvolume)
            label_name (str): name of labelblk instance
            border (int): size of substack border
            roiname (str): name of the roi (to restrict fetch precisely)
            compress (bool): true return compressed numpy

        Returns:
            RDD of compressed lableblk data (partitioner perserved)
            (subvolume, label_comp)
    
        """

        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid
        resource_server = self.workflow.resource_server
        resource_port = self.workflow.resource_port

        def mapper(subvolume):
            # get sizes of box
            size_x = subvolume.box.x2 + 2*subvolume.border - subvolume.box.x1
            size_y = subvolume.box.y2 + 2*subvolume.border - subvolume.box.y1
            size_z = subvolume.box.z2 + 2*subvolume.border - subvolume.box.z1

            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_labels():
                # extract labels 64
                # retrieve data from box start position considering border
                # Note: libdvid uses zyx order for python functions
                node_service = retrieve_node_service(server, uuid, resource_server, resource_port)
                if resource_server != "":
                    data = node_service.get_labels3D( str(label_name),
                                                      (size_z, size_y, size_x),
                                                      (subvolume.box.z1-subvolume.border, subvolume.box.y1-subvolume.border, subvolume.box.x1-subvolume.border),
                                                      compress=True, throttle=False )
                else:
                    data = node_service.get_labels3D( str(label_name),
                                                      (size_z, size_y, size_x),
                                                      (subvolume.box.z1-subvolume.border, subvolume.box.y1-subvolume.border, subvolume.box.x1-subvolume.border),
                                                      compress=True )

                # mask ROI
                if roiname != "":
                    mask_roi(data, subvolume, border=border)        

                return data
            return get_labels()
        return distrois.mapValues(mapper)

    def map_voxels(self, partitions, instance_name, scale=0, num_rdd_partitions=None):
        """
        Given a list of volumePartition objects, return an RDD of (partition, volume_data).
        """
        server = self.dvid_server
        uuid = self.uuid
        instance_name = str(instance_name)
        resource_server = self.workflow.resource_server
        resource_port = self.workflow.resource_port
        data_instance = DataInstance(server, uuid, instance_name)
        datatype = data_instance.datatype
        is_labels = data_instance.is_labels()

        def mapper(partition):
            assert isinstance(partition, volumePartition)
            assert np.prod(partition.volsize) > 0, \
                "volumePartition must have nonzero size.  You gave: {}".format( volumePartition )
            
            # Two-levels of auto-retry:
            # 1. Auto-retry up to three time for any reason.
            # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
            @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                        predicate=lambda ex: '503' in ex.args[0] or '504' in ex.args[0])
            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_voxels():
                return sparkdvid.get_voxels( server, uuid, instance_name, scale,
                                             datatype, is_labels,
                                             partition.volsize, partition.offset,
                                             resource_server, resource_port )

        if num_rdd_partitions is None:
            num_rdd_partitions = len(partitions)

        return self.sc.parallelize(partitions, num_rdd_partitions).map(mapper)

    @classmethod
    def get_voxels( cls, server, uuid, instance_name, scale,
                    instance_type, is_labels,
                    volume_shape, offset,
                    resource_server="", resource_port=0, throttle="auto", supervoxels=False, node_service=None):

        if node_service is None:
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port)

        if throttle == "auto":
            throttle = (resource_server == "")
        
        if instance_type in ('labelarray', 'labelmap'):
            # Labelarray data can be fetched very efficiently if the request is block-aligned
            # So, block-align the request no matter what.
            aligned_start = np.array(offset) // 64 * 64
            aligned_stop = (np.array(offset) + volume_shape + 64-1) // 64 * 64
            aligned_shape = aligned_stop - aligned_start
            aligned_volume = node_service.get_labelarray_blocks3D( instance_name, aligned_shape, aligned_start, throttle, scale )
            requested_box_within_aligned = ( offset - aligned_start,
                                             offset - aligned_start + volume_shape )
            return extract_subvol(aligned_volume, requested_box_within_aligned )
                
        elif is_labels:
            assert scale == 0, "FIXME: get_labels3D() doesn't support scale yet!"
            # labelblk (or non-aligned labelarray) must be fetched the old-fashioned way
            return node_service.get_labels3D( instance_name, volume_shape, offset, throttle, compress=True, supervoxels=supervoxels )
        else:
            assert scale == 0, "FIXME: get_gray3D() doesn't support scale yet!"
            return node_service.get_gray3D( instance_name, volume_shape, offset, throttle, compress=False )

    @classmethod
    def post_voxels( cls, server, uuid, instance_name, scale,
                     instance_type, is_labels,
                     subvolume, offset,
                     resource_server="", resource_port=0,
                     throttle="auto",
                     disable_indexing=False,
                     node_service=None):

        if node_service is None:
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port)

        if throttle == "auto":
            throttle = (resource_server == "")
        
        if instance_type in ('labelarray', 'labelmap') and (np.array(offset) % 64 == 0).all() and (np.array(subvolume.shape) % 64 == 0).all():
            # Labelarray data can be posted very efficiently if the request is block-aligned
            node_service.put_labelblocks3D( instance_name, subvolume, offset, throttle, scale, disable_indexing)
                
        elif is_labels:
            assert disable_indexing == False, "Can't use put_labels3D in ingestion mode."
            assert scale == 0, "FIXME: put_labels3D() doesn't support scale yet!"
            # labelblk (or non-aligned labelarray) must be posted the old-fashioned way
            node_service.put_labels3D( instance_name, subvolume, offset, throttle)
        else:
            assert disable_indexing == False, "put_gray3D() is not aware of indexing."
            assert scale == 0, "FIXME: put_gray3D() doesn't support scale yet!"
            node_service.put_gray3D( instance_name, subvolume, offset, throttle)


    @classmethod
    def get_legacy_sparsevol(cls, server, uuid, instance_name, body_id, scale=0):
        """
        Returns the coordinates (Z,Y,X) of all voxels in the given body_id at the given scale.
        
        Note: For large bodies, this will be a LOT of coordinates at scale 0.
        
        Note: The returned coordinates are native to the requested scale.
              For instance, if the first Z-coordinate at scale 0 is 128,
              then at scale 1 it is 64, etc.
        
        Note: This function requests the data from DVID in the legacy 'rles' format,
              which is much less efficient than the newer 'blocks' format
              (but it's easy enough to parse that we can do it in Python).

        Return an array of coordinates of the form:
    
            [[Z,Y,X],
             [Z,Y,X],
             [Z,Y,X],
             ...
            ]
        """
        if not server.startswith('http://'):
            server = 'http://' + server
        session = default_dvid_session()
        r = session.get(f'{server}/api/node/{uuid}/{instance_name}/sparsevol/{body_id}?format=rles&scale={scale}')
        r.raise_for_status()
        
        return cls._parse_rle_response( r.content )

    @classmethod
    def get_coarse_sparsevol(cls, server, uuid, instance_name, body_id):
        """
        Return the 'coarse sparsevol' representation of a given body.
        This is similar to the sparsevol representation at scale=6,
        EXCEPT that it is generated from the label index, so no blocks
        are lost from downsampling.

        Return an array of coordinates of the form:
    
            [[Z,Y,X],
             [Z,Y,X],
             [Z,Y,X],
             ...
            ]
        """
        if not server.startswith('http://'):
            server = 'http://' + server
        session = default_dvid_session()
        r = session.get(f'{server}/api/node/{uuid}/{instance_name}/sparsevol-coarse/{body_id}')
        r.raise_for_status()
        
        return cls._parse_rle_response( r.content )

    @classmethod
    def _parse_rle_response(cls, response_bytes):
        """
        Parse the (legacy) RLE response from DVID's 'sparsevol' and 'sparsevol-coarse' endpoints.
        
        Return an array of coordinates of the form:
    
            [[Z,Y,X],
             [Z,Y,X],
             [Z,Y,X],
             ...
            ]
        """
        descriptor = response_bytes[0]
        ndim = response_bytes[1]
        run_dimension = response_bytes[2]

        assert descriptor == 0, f"Don't know how to handle this payload. (descriptor: {descriptor})"
        assert ndim == 3, "Expected XYZ run-lengths"
        assert run_dimension == 0, "FIXME, we assume the run dimension is X"

        content_as_int32 = np.frombuffer(response_bytes, np.int32)
        _voxel_count = content_as_int32[1]
        run_count = content_as_int32[2]
        rle_items = content_as_int32[3:].reshape(-1,4)

        assert len(rle_items) == run_count, \
            f"run_count ({run_count}) doesn't match data array length ({len(rle_items)})"

        rle_starts_xyz = rle_items[:,:3]
        rle_starts_zyx = rle_starts_xyz[:,::-1]
        rle_lengths = rle_items[:,3]

        # Sadly, the decode function requires contiguous arrays, so we must copy.
        rle_starts_zyx = rle_starts_zyx.copy('C')
        rle_lengths = rle_lengths.copy('C')

        # For now, DVID always returns a voxel_count of 0, so we can't make this assertion.
        #assert rle_lengths.sum() == _voxel_count,\
        #    f"Voxel count ({voxel_count}) doesn't match expected sum of run-lengths ({rle_lengths.sum()})"

        dense_coords = runlength_decode_from_lengths(rle_starts_zyx, rle_lengths)
        
        assert rle_lengths.sum() == len(dense_coords), "Got the wrong number of coordinates!"
        return dense_coords


    @classmethod
    def get_union_block_mask_for_bodies( cls, server, uuid, instance_name, body_ids ):
        """
        Given a list of body IDs, fetch their sparse blocks from DVID and
        return a mask of all blocks touched by those bodies.
        A binary image is returned, in which each voxel of the result
        corresponds to a block in DVID.
        
        Returns: mask, box, block_shape
            Where mask is a binary image (each voxel represents a block),
            box is the bounding box of the mask, in BLOCK coordinates,
            and block_shape is the DVID server's native block shape (usually (64,64,64)).
        """
        all_block_coords = {} # { body : coord_list }
        
        union_box = None
        
        for body_id in body_ids:
            block_coords = cls.get_coarse_sparsevol( server, uuid, instance_name, body_id )
            
            all_block_coords[body_id] = block_coords
            
            min_coord = block_coords.min(axis=0)
            max_coord = block_coords.max(axis=0)

            if union_box is None:
                union_box = np.array( (min_coord, max_coord+1) )
            else:
                union_box[0] = np.minimum( min_coord, union_box[0] )
                union_box[1] = np.maximum( max_coord+1, union_box[1] )

        union_shape = union_box[1] - union_box[0]
        union_mask = np.zeros( union_shape, bool )

        for body_id, block_coords in all_block_coords.items():
            block_coords[:] -= union_box[0]
            union_mask[tuple(block_coords.transpose())] = True

        return union_mask, union_box, get_blocksize(server, uuid, instance_name)


    def get_volume_accessor(self, instance_name, scale=0):
        """
        Returns a volume_accessor_func in the form expected by brick.py
        """
        server = self.dvid_server
        uuid = self.uuid
        instance_name = str(instance_name)
        resource_server = self.workflow.resource_server
        resource_port = self.workflow.resource_port
        data_instance = DataInstance(server, uuid, instance_name)
        datatype = data_instance.datatype
        is_labels = data_instance.is_labels()

        # Two-levels of auto-retry:
        # 1. Auto-retry up to three time for any reason.
        # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
        @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                    predicate=lambda ex: '503' in ex.args[0] or '504' in ex.args[0])
        @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
        def get_voxels(box):
            shape = np.asarray(box[1]) - box[0]
            return sparkdvid.get_voxels( server, uuid, instance_name, scale,
                                         datatype, is_labels,
                                         shape, box[0],
                                         resource_server, resource_port )

        return get_voxels

    def map_labels64_pair(self, distrois, label_name, dvidserver2, uuid2, label_name2, roiname="", level=0):
        """Creates RDD of two subvolumes (same ROI but different datasets)

        This functionality is used to compare two subvolumes.

        Note: Numpy arrays are compressed which leads to some savings.
        
        Args:
            distrois (RDD): (subvolume id, subvolume)
            label_name (str): name of labelblk instance
            dvidserver2 (str): name of dvid server for label_name2
            uuid2 (str): dataset uuid version for label_name2
            label_name2 (str): name of labelblk instance
            roiname (str): name of the roi (to restrict fetch precisely)

        Returns:
            RDD of compressed lableblk, labelblk data (partitioner perserved).
            (subvolume, label1_comp, label2_comp)

        """

        # copy local context to minimize sent data
        server = self.dvid_server
        server2 = dvidserver2
        uuid = self.uuid
        resource_server = self.workflow.resource_server
        resource_port = self.workflow.resource_port

        # check whether to use labelarray or labelblk interface
        ns_temp = retrieve_node_service(server, uuid, resource_server, resource_port)
        labeltype = ns_temp.get_typeinfo(str(label_name))["Base"]["TypeName"]
        islabelarray = False
        if labeltype == "labelarray" or labeltype == "labelmap":
            islabelarray = True

        islabelarray2 = islabelarray

        # just assume gt and seg have same blocksize and isotropic
        blocksize = get_blocksize(server, uuid, label_name)[0]

        if server2 != "":
            ns_temp = retrieve_node_service(server2, uuid2, resource_server, resource_port)
            labeltype = ns_temp.get_typeinfo(str(label_name2))["Base"]["TypeName"]
            islabelarray2 = False
            if labeltype == "labelarray" or labeltype == "labelmap":
                islabelarray2 = True

        def mapper(subvolume):
            # get sizes of box
            size_x = subvolume.box.x2 - subvolume.box.x1
            size_y = subvolume.box.y2 - subvolume.box.y1
            size_z = subvolume.box.z2 - subvolume.box.z1

            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_labels():
                # extract labels 64
                # retrieve data from box start position
                # Note: libdvid uses zyx order for python functions
                node_service = retrieve_node_service(server, uuid, resource_server, resource_port)

                get3d = node_service.get_labels3D
                if islabelarray:
                    get3d = node_service.get_labelarray_blocks3D


                z1 = subvolume.box.z1
                y1 = subvolume.box.y1
                x1 = subvolume.box.x1
                sz = size_z
                sy = size_y
                sx = size_x

                sz = sz + (z1 % blocksize)
                sy = sy + (y1 % blocksize)
                sx = sx + (x1 % blocksize)
                z1 = z1 - (z1 % blocksize)
                y1 = y1 - (y1 % blocksize)
                x1 = x1 - (x1 % blocksize)

                if (sz % blocksize) > 0:
                    sz = sz + blocksize - (sz%blocksize)
                if (sy % blocksize) > 0:
                    sy = sy + blocksize - (sy%blocksize)
                if (sx % blocksize) > 0:
                    sx = sx + blocksize - (sx%blocksize)

                if level > 0:
                    if resource_server != "":
                        data = get3d( str(label_name),
                                      (sz, sy, sx),
                                      (z1, y1, x1), throttle=False, scale=level)
                    else:
                        data = get3d( str(label_name),
                                      (sz, sy, sx),
                                      (z1, y1, x1), throttle=True, scale=level)

                    data = data[(subvolume.box.z1-z1):(subvolume.box.z1-z1+size_z),(subvolume.box.y1-y1):(subvolume.box.y1-y1+size_y),(subvolume.box.x1-x1):(subvolume.box.x1-x1+size_x)]

                else:
                    if resource_server != "":
                        data = get3d( str(label_name),
                                      (sz, sy, sx),
                                      (z1, y1, x1), throttle=False)
                    else:
                        data = get3d( str(label_name),
                                      (sz, sy, sx),
                                      (z1, y1, x1), throttle=True)
                    data = data[(subvolume.box.z1-z1):(subvolume.box.z1-z1+size_z),(subvolume.box.y1-y1):(subvolume.box.y1-y1+size_y),(subvolume.box.x1-x1):(subvolume.box.x1-x1+size_x)]

                # mask ROI
                if roiname != "":
                    mask_roi(data, subvolume) 

                return data
            label_volume = get_labels()

            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_labels2():
                # fetch dummy value if no server2
                if server2 == "":
                    label_volume2 = label_volume.copy()
                    label_volume2[:,:,:] = 1
                    return label_volume2

                # fetch second label volume
                # retrieve data from box start position
                # Note: libdvid uses zyx order for python functions
                node_service2 = retrieve_node_service(server2, uuid2, resource_server, resource_port)
                
                get3d = node_service2.get_labels3D
                if islabelarray2:
                    get3d = node_service2.get_labelarray_blocks3D
                
                z1 = subvolume.box.z1
                y1 = subvolume.box.y1
                x1 = subvolume.box.x1
                sz = size_z
                sy = size_y
                sx = size_x

                sz = sz + (z1 % blocksize)
                sy = sy + (y1 % blocksize)
                sx = sx + (x1 % blocksize)
                z1 = z1 - (z1 % blocksize)
                y1 = y1 - (y1 % blocksize)
                x1 = x1 - (x1 % blocksize)

                if (sz % blocksize) > 0:
                    sz = sz + blocksize - (sz%blocksize)
                if (sy % blocksize) > 0:
                    sy = sy + blocksize - (sy%blocksize)
                if (sx % blocksize) > 0:
                    sx = sx + blocksize - (sx%blocksize)

                if level > 0:
                    if resource_server != "":
                        data = get3d( str(label_name2),
                                      (sz, sy, sx),
                                      (z1, y1, x1), throttle=False, scale=level)
                    else:
                        data = get3d( str(label_name2),
                                      (sz, sy, sx),
                                      (z1, y1, x1), throttle=True, scale=level)
                    data = data[(subvolume.box.z1-z1):(subvolume.box.z1-z1+size_z),(subvolume.box.y1-y1):(subvolume.box.y1-y1+size_y),(subvolume.box.x1-x1):(subvolume.box.x1-x1+size_x)]
                else:
                    if resource_server != "":
                        data = get3d( str(label_name2),
                                      (sz, sy, sx),
                                      (z1, y1, x1), throttle=False)
                    else:
                        data = get3d( str(label_name2),
                                      (sz, sy, sx),
                                      (z1, y1, x1), throttle=True)
                    data = data[(subvolume.box.z1-z1):(subvolume.box.z1-z1+size_z),(subvolume.box.y1-y1):(subvolume.box.y1-y1+size_y),(subvolume.box.x1-x1):(subvolume.box.x1-x1+size_x)]

                if roiname != "":
                    mask_roi(data, subvolume)        
                return data

            label_volume2 = get_labels2()

            # zero out label_volume2 where GT is 0'd out !!
            label_volume2[label_volume==0] = 0

            return (subvolume, label_volume, label_volume2)

        return distrois.mapValues(mapper)


    # foreach will write graph elements to DVID storage
    def foreachPartition_graph_elements(self, elements, graph_name):
        """Write graph nodes or edges to DVID labelgraph.

        Write nodes and edges of the specified size and weight
        to DVID.

        This operation works over a partition which could
        have many Sparks tasks.  The reason for this is to
        consolidate the number of small requests made to DVID.

        Note: edges and vertices are both encoded in the same
        datastructure (node1, node2).  node2=-1 for vertices.

        Args:
            elements (RDD): graph elements ((node1, node2), size)
            graph_name (str): name of DVID labelgraph (already created)

        """
        
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid
        resource_server = self.workflow.resource_server
        resource_port = self.workflow.resource_port
        
        def writer(element_pairs):
            from libdvid import Vertex, Edge
            
            # write graph information
            if element_pairs is None:
                return

            vertices = []
            edges = []
            for element_pair in element_pairs:
                edge, weight = element_pair
                v1, v2 = edge

                if v2 == -1:
                    vertices.append(Vertex(v1, weight))
                else:
                    edges.append(Edge(v1, v2, weight))
    
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port)
            if len(vertices) > 0:
                node_service.update_vertices(str(graph_name), vertices) 
            
            if len(edges) > 0:
                node_service.update_edges(str(graph_name), edges) 
            
            return []

        elements.foreachPartition(writer)

    def foreach_ingest_labelarray(self, label_name, seg_chunks):
        """
        Create a labelarray instance and indest the given RDD of
        segmentation chunks using DVID's API for high-speed ingestion
        of native blocks.
        
        Note: seg_chunks must be block-aligned!
        """
        server = self.dvid_server
        uuid = self.uuid
        resource_server = self.workflow.resource_server
        resource_port = self.workflow.resource_port

        # Create labelarray instance if necessary
        create_label_instance(server, uuid, label_name)
        
        def writer(subvolume_seg):
            _key, (subvolume, seg) = subvolume_seg
            
            # Discard border (if any)
            b = subvolume.border
            seg = seg[b:-b, b:-b, b:-b]
            
            # Ensure contiguous
            seg = np.asarray(seg, order='C')
            
            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def put_labels():
                throttle = (resource_server == "")
                node_service = retrieve_node_service(server, uuid, resource_server, resource_port)
                node_service.put_labelblocks3D( str(label_name),
                                                seg,
                                                subvolume.box[:3],
                                                throttle )
            put_labels()

        return seg_chunks.foreach(writer)


    # (key, (ROI, segmentation compressed+border))
    # => segmentation output in DVID
    def foreach_write_labels3d(self, label_name, seg_chunks, roi_name=None, mutateseg="auto"):
        """Writes RDD of label volumes to DVID.

        For each subvolume ID, this function writes the subvolume
        ROI not including the border.  The data is actually sent
        compressed to minimize network latency.

        Args:
            label_name (str): name of already created DVID labelblk
            seg_chunks (RDD): (key, (subvolume, label volume)
            roi_name (str): restrict write to within this ROI
            mutateseg (str): overwrite previous seg ("auto", "yes", "no"
            "auto" will check existence of labels beforehand)

        """

        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid
        resource_server = self.workflow.resource_server
        resource_port = self.workflow.resource_port

        # create labels type
        node_service = retrieve_node_service(server, uuid, resource_server, resource_port)
        success = node_service.create_labelblk(str(label_name))

        # check whether seg should be mutated
        mutate=False
        if (not success and mutateseg == "auto") or mutateseg == "yes":
            mutate=True

        def writer(subvolume_seg):
            import numpy
            # write segmentation
            
            (key, (subvolume, seg)) = subvolume_seg
            # get sizes of subvolume 
            size1 = subvolume.box.x2-subvolume.box.x1
            size2 = subvolume.box.y2-subvolume.box.y1
            size3 = subvolume.box.z2-subvolume.box.z1

            border = subvolume.border

            # extract seg ignoring borders (z,y,x)
            seg = seg[border:size3+border, border:size2+border, border:size1+border]

            # copy the slice to make contiguous before sending 
            seg = numpy.copy(seg, order='C')

            @auto_retry(3, pause_between_tries=600.0, logging_name= __name__)
            def put_labels():
                # send data from box start position
                # Note: libdvid uses zyx order for python functions
                node_service = retrieve_node_service(server, uuid, resource_server, resource_port)
                
                throttlev = True
                if resource_server != "":
                    throttlev = False
                if roi_name is None:
                    node_service.put_labels3D( str(label_name),
                                               seg,
                                               (subvolume.box.z1, subvolume.box.y1, subvolume.box.x1),
                                               compress=True, throttle=throttlev,
                                               mutate=mutate )
                else: 
                    node_service.put_labels3D( str(label_name),
                                               seg,
                                               (subvolume.box.z1, subvolume.box.y1, subvolume.box.x1),
                                               compress=True, throttle=throttlev,
                                               roi=str(roi_name),
                                               mutate=mutate )
            put_labels()

        return seg_chunks.foreach(writer)

