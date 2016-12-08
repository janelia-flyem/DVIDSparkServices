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

import numpy as np
from DVIDSparkServices.sparkdvid.Subvolume import Subvolume

import logging
logger = logging.getLogger(__name__)

from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.util import mask_roi, coordlist_to_boolmap, bb_to_slicing

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
            import requests
            addr = server + "/api/server/reload-metadata"
            if not server.startswith("http://"):
                addr = "http://" + addr

            requests.post(addr)
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
    def parallelize_roi(self, roi, chunk_size, border=0, find_neighbors=False):
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

        # (map blocks to y,z lines, iterate everything including border and add relevant xy lines) 

        # function will export and should include dependencies
        subvolumes = []
        
        # extract roi for a given chunk size
        node_service = retrieve_node_service(self.dvid_server, self.uuid, self.workflow.resource_server, self.workflow.resource_port)
        substacks, packing_factor = node_service.get_roi_partition(str(roi), chunk_size / self.BLK_SIZE)
        
        # libdvid returns substack namedtuples as (size, z, y, x), but we want just (z,y,x)
        substacks = map(lambda s: (s.z, s.y, s.x), substacks)
      
        # create roi array giving unique substack ids
        for substack_id, substack in enumerate(substacks):
            # use substack id as key
            subvolumes.append((substack_id, Subvolume(substack_id, substack, chunk_size, border)))

        # grab roi blocks (should use libdvid but there are problems handling 206 status)
        import requests
        addr = self.dvid_server + "/api/node/" + str(self.uuid) + "/" + str(roi) + "/roi"
        if not self.dvid_server.startswith("http://"):
            addr = "http://" + addr
        data = requests.get(addr)
        roi_blockruns = data.json()
        
        roi_blocks = []
        for (z,y,x_first, x_last) in roi_blockruns:
            for x in range(x_first, x_last+1):
                roi_blocks.append((z,y,x))

        # Make a map of the entire ROI
        # Since roi blocks are 32^2, this won't be too huge.
        # For example, a ROI that's 10k*10k*100k pixels, this will be ~300 MB
        # For a 100k^3 ROI, this will be 30 GB (still small enough to fit in RAM on the driver) 
        full_roi_blockmask, (roi_blocks_start, roi_blocks_stop) = coordlist_to_boolmap(roi_blocks)
        roi_blocks_shape = roi_blocks_stop - roi_blocks_start
  
        # Initialize each subvolume's 'intersecting_blocks' member for the ROI blocks it contains.
        for (_sid, subvol) in subvolumes:
            # Subvol bounding-box in pixels
            subvol_start_px = np.array((subvol.roi.z1, subvol.roi.y1, subvol.roi.x1)) - subvol.border
            subvol_stop_px  = np.array((subvol.roi.z2, subvol.roi.y2, subvol.roi.x2)) + subvol.border
            
            # Subvol bounding box in block coords
            subvol_blocks_start = subvol_start_px // subvol.roi_blocksize
            subvol_blocks_stop = (subvol_stop_px + subvol.roi_blocksize-1) // subvol.roi_blocksize
            subvol_blocks_shape = subvol_blocks_stop - subvol_blocks_start

            # Where does this subvolume start within full_roi_blockmask?
            # Offset, since the ROI didn't necessarily start at (0,0,0)
            subvol_blocks_offset = subvol_blocks_start - roi_blocks_start
            
            # Clip the extracted region, since subvol may extend outside of ROI and therefore outside of full_roi_blockmask
            subvol_blocks_clipped_start = np.maximum(subvol_blocks_offset, (0,0,0))
            subvol_blocks_clipped_stop = np.minimum(roi_blocks_shape, (subvol_blocks_start + subvol_blocks_shape) - roi_blocks_start)
            
            # Extract the portion of the mask for this subvol
            subvol_blocks_mask = full_roi_blockmask[bb_to_slicing(subvol_blocks_clipped_start, subvol_blocks_clipped_stop)]
            subvol_block_coords = np.transpose( subvol_blocks_mask.nonzero() )
            
            # Un-offset.
            subvol_block_coords += (subvol_blocks_clipped_start + roi_blocks_start)
            
            subvol.intersecting_blocks = subvol_block_coords

        # grab all neighbors for each substack
        if find_neighbors:
            # inefficient search for all boundaries
            for i in range(0, len(subvolumes)-1):
                for j in range(i+1, len(subvolumes)):
                    subvolumes[i][1].recordborder(subvolumes[j][1])

        # Potential TODO: custom partitioner for grouping close regions
        return self.sc.parallelize(subvolumes, len(subvolumes))

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
            size_x = subvolume.roi.x2 + 2*subvolume.border - subvolume.roi.x1
            size_y = subvolume.roi.y2 + 2*subvolume.border - subvolume.roi.y1
            size_z = subvolume.roi.z2 + 2*subvolume.border - subvolume.roi.z1

            #logger = logging.getLogger(__name__)
            #logger.warn("FIXME: As a temporary hack, this introduces a pause before accessing grayscale, to offset accesses to dvid")
            #import time
            #import random
            #time.sleep( random.randint(0,512) )

            # retrieve data from roi start position considering border
            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_gray():
                # Note: libdvid uses zyx order for python functions
                node_service = retrieve_node_service(server, uuid,resource_server, resource_port)
                if resource_server != "":
                    return node_service.get_gray3D( str(gray_name),
                                                    (size_z, size_y, size_x),
                                                    (subvolume.roi.z1-subvolume.border, subvolume.roi.y1-subvolume.border, subvolume.roi.x1-subvolume.border), throttle=False )
                else:
                    return node_service.get_gray3D( str(gray_name),
                                                    (size_z, size_y, size_x),
                                                    (subvolume.roi.z1-subvolume.border, subvolume.roi.y1-subvolume.border, subvolume.roi.x1-subvolume.border) )

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
            # get sizes of roi
            size_x = subvolume.roi.x2 + 2*subvolume.border - subvolume.roi.x1
            size_y = subvolume.roi.y2 + 2*subvolume.border - subvolume.roi.y1
            size_z = subvolume.roi.z2 + 2*subvolume.border - subvolume.roi.z1

            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_labels():
                # extract labels 64
                # retrieve data from roi start position considering border
                # Note: libdvid uses zyx order for python functions
                node_service = retrieve_node_service(server, uuid, resource_server, resource_port)
                if resource_server != "":
                    data = node_service.get_labels3D( str(label_name),
                                                      (size_z, size_y, size_x),
                                                      (subvolume.roi.z1-subvolume.border, subvolume.roi.y1-subvolume.border, subvolume.roi.x1-subvolume.border),
                                                      compress=True, throttle=False )
                else:
                    data = node_service.get_labels3D( str(label_name),
                                                      (size_z, size_y, size_x),
                                                      (subvolume.roi.z1-subvolume.border, subvolume.roi.y1-subvolume.border, subvolume.roi.x1-subvolume.border),
                                                      compress=True )

                # mask ROI
                if roiname != "":
                    mask_roi(data, subvolume, border=border)        

                return data
            return get_labels()
        return distrois.mapValues(mapper)

    
    def map_labels64_pair(self, distrois, label_name, dvidserver2, uuid2, label_name2, roiname=""):
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

        def mapper(subvolume):
            # get sizes of roi
            size_x = subvolume.roi.x2 - subvolume.roi.x1
            size_y = subvolume.roi.y2 - subvolume.roi.y1
            size_z = subvolume.roi.z2 - subvolume.roi.z1

            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_labels():
                # extract labels 64
                # retrieve data from roi start position
                # Note: libdvid uses zyx order for python functions
                node_service = retrieve_node_service(server, uuid, resource_server, resource_port)
                if resource_server != "":
                    data = node_service.get_labels3D( str(label_name),
                                                      (size_z, size_y, size_x),
                                                      (subvolume.roi.z1, subvolume.roi.y1, subvolume.roi.x1), throttle=False)
                else:
                    data = node_service.get_labels3D( str(label_name),
                                                      (size_z, size_y, size_x),
                                                      (subvolume.roi.z1, subvolume.roi.y1, subvolume.roi.x1))


                # mask ROI
                if roiname != "":
                    mask_roi(data, subvolume)        

                return data
            label_volume = get_labels()

            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def get_labels2():
                # fetch second label volume
                # retrieve data from roi start position
                # Note: libdvid uses zyx order for python functions
                node_service2 = retrieve_node_service(server2, uuid2, resource_server, resource_port)
                if resource_server != "":
                    return node_service2.get_labels3D( str(label_name2),
                                                       (size_z, size_y, size_x),
                                                       (subvolume.roi.z1, subvolume.roi.y1, subvolume.roi.x1), throttle=False)
                else:
                    return node_service2.get_labels3D( str(label_name2),
                                                       (size_z, size_y, size_x),
                                                       (subvolume.roi.z1, subvolume.roi.y1, subvolume.roi.x1))

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
            size1 = subvolume.roi.x2-subvolume.roi.x1
            size2 = subvolume.roi.y2-subvolume.roi.y1
            size3 = subvolume.roi.z2-subvolume.roi.z1

            border = subvolume.border

            # extract seg ignoring borders (z,y,x)
            seg = seg[border:size3+border, border:size2+border, border:size1+border]

            # copy the slice to make contiguous before sending 
            seg = numpy.copy(seg, order='C')

            @auto_retry(3, pause_between_tries=600.0, logging_name= __name__)
            def put_labels():
                # send data from roi start position
                # Note: libdvid uses zyx order for python functions
                node_service = retrieve_node_service(server, uuid, resource_server, resource_port)
                if roi_name is None:
                    node_service.put_labels3D( str(label_name),
                                               seg,
                                               (subvolume.roi.z1, subvolume.roi.y1, subvolume.roi.x1),
                                               compress=True,
                                               mutate=mutate )
                else: 
                    node_service.put_labels3D( str(label_name),
                                               seg,
                                               (subvolume.roi.z1, subvolume.roi.y1, subvolume.roi.x1),
                                               compress=True,
                                               roi=str(roi_name),
                                               mutate=mutate )
            put_labels()

        return seg_chunks.foreach(writer)

