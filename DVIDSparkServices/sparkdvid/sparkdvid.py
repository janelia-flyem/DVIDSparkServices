import json
from DVIDSparkServices.sparkdvid.Subvolume import Subvolume
from CompressedNumpyArray import CompressedNumpyArray

class sparkdvid(object):
    BLK_SIZE = 32
    
    def __init__(self, context, dvid_server, dvid_uuid):
        self.sc = context
        self.dvid_server = dvid_server
        self.uuid = dvid_uuid

    # Produce RDDs for each subvolume partition (this will replace default implementation)
    # Treats subvolum index as the RDD key and maximizes partition count for now
    # Assumes disjoint subsvolumes in ROI
    def parallelize_roi(self, roi, chunk_size, border=0, find_neighbors=False):
        # function will export and should include dependencies
        subvolumes = [] # x,y,z,x2,y2,z2

        from libdvid import DVIDNodeService, SubstackXYZ
        
        # extract roi for a given chunk size
        node_service = DVIDNodeService(str(self.dvid_server), str(self.uuid))
        substacks, packing_factor = node_service.get_roi_partition(str(roi), chunk_size / self.BLK_SIZE) 
       
        # create roi array giving unique substack ids
        for substack_id, substack in enumerate(substacks):
            # use substack id as key
            subvolumes.append((substack_id, Subvolume(substack_id, substack, chunk_size, border))) 
   

        # grab all neighbors for each substack
        if find_neighbors:
            # inefficient search for all boundaries
            for i in range(0, len(subvolumes)-1):
                for j in range(i+1, len(subvolumes)):
                    subvolumes[i][1].recordborder(subvolumes[j][1])

        # Potential TODO: custom partitioner for grouping close regions
        return self.sc.parallelize(subvolumes, len(subvolumes))



    # (key, ROI) => (key, ROI, grayscale)
    # ** Preserves partitioner 
    # Compression of numpy array is avoided since
    # lz4 will not be too effective on grayscale data
    def map_grayscale8(self, distsubvolumes, gray_name):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid

        # only grab value
        def mapper(subvolume):
            from libdvid import DVIDNodeService
            # extract grayscale x
            node_service = DVIDNodeService(str(server), str(uuid))
          
            # get sizes of subvolume
            size1 = subvolume.roi.x2+2*subvolume.border-subvolume.roi.x1
            size2 = subvolume.roi.y2+2*subvolume.border-subvolume.roi.y1
            size3 = subvolume.roi.z2+2*subvolume.border-subvolume.roi.z1

            # retrieve data from roi start position considering border 
            gray_volume = node_service.get_gray3D( str(gray_name), (size1,size2,size3), (subvolume.roi.x1-subvolume.border, subvolume.roi.y1-subvolume.border, subvolume.roi.z1-subvolume.border) )

            # flip to be in C-order (no performance penalty)
            gray_volume = gray_volume.transpose((2,1,0))
            
            return (subvolume, gray_volume)

        return distsubvolumes.mapValues(mapper)


    # produce mapped ROI RDDs
    def map_labels64(self, distrois, label_name, border, roiname=""):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid

        def mapper(subvolume):
            from libdvid import DVIDNodeService
            # extract labels 64
            node_service = DVIDNodeService(str(server), str(uuid))
          
            # get sizes of roi
            size1 = subvolume.roi[3]+2*border-subvolume.roi[0]
            size2 = subvolume.roi[4]+2*border-subvolume.roi[1]
            size3 = subvolume.roi[5]+2*border-subvolume.roi[2]

            # retrieve data from roi start position considering border
            label_volume = node_service.get_labels3D( str(label_name), (size1,size2,size3), (subvolume.roi[0]-border, subvolume.roi[1]-border, subvolume.roi[2]-border), compress=True, roi=str(roiname) )

            # flip to be in C-order (no performance penalty)
            label_volume = label_volume.transpose((2,1,0))
            
            return label_volume

        return distrois.mapValues(mapper)

    # produce mapped ROI RDDs
    def map_labels64_pair(self, distrois, label_name, dvidserver2, uuid2, label_name2, roiname=""):
        # copy local context to minimize sent data
        server = self.dvid_server
        server2 = dvidserver2
        uuid = self.uuid

        def mapper(subvolume):
            from libdvid import DVIDNodeService
            
            # extract labels 64
            node_service = DVIDNodeService(str(server), str(uuid))

            # get sizes of roi
            size1 = subvolume.roi[3]-subvolume.roi[0]
            size2 = subvolume.roi[4]-subvolume.roi[1]
            size3 = subvolume.roi[5]-subvolume.roi[2]

            # retrieve data from roi start position
            label_volume = node_service.get_labels3D(str(label_name),
                (size1,size2,size3),
                (subvolume.roi[0], subvolume.roi[1],
                subvolume. roi[2]), roi=str(roiname))

            # flip to be in C-order (no performance penalty)
            label_volume = label_volume.transpose((2,1,0))
            
            # fetch second label volume
            node_service2 = DVIDNodeService(str(server2), str(uuid2))
 
            # retrieve data from roi start position
            label_volume2 = node_service2.get_labels3D(str(label_name2),
                (size1,size2,size3),
                (subvolume.roi[0], subvolume.roi[1],
                subvolume. roi[2]))

            # flip to be in C-order (no performance penalty)
            label_volume2 = label_volume2.transpose((2,1,0))

            # zero out label_volume2 where GT is 0'd out !!
            label_volume2[label_volume==0] = 0

            return (subvolume, CompressedNumpyArray(label_volume),
                               CompressedNumpyArray(label_volume2))

        return distrois.mapValues(mapper)


    # foreach will write graph elements to DVID storage
    def foreachPartition_graph_elements(self, elements, graph_name):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid
        
        def writer(element_pairs):
            from libdvid import DVIDNodeService, Vertex, Edge
            
            # write graph information
            node_service = DVIDNodeService(str(server), str(uuid))
       
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
    
            if len(vertices) > 0:
                node_service.update_vertices(str(graph_name), vertices) 
            
            if len(edges) > 0:
                node_service.update_edges(str(graph_name), edges) 
            
            return []

        elements.foreachPartition(writer)

    # (key, (ROI, segmentation compressed+border))
    # => segmentation output in DVID
    def foreach_write_labels3d(self, label_name, seg_chunks, roi_name=None):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid

        from libdvid import DVIDNodeService
        # create labels type
        node_service = DVIDNodeService(str(server), str(uuid))
        node_service.create_labelblk(str(label_name))

        def writer(subvolume_seg):
            from libdvid import DVIDNodeService
            import numpy
            # write segmentation
            node_service = DVIDNodeService(str(server), str(uuid))
            
            (key, (subvolume, segcomp)) = subvolume_seg
            # uncompress data
            seg = segcomp.deserialize() 

            # get sizes of subvolume 
            size1 = subvolume.roi.x2-subvolume.roi.x1
            size2 = subvolume.roi.y2-subvolume.roi.y1
            size3 = subvolume.roi.z2-subvolume.roi.z1

            border = subvolume.border

            # extract seg ignoring borders (z,y,x)
            seg = seg[border:size3+border, border:size2+border, border:size1+border]

            # put in x,y,z and send (copy the slice to make contiguous) 
            seg = numpy.copy(seg.transpose((2,1,0)))
            # send data from roi start position
            node_service.put_labels3D(str(label_name), seg, (subvolume.roi.x1, subvolume.roi.y1, subvolume.roi.z1), compress=True, roi=str(roi_name))

        return seg_chunks.foreach(writer)

