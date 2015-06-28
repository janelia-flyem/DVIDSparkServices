import json

BLK_SIZE = 32

class sparkdvid(object):
    def __init__(self, context, dvid_server, dvid_uuid):
        self.sc = context
        self.dvid_server = dvid_server
        self.uuid = dvid_uuid

    # produce RDDs for each ROI partition -- should there be an option for number of slices?
    def parallelize_roi(self, roi, chunk_size):
        # function will export and should include dependencies
        rois = [] # x,y,z,x2,y2,z2

        from libdvid import DVIDNodeService, SubstackXYZ
        
        # extract roi
        node_service = DVIDNodeService(str(self.dvid_server), str(self.uuid))

        substacks, packing_factor = node_service.get_roi_partition(str(roi), chunk_size / BLK_SIZE) 
        for substack in substacks:
            rois.append((substack[0], substack[1], substack[2],
                    substack[0] + chunk_size,
                    substack[1] + chunk_size,
                    substack[2] + chunk_size)) 

        return self.sc.parallelize(rois, len(rois))


    # produce mapped ROI RDDs
    def map_labels64(self, distrois, label_name, overlap):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid

        def mapper(roi):
            from libdvid import DVIDNodeService
            # extract labels 64
            node_service = DVIDNodeService(str(server), str(uuid))
          
            # get sizes of roi
            size1 = roi[3]+2*overlap-roi[0]
            size2 = roi[4]+2*overlap-roi[1]
            size3 = roi[5]+2*overlap-roi[2]

            # retrieve data from roi start position considering overlap
            label_volume = node_service.get_labels3D( str(label_name), (size1,size2,size3), (roi[0]-overlap, roi[1]-overlap, roi[2]-overlap) )

            # flip to be in C-order (no performance penalty)
            label_volume = label_volume.transpose((2,1,0))
            
            return label_volume

        return distrois.map(mapper)

    # produce mapped ROI RDDs
    def map_labels64_pair(self, distrois, label_name, dvidserver2, uuid2, label_name2):
        # copy local context to minimize sent data
        server = self.dvid_server
        server2 = dvidserver2
        uuid = self.uuid

        def mapper(roi):
            from libdvid import DVIDNodeService
            
            # extract labels 64
            node_service = DVIDNodeService(str(server), str(uuid))

            # get sizes of roi
            size1 = roi[3]-roi[0]
            size2 = roi[4]-roi[1]
            size3 = roi[5]-roi[2]

            # retrieve data from roi start position
            label_volume = node_service.get_labels3D( str(label_name), (size1,size2,size3), (roi[0], roi[1], roi[2]) )

            # flip to be in C-order (no performance penalty)
            label_volume = label_volume.transpose((2,1,0))
            
            # fetch second label volume
            node_service2 = DVIDNodeService(str(server2), str(uuid2))
 
            # retrieve data from roi start position
            label_volume2 = node_service2.get_labels3D( str(label_name2), (size1,size2,size3), (roi[0], roi[1], roi[2]) )

            # flip to be in C-order (no performance penalty)
            label_volume2 = label_volume2.transpose((2,1,0))

            pt1 = (roi[0], roi[1], roi[2])
            pt2 = (roi[3], roi[4], roi[5])


            return (pt1, pt2, label_volume, label_volume2)

        return distrois.map(mapper)


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

