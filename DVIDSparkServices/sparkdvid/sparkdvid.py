import json


class sparkdvid(object):
    BLK_SIZE = 32
    
    def __init__(self, context, dvid_server, dvid_uuid):
        self.sc = context
        self.dvid_server = dvid_server
        self.uuid = dvid_uuid

    # Produce RDDs for each subvolume partition (this will replace default implementation)
    # Treats subvolum index as the RDD key and maximizes partition count for now
    # Assumes disjoint subsvolumes in ROI
    def parallelize_roi_new(self, roi, chunk_size, border, find_neighbors=True):
        # function will export and should include dependencies
        subvolumes = [] # x,y,z,x2,y2,z2

        from libdvid import DVIDNodeService, SubstackXYZ
        
        # extract roi for a given chunk size
        node_service = DVIDNodeService(str(self.dvid_server), str(self.uuid))
        substacks, packing_factor = node_service.get_roi_partition(str(roi), chunk_size / BLK_SIZE) 
        
        # create roi array giving unique substack ids
        substack_id = 0
        for substack in substacks:
            # use substack id as key
            subvolumes.append((substack_id, Subvolume(substack_id, substack, chunk_size, border))) 
            substack_id += 1
    
        # grab all neighbors for each substack
        if find_neighbors:
            # inefficient search for all boundaries
            for i in range(0, len(subvolumes)-1):
                for j in range(i+1, len(subvolumes)):
                    subvolumes[i].recordborder(subvolumes[j], border)

        # Potential TODO: custom partitioner for grouping close regions
        return self.sc.parallelize(subvolumes, len(subvolumes))

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


    # (key, ROI) => (key, ROI, grayscale)
    # ** Preserves partitioner 
    # Compression of numpy array is avoided since
    # lz4 will not be too effective on grayscale data
    def map_grayscale8(self, distsubvolumes, gray_name, border=0):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid

        def mapper(subvolume_key_value):
            key, subvolume = subvolume_key_value

            from libdvid import DVIDNodeService
            # extract grayscale x
            node_service = DVIDNodeService(str(server), str(uuid))
          
            # get sizes of subvolume
            size1 = subvolume.roi.x2+2*border-subvolume.roi.x1
            size2 = subvolume.roi.y2+2*border-subvolume.roi.y1
            size3 = subvolume.roi.z2+2*border-subvolume.roi.z1

            # retrieve data from roi start position considering border 
            gray_volume = node_service.get_gray3D( str(gray_name), (size1,size2,size3), (subvolume.roi.x1-border, subvolume.roi.y1-border, subvolume.roi.z1-border) )

            # flip to be in C-order (no performance penalty)
            gray_volume = gray_volume.transpose((2,1,0))
            
            return (key, (subvolume, gray_volume))

        return distsubvolumes.mapValues(mapper)


    # produce mapped ROI RDDs
    def map_labels64(self, distrois, label_name, border):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid

        def mapper(roi):
            from libdvid import DVIDNodeService
            # extract labels 64
            node_service = DVIDNodeService(str(server), str(uuid))
          
            # get sizes of roi
            size1 = roi[3]+2*border-roi[0]
            size2 = roi[4]+2*border-roi[1]
            size3 = roi[5]+2*border-roi[2]

            # retrieve data from roi start position considering border
            label_volume = node_service.get_labels3D( str(label_name), (size1,size2,size3), (roi[0]-border, roi[1]-border, roi[2]-border) )

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

    # (key, (ROI, segmentation compressed+border))
    # => segmentation output in DVID
    def foreachPartition_write_labels3d(self, label_name, seg_chunks, border = 0):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid

        # create labels type
        node_service = DVIDNodeService(label_name, uuid)
        node_service.create_labels3d(label_name)

        def writer(subvolume_seg):
            from libdvid import DVIDNodeService
            # write segmentation
            node_service = DVIDNodeService(str(server), str(uuid))
            
            (key, subvolume, segcomp) = subvolume_seg
            # uncompress data
            seg = segcomp.decompress() 

            # get sizes of subvolume 
            size1 = subvolume.roi.x2-subvolume.roi.x1
            size2 = subvolume.roi.y2-subvolume.roi.y1
            size3 = subvolume.roi.z2-subvolume.roi.z1

            # extract seg ignoring borders (z,y,x)
            seg = seg[border:size3+border, border:size2+border, border:size1+border]

            # put in x,y,z and send
            seg = seg.transpose((2,1,0))
             
            # send data from roi start position
            node_service.put_labels3D(label_name, (size1,size2,size3), (subvolume.roi.x1, subvolume.roi.y1, subvolume.roi.z1))

        return distrois.foreach(writer)

