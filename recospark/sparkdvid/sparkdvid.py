import requests
import json
import dill

class sparkdvid(object):
    def __init__(self, context, dvid_server, dvid_uuid):
        self.sc = context
        self.dvid_server = dvid_server
        self.uuid = dvid_uuid

    # produce RDDs for each ROI partition -- should there be an option for number of slices?
    def parallelize_roi(self, roi, chunk_size):
        # function will export and should include dependencies
        rois = [] # x,y,z,x2,y2,z2

        # extract roi
        r = requests.get("http://" + self.dvid_server + "/api/node/" + self.uuid + "/" + 
                roi + "/partition?batchsize=" + str(chunk_size/32)) 
        substack_data = r.json()
        for subvolume in substack_data["Subvolumes"]:
            substack = subvolume["MinPoint"]
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
            import httplib
            from pydvid.voxels import VoxelsAccessor
            import numpy

            conn = httplib.HTTPConnection(server)
            labels_access = VoxelsAccessor(conn, uuid, label_name,
                    throttle=True, retry_timeout=1800.0)
            
            pt1 = (0, roi[0]-overlap, roi[1]-overlap, roi[2]-overlap)
            pt2 = (1, roi[3]+overlap, roi[4]+overlap, roi[5]+overlap)
            
            label_volume = labels_access.get_ndarray( pt1, pt2 )
            label_volume = label_volume.transpose((3,2,1,0)).squeeze()
            
            # convert to RDD string and return -- always assume little endian!
            #data = numpy.getbuffer(label_volume)
            #return str(data)
            return label_volume

        return distrois.map(mapper)


    # foreach will write graph elements to DVID storage
    def foreach_graph_elements(self, elements, graph_name):
        # copy local context to minimize sent data
        server = self.dvid_server
        uuid = self.uuid
        
        def writer(element_pair):
            import httplib
            from pydvid.labelgraph import labelgraph
            conn = httplib.HTTPConnection(server)

            edge, weight = element_pair
            v1, v2 = edge

            if v2 == -1:
                labelgraph.update_vertex(conn, uuid, graph_name, int(v1), int(weight)) 
            else:
                labelgraph.update_edge(conn, uuid, graph_name, int(v1), int(v2), int(weight)) 

        elements.foreach(writer)

