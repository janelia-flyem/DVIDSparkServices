import struct

class SimpleGraph(object):
    def __init__(self, config):
        if "graph-builder-exe" in config:
            self.external_prog = config["graph-builder-exe"]
        else:
            self.external_prog = ""

    def vol2string(self, label_chunk):
        x,y,z = label_chunk.shape

        # pack little endian 64-bit coordinate numbers
        coord_data = b'<QQQ'
        coord_bin = struct.pack(coord_data, x, y, z)
        return bytes(coord_bin) + label_chunk.tobytes()

    def string2graph(self, graphstr):
        offset = 0
        
        # get number of vertices
        num_verts,  = struct.unpack_from('<Q', graphstr, offset)
        offset += 8
        elements = []


        # weight should be a double
        for vertex in range(0, num_verts):
            vert1, weight = struct.unpack_from('<Qd', graphstr, offset)
            offset += 16
            elements.append(((vert1, -1), weight)) 

        # get number of edges
        num_edges, = struct.unpack_from('<Q', graphstr, offset)
        offset += 8
        

        # weight should be a double
        for edge in range(0, num_edges):
            vert1, vert2, weight = struct.unpack_from('<QQd', graphstr, offset)
            offset += 24
            elements.append(((vert1, vert2), weight)) 

        return elements


    def build_graph(self, key_label_chunk):
        if self.external_prog == "":
            # TODO: default slow python implementation ??
            #return []
            import numpy
            vertices = numpy.unique(key_label_chunk[1])
           
            # just add a vertex of size 1 for each
            elements = []
            for iter1 in range(len(vertices)):
                vert = vertices[iter1]
                elements.append(((vert, -1), 1))
                #if ((vert % 50) == 0) and (iter1 < (len(vertices)-1)):
                #    elements.append(((vert, vertices[iter1+1]), 3))

            return elements
        else:
            # call program that expects label stdin and produces graph stdout
            from subprocess import Popen, PIPE, STDOUT
            p = Popen([self.external_prog], stdout=PIPE, stdin=PIPE, stderr=PIPE)
            graphstr = p.communicate(input=self.vol2string(key_label_chunk[1]))[0]
            return self.string2graph(graphstr) 
        
    def is_vertex(self, element_pair):
        edge, value = element_pair
        v1, v2 = edge
        if v2 == -1:
            return True
        return False

    def is_edge(self, element_pair):
        edge, value = element_pair
        v1, v2 = edge
        if v2 == -1:
            return False
        return True

