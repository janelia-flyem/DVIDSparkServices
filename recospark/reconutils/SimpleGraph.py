import dill

class SimpleGraph(object):
    def __init__(self, config):
        if "graph-builder-exe" in config:
            self.external_prog = config["graph-builder-exe"]
        else:
            self.external_prog = ""

    def retrieve_schema(self):
        pass
        # TODO

    def build_graph(self, label_chunk):
        # ?! pipe numpy array to string or temporary file

        #data = numpy.getbuffer(label_volume)
        #return str(data)

        # ?! read executable back from string

        # make a dummy vertex and edge results for now
        import numpy
        vertices = numpy.unique(label_chunk)
        
        elements = []
        for iter1 in range(len(vertices)):
            vert = vertices[iter1]
            elements.append(((vert, -1), 1))
            if ((vert % 50) == 0) and (iter1 < (len(vertices)-1)):
                elements.append(((vert, vertices[iter1+1]), 3))

        return elements

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

