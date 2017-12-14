"""Framework for generating affinity between nodes in a large segmentation graph.

Source segmentation and grayscale is divided into subvolumes (grayscale will have surrounding
context).  Voxel prediction is performed on grayscale.  Features between
source segmentation given this voxel prediction is shuffled around to allow affinity
to be computed globally for each edge.  Current edge features and affinity computation
is done by NeuroProof.  A resulting graph with voxel size and edge
weight as affinity is pushed to either DVID labelgraph or to disk.

"""
from __future__ import print_function, absolute_import
from __future__ import division
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
import DVIDSparkServices
from functools import partial
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.util import NumpyConvertingEncoder

class ComputeEdgeProbs(DVIDWorkflow):
    # schema for creating segmentation
    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to compute affinity between image segments given grayscale data and labels",
      "type": "object",
      "properties": {
        "dvid-info" : {
          "type": "object",
          "properties": {
            "dvid-server": {
              "description": "location of DVID server",
              "type": "string",
              "minLength": 1
            },
            "uuid": {
              "description": "dvid version node",
              "type": "string",
              "minLength": 1
            },
            "roi": {
              "description": "region of interest",
              "type": "string",
              "minLength": 1
            },
            "grayscale": {
              "description": "grayscale data",
              "type": "string",
              "minLength": 1,
              "default": "grayscale"
            },
            "segmentation-name": {
              "description": "intial segmentation (only 32 bit)",
              "type": "string",
              "minLength": 1
            },
            "graph-name": {
              "description": "destination name of DVID labelgraph",
              "type": "string" 
            }
          },
          "required": ["dvid-server", "uuid", "roi", "grayscale", "segmentation-name", "graph-name"],
          "additionalProperties": False
        },
        "options" : {
          "type": "object",
          "properties": {
            "predict-voxels": {
              "description": "Custom configuration for voxel prediction.",
              "type": "object",
              "properties": {
                "function": {
                  "description": "function that implements voxel prediction",
                  "type": "string"
                },
                "parameters": {
                  "description": "parameters for voxel prediction function",
                  "type" : "object",
                  "default" : {}
                }
              }
            },
            "segment-classifier": {
              "description": "Classifier to predict confidence between segments (currently must be directory path)",
              "type": "string"
            },
            "output-file": { 
              "description": "File where graph will be written",
              "type": "string" 
            },
            "checkpoint-dir": {
                "description": "Specify checkpoint directory",
                "type": "string",
                "default": ""
            },
            "chunk-size": {
              "description": "Size of blocks to process independently",
              "type": "integer",
              "default": 256
            },
            "iteration-size": {
              "description": "Number of tasks per iteration (0 -- max size)",
              "type": "integer",
              "default": 0
            },
            "checkpoint": {
              "description": "Reuse previous edge features computed",
              "type": "boolean",
              "default": False
            },
            "debug": {
              "description": "Enable certain debugging functionality.  Mandatory for integration tests.",
              "type": "boolean",
              "default": False
            }
          },
          "required": ["predict-voxels", "segment-classifier"]
        }
      }
    }

    @classmethod
    def schema(cls):
        return ComputeEdgeProbs.Schema

    # assume blocks are 32x32x32
    blocksize = 32

    # context for each subvoume 
    # For convenience, we use the same default overlap that CreateSegmentation uses,
    # so the same cached probabilities can be used if they're available.
    contextbuffer = 20

    def __init__(self, config_filename):
        super(ComputeEdgeProbs, self).__init__(config_filename, self.schema(), "ComputeEdgeProbs")

    def execute(self):
        # TODO: handle 64 bit segmentation

        from pyspark import SparkContext
        from pyspark import StorageLevel
        from DVIDSparkServices.reconutils.Segmentor import Segmentor

        self.chunksize = self.config_data["options"]["chunk-size"]

        # create datatype in the beginning
        node_service = retrieve_node_service(self.config_data["dvid-info"]["dvid-server"], 
                self.config_data["dvid-info"]["uuid"], self.resource_server, self.resource_port)
        
        # grab ROI subvolumes and find neighbors
        distsubvolumes = self.sparkdvid_context.parallelize_roi(
                self.config_data["dvid-info"]["roi"],
                self.chunksize, self.contextbuffer, True)

        contextbuffer = self.contextbuffer
        # do not recompute ROI for each iteration
        distsubvolumes.persist()

        # instantiate the voxel prediction plugin
        import importlib
        full_function_name = self.config_data["options"]["predict-voxels"]["function"]
        module_name = '.'.join(full_function_name.split('.')[:-1])
        function_name = full_function_name.split('.')[-1]
        module = importlib.import_module(module_name)
        
        parameters = self.config_data["options"]["predict-voxels"]["parameters"]
        vprediction_function = partial( getattr(module, function_name), **parameters )

        # determine number of iterations
        num_parts = len(distsubvolumes.collect())
        iteration_size = self.config_data["options"]["iteration-size"]
        if iteration_size == 0:
            iteration_size = num_parts

        num_iters = num_parts // iteration_size
        if num_parts % iteration_size > 0:
            num_iters += 1

        feature_chunk_list = []

        # enable checkpointing if not empty
        checkpoint_dir = self.config_data["options"]["checkpoint-dir"]

        # enable rollback of iterations if necessary
        rollback = False
        if self.config_data["options"]["checkpoint"]:
            rollback = True
       
        for iternum in range(0, num_iters):
            # it might make sense to randomly map partitions for selection
            # in case something pathological is happening -- if original partitioner
            # is randomish than this should be fine
            def subset_part(sid_data):
                (s_id, _data) = sid_data
                if (s_id % num_iters) == iternum:
                    return True
                return False
            
            # should preserve partitioner
            distsubvolumes_part = distsubvolumes.filter(subset_part)

            # get grayscale chunks with specified overlap
            gray_chunks = self.sparkdvid_context.map_grayscale8(distsubvolumes_part,
                    self.config_data["dvid-info"]["grayscale"])

            pred_checkpoint_dir = ""
            if checkpoint_dir:
                pred_checkpoint_dir = checkpoint_dir + "/prediter-" + str(iternum)

            # For now, we always read predictions if available, and always write them if not.
            # TODO: Add config settings to control read/write behavior.
            @Segmentor.use_block_cache(pred_checkpoint_dir, allow_read=True, allow_write=True)
            def predict_voxels( sv_gray ):
                (_subvolume, gray) = sv_gray
                return vprediction_function(gray, None)

            vox_preds = gray_chunks.values().map( predict_voxels ) # predictions only
            vox_preds = distsubvolumes_part.values().zip( vox_preds ) # (subvolume, predictions)

            pdconf = self.config_data["dvid-info"]
            resource_server = self.resource_server
            resource_port = self.resource_port

            # retrieve segmentation and generate features
            def generate_features(vox_pred):
                import numpy
                (subvolume, pred) = vox_pred
                pred = numpy.ascontiguousarray(pred)


                # extract labelblks
                border = 1 # only one pixel needed to find edges
                
                # get sizes of box
                size_z = subvolume.box.z2 + 2*border - subvolume.box.z1
                size_y = subvolume.box.y2 + 2*border - subvolume.box.y1
                size_x = subvolume.box.x2 + 2*border - subvolume.box.x1

                # retrieve data from box start position considering border
                # !! technically ROI is not respected but unwritten segmentation will be ignored since it will have 0-valued pixels.
                @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
                def get_seg():
                    node_service = retrieve_node_service(pdconf["dvid-server"], 
                            pdconf["uuid"], resource_server, resource_port)
                    # retrieve data from box start position
                    # Note: libdvid uses zyx order for python functions
                    
                    if resource_server != "": 
                        return node_service.get_labels3D(str(pdconf["segmentation-name"]),
                            (size_z, size_y, size_x),
                            (subvolume.box.z2-border, subvolume.box.y1-border, subvolume.box.x1-border))
                    else:
                        return node_service.get_labels3D(str(pdconf["segmentation-name"]),
                             (size_z, size_y, size_x),
                             (subvolume.box.z2-border, subvolume.box.y1-border, subvolume.box.x1-border))

                initial_seg = get_seg()

                # !!! potentially dangerous but needed for now
                initial_seg = initial_seg.astype(numpy.uint32)

                pred2 = pred[(contextbuffer-border):-(contextbuffer-border), (contextbuffer-border):-(contextbuffer-border), (contextbuffer-border):-(contextbuffer-border), :].copy()
                z,y,x,num_chans = pred2.shape

                # call neuroproof and generate features
                from neuroproof import FocusedProofreading 
                # "edges": [ edge ] where edge = [node1, node2, edgesize, all features...]
                # "vertices": [vertex ] where vertex = [id, size, all features...]
                features = FocusedProofreading.extract_features(initial_seg, pred2) 
                
                element_list = []
                # iterate edges and create ((node1, node2), features)
                if "Edges" in features:
                    # could have only one vertex in a partition and no edges
                    for edge in features["Edges"]:
                        n1 = edge["Id1"]
                        n2 = edge["Id2"]
                        edge["Loc1"][0] += subvolume.box.x1
                        edge["Loc1"][1] += subvolume.box.y1
                        edge["Loc1"][2] += subvolume.box.z1
                        
                        edge["Loc2"][0] += subvolume.box.x1
                        edge["Loc2"][1] += subvolume.box.y1
                        edge["Loc2"][2] += subvolume.box.z1
                        
                        if n1 > n2:
                            n1, n2 = n2, n1
                        element_list.append(((n1,n2), (num_chans, edge)))

                for node in features["Vertices"]:
                    n1 = node["Id"]
                    element_list.append(((n1,-1), (num_chans, node)))

                return element_list 

            features = vox_preds.flatMap(generate_features)

            # retrieve previously computed RDD or save current RDD
            if checkpoint_dir != "":
                features = self.sparkdvid_context.checkpointRDD(features, 
                        checkpoint_dir + "/featureiter-" + str(iternum), rollback)  

            # any forced persistence will result in costly
            # pickling, lz4 compressed numpy array should help
            features.persist(StorageLevel.MEMORY_AND_DISK_SER)

            feature_chunk_list.append(features)

        features = feature_chunk_list[0]

        for iter1 in range(1, len(feature_chunk_list)):
            # this could cause a serialization problems if there are a large number of iterations (>100)
            features = feature.union(feature_chunk_list[iter1])
    
        # grab num channels from boundary prediction
        features.persist(StorageLevel.MEMORY_AND_DISK_SER)
        first_feature = features.first()
        (key1, key2), (num_channels, foo) = first_feature

        # remove num channels from features
        def remove_num_channels(featurepair):
            foo, feature = featurepair
            return feature
        features = features.mapValues(remove_num_channels)
       
        import json

        # merge edge and node features -- does not require reading classifier
        # node features are encoded as (vertex id, -1)
        def combine_edge_features(element1, element2):
            from neuroproof import FocusedProofreading
            
            if "Id2" in element1:
                # are edges
                return FocusedProofreading.combine_edge_features( json.dumps(element1, cls=NumpyConvertingEncoder),
                                                                  json.dumps(element2, cls=NumpyConvertingEncoder),
                                                                  num_channels )
            else:
                # are vertices
                return FocusedProofreading.combine_vertex_features( json.dumps(element1, cls=NumpyConvertingEncoder),
                                                                    json.dumps(element2, cls=NumpyConvertingEncoder),
                                                                    num_channels )

        features_combined = features.reduceByKey(combine_edge_features)
     
        #features_combined.persist()
        # TODO: option to serialize features to enable other analyses
       
        # join node and edge probs
        def retrieve_nodes(val):
            (n1,n2),features = val
            if n2 == -1:
                return True
            return False

        def retrieve_edges(val):
            (n1,n2),features = val
            if n2 == -1:
                return False
            return True

        node_features = features_combined.filter(retrieve_nodes)
        edge_features = features_combined.filter(retrieve_edges)

       
        node_features = node_features.map(lambda x: (x[0][0], x[1]))
        edge1_features = edge_features.map(lambda x: (x[0][0], x[1]))
        edge2_features = edge_features.map(lambda x: (x[0][1], x[1]))

        # multiple edges with the same key
        edge1_node_features = edge1_features.leftOuterJoin(node_features)
        edge2_node_features = edge2_features.leftOuterJoin(node_features)

        def reset_edgekey(val):
            key, (edge, node) = val
            n1 = edge["Id1"]
            n2 = edge["Id2"]
            if n1 > n2:
                n1, n2 = n2, n1
            return ((n1,n2), (edge, node))

        edge1_node_features = edge1_node_features.map(reset_edgekey)
        edge2_node_features = edge2_node_features.map(reset_edgekey)

        edge_node_features = edge1_node_features.join(edge2_node_features)

        # generate prob for each edge (JSON: body sizes, edge list with prob)
        classifierlocation = self.config_data["options"]["segment-classifier"]
        def compute_prob(edge_node_features):
            from neuroproof import FocusedProofreading 
            classifier = FocusedProofreading.ComputeProb(str(classifierlocation), num_channels) 
            
            res_list = []
            for edge_node_edge_node in edge_node_features:
                edge_key, ((edge, node1), (edge_dummy, node2)) = edge_node_edge_node
                weight = classifier.compute_prob( json.dumps(edge, cls=NumpyConvertingEncoder),
                                                  json.dumps(node1, cls=NumpyConvertingEncoder),
                                                  json.dumps(node2, cls=NumpyConvertingEncoder) )
                # node1, node2
                res_list.append((int(node1["Id"]),int(node2["Id"]),int(node1["Weight"]),int(node2["Weight"]),int(edge["Weight"]),weight,edge["Loc1"], edge["Loc2"]))

            return res_list

        # avoid loading large classifier for each small edge
        allprobs = edge_node_features.mapPartitions(compute_prob)
    
        # collect all edges and send to DVID (TODO: add option to dump to disk) 
        allprobs_combined = allprobs.collect()

        bodyinfo = {}
        edges = []

        for edge_info in allprobs_combined:
            node1, node2, node1_size, node2_size, edge_size, weight, loc1, loc2 = edge_info
            bodyinfo[node1] = node1_size            
            bodyinfo[node2] = node2_size            
            edges.append({"Id1": node1, "Id2": node2, "Weight": weight, "Loc1": loc1, "Loc2": loc2})
        
        bodies = []
        for (key, val) in bodyinfo.items():
            bodies.append({"Id": key, "Weight": val})

        graph = {}
        graph["Vertices"] = bodies
        graph["Edges"] = edges

        SAVE_TO_FILE = False
        if SAVE_TO_FILE:
            graph_filepath = '/tmp/graph-output.json'
            with open(graph_filepath, 'w') as f:
                self.workflow_entry_exit_printer.warn("Writing graph json to file:\n{}".format(graph_filepath))
                import json
                json.dump(graph, f, indent=4, separators=(',', ': '), cls=NumpyConvertingEncoder)
            self.workflow_entry_exit_printer.write_data("Wrote graph to disk") # write to logger after spark job

        UPLOAD_TO_DVID = True
        if UPLOAD_TO_DVID:
            import requests
            # load entire graph into DVID
            node_service.create_graph(str(self.config_data["dvid-info"]["graph-name"]))
            server = str(self.config_data["dvid-info"]["dvid-server"])
            #if not server.startswith("http://"):
            #    server = "http://" + server
            #requests.post(server + "/api/node/" + str(self.config_data["dvid-info"]["uuid"]) + "/" + str(self.config_data["dvid-info"]["graph-name"]) + "/subgraph", json=graph)
            #self.workflow_entry_exit_printer.write_data("Wrote DVID graph") # write to logger after spark job


        if self.config_data["options"]["debug"]:
            import json
            print("DEBUG:", json.dumps(graph, cls=NumpyConvertingEncoder))
     
        # write dvid to specified file (if provided)
        if "output-file" in self.config_data["options"] and self.config_data["options"]["output-file"] != "":
            filename = self.config_data["options"]["output-file"] 

            edgelist = []
            for edge in graph["Edges"]:
                edgelist.append({"node1": edge["Id1"], "node2": edge["Id2"], "weight": edge["Weight"], "loc1": edge["Loc1"], "loc2": edge["Loc2"]})

            npgraph = {}
            npgraph["edge_list"] = edgelist
            fout = open(filename, 'w')
            fout.write(json.dumps(npgraph, cls=NumpyConvertingEncoder))
