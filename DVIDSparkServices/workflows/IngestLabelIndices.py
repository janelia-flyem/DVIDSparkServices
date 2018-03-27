import os
import datetime
import logging

import numpy as np
import pandas as pd

from dvidutils import LabelMapper
from dvid_resource_manager.client import ResourceManagerClient

import DVIDSparkServices.rddtools as rt
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.util import Timer, default_dvid_session

from DVIDSparkServices.io_util.volume_service import DvidSegmentationServiceSchema, LabelMapSchema
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap
from DVIDSparkServices.dvid.ingest_label_indexes import ingest_mapping
from DVIDSparkServices.dvid.metadata import DataInstance

# The labelops_pb2 file was generated with the following commands:
# $ cd DVIDSparkServices/dvid
# $ protoc --python_out=. labelops.proto
# $ sed -i '' s/labelops_pb2/DVIDSparkServices.dvid.labelops_pb2/g labelops_pb2.py
from DVIDSparkServices.dvid.labelops_pb2 import LabelIndex, LabelIndices

logger = logging.getLogger(__name__)


class IngestLabelIndices(Workflow):
    
    Schema = \
    {
        "$schema": "http://json-schema.org/schema#",
        "title": "Service to load label indexes and mappings into a dvid 'labelmap' instance.",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dvid": DvidSegmentationServiceSchema,
            "block-stats-file": {
                "description": "CSV file containing block supervoxel statistics to load",
                "type": "string"
            },
            "agglomeration-mapping": LabelMapSchema,
            "options": {
                "csv-chunk-size": {
                    "description": "How many lines of CSV to distribute to each RDD element at the beginning of the workflow.\n",
                    "type": "integer",
                    "default": 10_000_000
                },
                "operation": {
                    "description": "Whether to load labelindices, mappings, or both.",
                    "type": "string",
                    "enum": ["labelindices", "mappings", "both"],
                    "default": "labelindices"
                },
                "mutation-id": {
                    "description": "The mutation ID to use for these ingestions. By default, uses 0 for supervoxels (no agglomeration) and 1 otherwise.",
                    "type": "integer",
                    "default": -1
                }
            },
        }
    }

    @classmethod
    def schema(cls):
        return IngestLabelIndices.Schema

    # name of application for DVID queries
    APPNAME = "ingestlabelindices"


    def __init__(self, config_filename):
        super(IngestLabelIndices, self).__init__( config_filename,
                                                  IngestLabelIndices.schema(),
                                                  "Ingest Label Indices" )
    
    def _sanitize_config(self):
        config = self.config_data

        server = config["dvid"]["server"]
        if not server.startswith("http://"):
            server = "http://" + server
            config["dvid"]["server"] = server

        mutid = config["options"]["mutation-id"]
        if mutid == -1:
            if config["agglomeration-mapping"]["file-type"] == "__invalid__":
                mutid = 0
            else:
                mutid = 1
            config["options"]["mutation-id"] = mutid

        config['block-stats-file'] = self.relpath_to_abspath(config['block-stats-file'])

        instance_info = DataInstance(server, config["dvid"]["uuid"], config["dvid"]["segmentation-name"])
        if instance_info.datatype != 'labelmap':
            raise RuntimeError(f'DVID instance is not a labelmap: {config["dvid"]["segmentation-name"]}')
        
    
    def execute(self):
        self._sanitize_config()
        config = self.config_data
        options = config["options"]
        
        ##
        ## Load mapping file (if any)
        ##
        if config["agglomeration-mapping"]["file-type"] == "__invalid__":
            mapping_df = None
        else:
            mapping_pairs = load_labelmap(config["agglomeration-mapping"], self.config_dir)
            if mapping_pairs.max() < 2**32:
                mapping_pairs = mapping_pairs.astype(np.uint32)
            mapping_df = pd.DataFrame(mapping_pairs, columns=['segment_id', 'body_id'])

        ##
        ## Label Indices
        ##
        if options["operation"] in ("labelindices", "both"):
            self._execute_labelindices(mapping_df)

        ##
        ## Mappings
        ##
        if options["operation"] in ("mappings", "both"):
            self._execute_mappings(mapping_df)

        logger.info("DONE.")


    def _execute_labelindices(self, mapping_df):
        config = self.config_data
        options = config["options"]
        resource_manager_client = ResourceManagerClient( options["resource-server"], options["resource-port"] )

        server = config["dvid"]["server"]
        uuid = config["dvid"]["uuid"]
        instance_name = config["dvid"]["segmentation-name"]

        ##
        ## Distribute block statistics file (in chunks of CSV rows)
        ##
        csv_chunk_size = config["options"]["csv-chunk-size"]
        dtypes = { 'segment_id': np.uint64, 'z': np.int32, 'y': np.int32, 'x':np.int32, 'count': np.uint32 }
        block_stats_df_chunks = pd.read_csv(config['block-stats-file'], engine='c', dtype=dtypes, chunksize=csv_chunk_size)
        stats_chunks = self.sc.parallelize( block_stats_df_chunks )
        rt.persist_and_execute(stats_chunks, "Loading and distributing block statistics", logger)

        ##
        ## Map segment id -> body id, append body column
        ##

        def append_body_column_to_partitions(stats_chunks_partitions):
            updated_chunks = []
            for stats_chunk in stats_chunks_partitions:
                if mapping_df is None:
                    stats_chunk['body_id'] = stats_chunk['segment_id']
                else:
                    mapper = LabelMapper(mapping_df['segment_id'].values, mapping_df['body_id'].values)
                    stats_chunk['body_id'] = mapper.apply(stats_chunk['segment_id'].values, allow_unmapped=True)
                updated_chunks.append(stats_chunk)
            return updated_chunks

        stats_chunks = stats_chunks.mapPartitions(append_body_column_to_partitions)
        rt.persist_and_execute(stats_chunks, "Applying labelmap to stats", logger)

        ##
        ## Split by body and concatenate
        ##

        def split_by_body(stats_chunk_df):
            split_chunks = []
            for body_id, body_group_df in stats_chunk_df.groupby('body_id'):
                split_chunks.append( (body_id, body_group_df) )
            return split_chunks

        stats_chunks = stats_chunks.flatMap(split_by_body)
        rt.persist_and_execute(stats_chunks, "Splitting chunks by body", logger)

        def merge_chunks(chunk_df_A, chunk_df_B):
            return pd.concat((chunk_df_A, chunk_df_B), ignore_index=True)
        stats_chunks_by_body = stats_chunks.reduceByKey( merge_chunks )
        rt.persist_and_execute(stats_chunks_by_body, "Grouping split chunks by body", logger)
        
        ##
        ## Encode block-id
        ##
        
        instance_info = DataInstance(server, uuid, instance_name)
        bz, by, bx = instance_info.blockshape_zyx

        def encode_block_ids(body_and_stats):
            body_id, stats_chunk_df = body_and_stats
            
            # Convert coords into block indexes
            stats_chunk_df['z'] //= bz
            stats_chunk_df['y'] //= by
            stats_chunk_df['x'] //= bx

            # Encode block indexes into a shared uint64    
            # Sadly, there is no clever way to speed this up with pd.eval()
            encoded_block_ids = np.zeros( len(stats_chunk_df), dtype=np.uint64 )
            encoded_block_ids |= (stats_chunk_df['z'].values.astype(np.int64) << 42).view(np.uint64)
            encoded_block_ids |= (stats_chunk_df['y'].values.astype(np.int64) << 21).view(np.uint64)
            encoded_block_ids |= (stats_chunk_df['x'].values.astype(np.int64)).view(np.uint64)
        
            del stats_chunk_df['z']
            del stats_chunk_df['y']
            del stats_chunk_df['x']
            stats_chunk_df['encoded_block_id'] = encoded_block_ids

            return (body_id, stats_chunk_df)

        stats_chunks_by_body = stats_chunks_by_body.map(encode_block_ids)
        rt.persist_and_execute(stats_chunks_by_body, "Encoding block ids", logger)
        
        ##
        ## Construct protobuf structures
        ##
        
        user = os.environ["USER"]
        mod_time = datetime.datetime.now().isoformat()
        last_mutid = config["options"]["mutation-id"]

        def label_index_for_body(item):
            body_id, body_group_df = item
            label_index = LabelIndex()
            label_index.label = body_id
            label_index.last_mutid = last_mutid
            label_index.last_mod_user = user
            label_index.last_mod_time = mod_time
            
            for encoded_block_id, block_df in body_group_df.groupby(['encoded_block_id']):
                label_index.blocks[encoded_block_id].counts.update( zip(block_df['segment_id'], block_df['count']) )
            return label_index

        label_indexes = stats_chunks_by_body.map(label_index_for_body)
        rt.persist_and_execute(label_indexes, "Creating LabelIndexes", logger)
        
        ##
        ## Serialize protobuf and send to DVID
        ##
        
        session = default_dvid_session()
        def send_labelindexes(partition):
            partition = list(partition)
            batch_size = 100
            
            # Send in batches
            for batch_start in range(0, len(partition), batch_size):
                batch_stop = min(batch_start + batch_size, len(partition))
            
                label_indices = LabelIndices()
                label_indices.indices.extend(partition[batch_start:batch_stop])
                payload = label_indices.SerializeToString()
            
                with resource_manager_client.access_context( instance_name, False, 1, len(payload) ):
                    r = session.post(f'{server}/api/node/{uuid}/{instance_name}/indices', data=payload)
                    r.raise_for_status()

        with Timer("Sending LabelIndices to DVID", logger):
            label_indexes.foreachPartition(send_labelindexes)


    def _execute_mappings(self, mapping_df):
        config = self.config_data
        if mapping_df is None:
            raise RuntimeError("Can't load mappings: No agglomeration mapping provided.")

        # Just do this from a single machine (the driver), with a big batch size
        # The writes are serialized on the DVID side, anyway.
        with Timer("Sending mapping", logger):
            ingest_mapping( config["dvid"]["server"],
                            config["dvid"]["uuid"],
                            config["dvid"]["segmentation-name"],
                            config["options"]["mutation-id"],
                            mapping_df,
                            batch_size=100_000,
                            show_progress_bar=False,
                            session=default_dvid_session() )
