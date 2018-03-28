import os
import datetime
import logging

import numpy as np
import pandas as pd

from dvidutils import LabelMapper
from dvid_resource_manager.client import ResourceManagerClient

import DVIDSparkServices.rddtools as rt
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.util import Timer, default_dvid_session, cpus_per_worker, num_worker_nodes

from DVIDSparkServices.io_util.volume_service import DvidSegmentationServiceSchema, LabelMapSchema
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap
from DVIDSparkServices.dvid.ingest_label_indexes import load_stats_h5_to_records, StatsBatchProcessor, generate_stats_batches, ingest_mapping
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
                "batch-row-count": {
                    "description": "Approximately how many rows of block statistics each task should process at a time.\n",
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

        last_mutid = options["mutation-id"]
        server = config["dvid"]["server"]
        uuid = config["dvid"]["uuid"]
        instance_name = config["dvid"]["segmentation-name"]
        endpoint = f'{server}/api/node/{uuid}/{instance_name}/indices'

        processor = StatsBatchProcessor(last_mutid, endpoint)
        
        # Load the h5 file
        block_sv_stats = load_stats_h5_to_records(config["block-stats-file"])
        
        # Note: Initializing this generator involves sorting the (very large) stats array
        batch_rows = options["batch-row-count"]
        batch_generator = generate_stats_batches(block_sv_stats, mapping_df, batch_rows)
        
        with Timer("Distributing batches", logger):
            batches = self.sc.parallelize( batch_generator, cpus_per_worker() * num_worker_nodes() )
        
        def process_batch(item):
            stats_batch, total_rows = item
            approximate_bytes = 30 * total_rows # this is highly unscientific
            with resource_manager_client.access_context(server, False, 1, approximate_bytes):
                processor.process_batch( (stats_batch, total_rows) )
        
        with Timer("Processing/sending batches", logger):
            batches.foreach(process_batch)

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
