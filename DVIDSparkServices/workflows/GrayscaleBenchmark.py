import copy
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dvid_resource_manager.client import ResourceManagerClient

import DVIDSparkServices.rddtools as rt
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.util import default_dvid_session, num_worker_nodes, NumpyConvertingEncoder
from DVIDSparkServices.io_util.volume_service import VolumeService
from DVIDSparkServices.io_util.volume_service.dvid_volume_service import DvidGenericVolumeSchema
from DVIDSparkServices.io_util.brick import Grid, clipped_boxes_from_grid, boxes_from_grid

logger = logging.getLogger(__name__)


class GrayscaleBenchmark(Workflow):
    """
    This workflow will fetch grayscale data from a DVID server via the /specificblocks endpoint,
    and collect timing information.
    """
    GrayscaleBenchmarkOptionsSchema = copy.copy(Workflow.OptionsSchema)
    GrayscaleBenchmarkOptionsSchema["additionalProperties"] = False
    GrayscaleBenchmarkOptionsSchema["properties"].update(
    {
        "warmup-minutes": {
            "description": "Two sets of statistics are produced: One set for the entire job, \n"
                           "and one set that excludes requests made during the 'warmup' period, \n"
                           "as defined by this setting. \n",
            "type": "number",
            "default": 10
        }
    })

    Schema = \
    {
        "$schema": "http://json-schema.org/schema#",
        "title": "Service to test performance of fetching DVID data from multiple workers in parallel",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input": DvidGenericVolumeSchema,
            "options": GrayscaleBenchmarkOptionsSchema
        }
    }

    @classmethod
    def schema(cls):
        return GrayscaleBenchmark.Schema

    # name of application for DVID queries
    APPNAME = "grayscalebenchmark"


    def __init__(self, config_filename):
        super(GrayscaleBenchmark, self).__init__( config_filename, GrayscaleBenchmark.schema(), "Grayscale Benchmark" )
    
    def _sanitize_config(self):
        """
        - Normalize/overwrite certain config values
        - Check for config mistakes
        - Simple sanity checks
        """
        #options = self.config_data["options"]

    def execute(self):
        from pyspark import StorageLevel

        self._sanitize_config()
        config = self.config_data
        options = config["options"]

        resource_mgr_client = ResourceManagerClient(options["resource-server"], options["resource-port"])
        total_cpus = 16 * num_worker_nodes()
        
        concurrent_threads = total_cpus
        if options["resource-server"]:
            concurrent_threads = options["resource-server-config"]["read_reqs"]
            if concurrent_threads > total_cpus:
                msg = "You're attempting to use the resource manager to constrain concurrency, but you "\
                      "aren't running with a large enough cluster to saturate the resource manager settings"
                raise RuntimeError(msg)

        # We instantiate a VolumeService as an easy way to plug in missing config values as necessary.
        # (We won't actually use it.)
        volume_service = VolumeService.create_from_config(config["input"], self.config_dir)

        server = volume_service.server
        uuid = volume_service.uuid
        instance = volume_service.instance_name
        block_shape = 3*(volume_service.block_width,)

        def timed_fetch_blocks_from_box(box):
            """
            Fetch the blocks for a given box and return the time it took to fetch them.
            Do not bother decompressing the blocks or combining them into a single volume.
            """
            assert not (box % block_shape).any(), "For this test, all requests must be block-aligned"
            block_boxes = list( boxes_from_grid(box, Grid(block_shape)) )
            block_coords_xyz = np.array(block_boxes)[:,0,::-1] // block_shape
            block_coords_str = ','.join(map(str, block_coords_xyz.flat))

            voxel_count = np.prod(box[1] - box[0])

            session = default_dvid_session()
            url = f'{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords_str}'
            
            with resource_mgr_client.access_context(server, True, 1, voxel_count):
                timestamp = datetime.now()
                r = session.get(url)
            
            r.raise_for_status()
            return timestamp, voxel_count, len(r.content), r.elapsed.total_seconds()

        # This hash-related hackery is to ensure uniform partition lengths, which Spark is bad at by default.
        boxes = list(clipped_boxes_from_grid( volume_service.bounding_box_zyx, Grid(volume_service.preferred_message_shape) ))
        indexed_boxes = list(map(rt.tuple_with_hash, (enumerate(boxes))))
        for i_box in indexed_boxes:
            i_box.set_hash(i_box[0])

        rdd_boxes = self.sc.parallelize(indexed_boxes).values()
        timestamps_voxels_sizes_times = rdd_boxes.map(timed_fetch_blocks_from_box)
        
        # The only reason I'm persisting this is to see the partition distribution in the log.
        rt.persist_and_execute(timestamps_voxels_sizes_times, "Fetching blocks", logger, StorageLevel.MEMORY_ONLY) #@UndefinedVariable

        # Execute the workload
        timestamps, voxels, sizes, times = zip( *timestamps_voxels_sizes_times.collect() )
        
        # Process the results
        self.dump_stats(timestamps, voxels, sizes, times, block_shape, concurrent_threads)

    def dump_stats(self, timestamps, voxels, sizes, times, block_shape, concurrent_threads):
        config = self.config_data
        
        voxels = np.asarray(voxels, np.uint32)
        sizes = np.asarray(sizes, np.uint32)
        times = np.asarray(times, np.float32)

        df = pd.DataFrame({'timestamp': timestamps, 'voxel_count': voxels, 'payload_bytes': sizes, 'seconds': times})
        df = df[['timestamp', 'voxel_count', 'payload_bytes', 'seconds']]
        df.sort_values(['timestamp'], inplace=True)
        warmup_end = df['timestamp'].iloc[0] + timedelta(minutes=config["options"]["warmup-minutes"])
        df['warmup'] = df['timestamp'] < warmup_end

        df.to_csv(self.relpath_to_abspath('requests-table-complete.csv'), index=False, header=True)
        
        stats = compute_stats(block_shape, concurrent_threads, df)
        with open(self.relpath_to_abspath('stats-complete.json'), 'w') as f:
            json.dump(stats, f, indent=2, cls=NumpyConvertingEncoder)

        # Now compute the stats after warmup time
        df_after_warmup = df.query('not warmup')
        
        if len(df_after_warmup) == 0:
            logger.error("The job completed before the warmup period ended.  No after-warmup stats will be written.")
        else:
            logger.info(f"Dropping {df['warmup'].sum()} warmup requests from final stats")
            stats_after_warmup = compute_stats(block_shape, concurrent_threads, df_after_warmup)
            logger.info("Stats after warmup:\n" + json.dumps(stats_after_warmup, indent=2, cls=NumpyConvertingEncoder))
            with open(self.relpath_to_abspath('stats-after-warmup.json'), 'w') as f:
                json.dump(stats_after_warmup, f, indent=2, cls=NumpyConvertingEncoder)

        logger.info("DONE.")


def compute_stats(block_shape, concurrent_threads, df):
    total_blocks = df['voxel_count'].sum() / np.prod(block_shape)
    stats = {
        "total-analyzed-requests": len(df),

        "num-workers": num_worker_nodes(),
        "available-threads": 16 * num_worker_nodes(),
        "concurrent-threads": concurrent_threads,
        "approx-requests-per-thread": len(df) / (concurrent_threads),

        "blocks-per-request": total_blocks / len(df),
        "seconds-per-request": df['seconds'].mean(),
        "seconds-per-block": df['seconds'].sum() / total_blocks,
        
        "voxels-per-second": df['voxel_count'].sum() / df['seconds'].sum(),
        "mb-per-second": (df['payload_bytes'].sum() / 1e6) / df['seconds'].sum()
    }
    return stats
