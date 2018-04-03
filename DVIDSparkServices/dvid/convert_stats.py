import os
import logging
import argparse

from tqdm import tqdm

import numpy as np
import pandas as pd

from DVIDSparkServices.util import Timer

from DVIDSparkServices.dvid.ingest_label_indexes import STATS_DTYPE, load_stats_h5_to_records

logger = logging.getLogger(__name__)

def _convert_coords_to_block_indexes(block_sv_stats):
    segment_dtype = STATS_DTYPE[1]
    coords_dtype = ('coord_cols', (np.int32, 3))
    count_dtype = STATS_DTYPE[5]
    assert segment_dtype[0] == 'segment_id'
    assert count_dtype[0] == 'count'
    
    block_sv_stats = block_sv_stats.view([segment_dtype, coords_dtype, count_dtype])
    block_sv_stats['coord_cols'][:] //= 64

def export_stats(h5_path, output_csv_path, delimiter=' '):
    block_sv_stats = load_stats_h5_to_records(h5_path, False)

    with Timer(f"Sorting {len(block_sv_stats)} block stats", logger):
        block_sv_stats.sort(order=['segment_id', 'z', 'y', 'x', 'count'])

    with Timer(f"Converting coordinates to block indexes", logger):
        _convert_coords_to_block_indexes(block_sv_stats)

    if os.path.exists(output_csv_path):
        os.unlink(output_csv_path)
    
    with Timer(f"Writing sorted stats to {output_csv_path}", logger):
        chunk_size = 10_000_000
        for row_start in tqdm( range(0, len(block_sv_stats), chunk_size) ):
            row_stop = min(row_start + chunk_size, len(block_sv_stats))
            df = pd.DataFrame(block_sv_stats[row_start:row_stop])
            df.to_csv( output_csv_path, sep=' ', header=False, index=False, mode='a' )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delimiter', default=' ')
    parser.add_argument('stats_h5')
    parser.add_argument('output_csv')
    args = parser.parse_args()

    export_stats(args.stats_h5, args.output_csv, args.delimiter)

if __name__ == "__main__":
#     import sys
#     sys.argv += ["/magnetic/workspace/DVIDSparkServices/integration_tests/test_copyseg/temp_data/block-statistics.h5",
#                  "/tmp/sorted-stats.csv"]
    main()
