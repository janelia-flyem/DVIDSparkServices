#!/usr/bin/env python3
"""
Script to copy a list of mesh tarballs from one DVID node to another.
See --help for details.

Pre-requisites:

    conda install requests tqdm

Example usage:
    
    python copy-meshes.py --help
    python copy-meshes.py --parallelism=4 body-ids.csv emdata2:8700 0667 segmentation emdata2:8700 abc123 segmentation

"""
import sys
import csv
import logging
import argparse
import collections
import multiprocessing

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

InstanceInfo = collections.namedtuple('InstanceInfo', 'server uuid instance')

def main():
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tarball-type', '--tt', choices=['sv', 'body', 'both'], default='both',
                        help='Whether to copy SV meshes, body meshes, or both')
    parser.add_argument('--error-mode', '-e', choices=['fail', 'log-and-continue'], default='log-and-continue',
                        help='What to do if a mesh cannot be downloaded/uploaded. Either fail the entire job, or just log an error to the console and move on.')
    parser.add_argument('--parallelism', '-p', type=int, default=1,
                        help='How many processes to use to copy meshes in parallel')
    parser.add_argument('--skip-count', type=int, default=0,
                        help='Optionally skip the first N meshes in the list (e.g. to resume a stopped/failed job')

    parser.add_argument('body_list_csv', help='CSV file whose first (and possibly only) column contains a list of body IDs whose meshes should be copied')

    parser.add_argument('from_server', help='Example: "emdata3:8700"')
    parser.add_argument('from_uuid')
    parser.add_argument('from_instance')
    
    parser.add_argument('to_server', help='Example: "emdata3:8700"')
    parser.add_argument('to_uuid')
    parser.add_argument('to_instance')

    args = parser.parse_args()

    body_ids = read_body_ids(args.body_list_csv)
    body_ids = body_ids[args.skip_count:]

    if args.from_server.startswith('http://'):
        args.from_server = args.from_server[len('http://'):]
    info_from = InstanceInfo(args.from_server, args.from_uuid, args.from_instance)

    if args.to_server.startswith('http://'):
        args.to_server = args.to_server[len('http://'):]
    info_to = InstanceInfo(args.to_server, args.to_uuid, args.to_instance)

    logger.info(f"Copying meshes for {len(body_ids)} bodies.")
    copy_meshes(info_from, info_to, args.tarball_type, body_ids, args.parallelism, args.error_mode)
    logger.info("DONE.")


def copy_meshes(info_from, info_to, tarball_type, body_ids, parallelism=1, error_mode='fail'):
    pool = multiprocessing.Pool(parallelism)
    tasks = map(lambda body_id: pool.apply_async(copy_tarballs_for_body, (info_from, info_to, tarball_type, body_id)), body_ids)
    ids_and_tasks = zip(body_ids, tasks)
    
    # By evaluating the above map(), this immediately starts distributing tasks to the pool
    ids_and_tasks = list(ids_and_tasks)
    
    # Iterate over the 'results' in the queue.
    # If any failed in a worker process, the exception will be re-raised here upon calling get(), below.
    for i, (body_id, task) in enumerate(tqdm(ids_and_tasks)):
        try:
            task.get() # Ensure copy is complete; catch any pickled exception now
        except requests.RequestException:
            with tqdm.external_write_mode():
                logger.error(f"Error copying body {body_id} (mesh #{i} in the list)")
            if error_mode == 'fail':
                # Note: Since we're using a pool, it's possible that some meshes
                #       after this one have already successfully copied,
                #       but at least all meshes before this one have definitely succeeded.
                raise


def copy_tarballs_for_body(info_from, info_to, tarball_type, body_id):
    # Special constants used by neu3 to differentiate sv tarballs vs body tarballs
    keyEncodeLevel0 = 10000000000000
    keyEncodeLevel1 = 10100000000000

    if tarball_type in ('sv', 'both'):
        copy_key(f'{body_id + keyEncodeLevel0}.tar', info_from, info_to)

    if tarball_type in ('body', 'both'):
        copy_key(f'{body_id + keyEncodeLevel1}.tar', info_from, info_to)


def copy_key(key, info_from, info_to):
    r = requests.get(f"http://{info_from.server}/api/node/{info_from.uuid}/{info_from.instance}/key/{key}")
    r.raise_for_status()

    r2 = requests.post(f"http://{info_to.server}/api/node/{info_to.uuid}/{info_to.instance}/key/{key}", data=r.content)
    r2.raise_for_status()


def read_body_ids(csv_path):
    """
    Read a list of body IDs from the first column of the given csv file.
    The file may or may not contain a header row.
    """
    with open(csv_path, 'r') as csv_file:
        first_line = csv_file.readline()
        csv_file.seek(0)
        if ',' not in first_line:
            # csv.Sniffer doesn't work if there's only one column in the file
            try:
                int(first_line)
                has_header = False
            except:
                has_header = True
        else:
            has_header = csv.Sniffer().has_header(csv_file.read(1024))
            csv_file.seek(0)

        rows = iter(csv.reader(csv_file))
        if has_header:
            _header = next(rows) # Skip header
        
        # File is permitted to have multiple columns,
        # but body ID must be first column
        body_ids = [int(row[0]) for row in rows]

    return body_ids


if __name__ == "__main__":
    main()
