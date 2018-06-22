import os
import sys
import logging
import argparse
import subprocess

from tqdm import tqdm

from vol2mesh import Mesh
from neuclease.dvid import fetch_key, post_key

logger = logging.getLogger(__name__)

def main():
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('instance')
    parser.add_argument('body_ids', nargs='+', type=int)
    args = parser.parse_args()

    instance_info = (args.server, args.uuid, args.instance)
    for body_id in tqdm(args.body_ids):
        create_sv_tarball_from_sv_tarball(instance_info, body_id)
        
def create_sv_tarball_from_sv_tarball(instance_info, body_id):
    """
    Download a supervoxel mesh tarball from the given key-value instance,
    concatenate together the component meshes into a single body tarball,
    and upload it.
    """
    keyEncodeLevel0 = 10000000000000
    keyEncodeLevel1 = 10100000000000
    
    encoded_sv = str(body_id + keyEncodeLevel0)
    sv_tarball_path = f'/tmp/{encoded_sv}.tar'
    
    logger.info(f'Fetching {encoded_sv}.tar')
    tarball_contents = fetch_key(instance_info, f'{encoded_sv}.tar')
    with open(sv_tarball_path, 'wb') as f:
        f.write(tarball_contents)
    
    logger.info(f'Unpacking {encoded_sv}.tar')
    sv_dir = f'/tmp/{encoded_sv}'
    os.makedirs(sv_dir, exist_ok=True)
    os.chdir(sv_dir)
    subprocess.check_call(f'tar -xf {sv_tarball_path}', shell=True)

    encoded_body = str(body_id + keyEncodeLevel1)
    body_tarball_path = f'/tmp/{encoded_body}.tar'
    
    logger.info(f"Constructing {encoded_body}.drc")
    mesh = Mesh.from_directory(sv_dir)
    mesh.serialize(f'/tmp/{encoded_body}.drc')
    subprocess.check_call(f'tar -cf {body_tarball_path} /tmp/{encoded_body}.drc', shell=True)
    
    with open(body_tarball_path, 'rb') as f:
        logger.info(f'Posting {encoded_body}.tar')
        post_key(instance_info, f'{encoded_body}.tar', f)

if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        sys.argv += ['emdata3:8900', '275df4022f674852a15bf88514747ead', 'segmentation_meshes_tars']
        sys.argv += ['265879859', '270313572']
    main()
