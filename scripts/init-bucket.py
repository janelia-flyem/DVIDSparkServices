"""
Initialize a gbucket with a new dvid repo.

Example Usage:

    python init-bucket.py my-gbucket my-new-repo "This repo is for my stuff" 

    python init-bucket.py --create-bucket my-NEW-gbucket my-new-repo "This repo is for my stuff" 

"""
import os
import sys
import time
import textwrap
import argparse
import subprocess
import DVIDSparkServices

DVID_CONSOLE_DIR = '/magnetic/workspace/dvid-distro/dvid-console'
#BUCKET_NAME = 'alignment-eval-2017-06'
#BUCKET_NAME = 'flyem-alignment-quick-eval'

LOG_DIR = os.getcwd() + '/logs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    GOOGLE_APPLICATION_CREDENTIALS='/magnetic/workspace/DVIDSparkServices/cloud-keys/dvid-em-28a78d822e11.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

def get_toml_text(bucket_name, dvid_console_dir, log_dir):
    return textwrap.dedent("""\
        [server]
        httpAddress = ":8000"
        rpcAddress = ":8001"
        webClient = "{dvid_console_dir}"
        instance_id_gen = "sequential"
        instance_id_start = 100  # new ids start at least from this.
        
        note = \"""
        {{"source": "gs://{bucket_name}"}}
        \"""
        
        [logging]
        logfile = "{log_dir}/{bucket_name}.log"
        max_log_size = 500 # MB
        max_log_age = 30   # days
        [store]
            [store.mutable]
                engine = "gbucket"
                bucket= "{bucket_name}"
        """.format(**locals()))

def main():
    parser = argparse.ArgumentParser(description='Initialize a gbucket with a new dvid repo.')
    parser.add_argument('--create-bucket', action='store_true',
                        help='If provided, the bucket will be created first using gsutil. '
                             'Otherwise, the bucket is assumed to exist.')
    parser.add_argument('bucket_name')
    parser.add_argument('repo_name')
    parser.add_argument('repo_description')
    args = parser.parse_args()

    # Strip leading 'gs://', if provided
    if args.bucket_name.startswith('gs://'):
        args.bucket_name = args.bucket_name[len('gs://'):]

    if args.create_bucket:
        subprocess.check_call('gsutil mb -c regional -l us-east4 -p dvid-em gs://{}'.format(args.bucket_name), shell=True)

    toml_path = '{}.toml'.format(args.bucket_name)
    with open(toml_path, 'w') as f_toml:
        f_toml.write(get_toml_text(args.bucket_name, DVID_CONSOLE_DIR, LOG_DIR))
    
    print "Wrote {}".format(toml_path)
    
    try:
        cmd = 'dvid -verbose serve {toml_path}'.format(toml_path=toml_path)
        print cmd
        dvid_proc = subprocess.Popen(cmd, shell=True)
        
        print "Waiting 5 seconds for dvid to start...."
        time.sleep(5.0)
    
        cmd = 'dvid repos new "{}" "{}"'.format(args.repo_name, args.repo_description)
        print cmd
        response = subprocess.check_output(cmd, shell=True).strip()
        print response
        #repo_uuid = response.split()[-1]
    finally:
        dvid_proc.terminate()


if __name__ == "__main__":
    sys.exit( main() )
