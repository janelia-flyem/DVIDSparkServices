import re
import time
import subprocess
from datetime import datetime

class Bjob(object):
    """
    Convenience class for submitting a bjob.
    """
    
    def __init__( self,
                  shell_command,
                  name=None,
                  num_slots=1,
                  max_runtime_minutes=None,
                  estimated_runtime_minutes=None,
                  stdout_file=None,
                  stderr_file=None,
                  app_env=None ):

        # Copy all parameters as members
        for k,v in locals().items():
            if k != 'self':
                setattr(self, k, v)

        # Valid after submit()
        self.bsub_cmd = None
        self.job_id = None
        self.queue_name = None
        self.rtm_url = None

        # Valid after wait_for_start()
        self.hostname = None

    def submit(self, suspend=False, wait_for_start=True, write_log_header=True):
        """
        Submit the job.
        
        suspend:
            If true, submit the job but don't schedule it yet.
            Useful if you want to write certain job properties to
            the job's own logfile before it actually starts running.

        wait_for_start:
            Don't return from this function until the job has actually been scheduled.
            Obviously, it's not valid to use both suspend and wait_for_start simultaneously.
        
        write_log_header:
            Write some useful info (e.g. the bsub command, and a link to its RTM host graphs)
            to the job logfile before the job is scheduled.
        
        Returns:
            (job_id, queue_name, hostname)
            If wait_for_start is False, then hostname will be None.
            
        """
        assert not self.bsub_cmd, "Job is already submitted."
        assert not (suspend and wait_for_start), \
            "Can't wait for a job that will never start!"
        
        self.bsub_cmd = self.construct_bsub_command(suspend or write_log_header)
        
        print(self.bsub_cmd + "\n")
        bsub_output = subprocess.check_output(self.bsub_cmd, shell=True).decode()
        print(bsub_output)
        self.job_id, self.queue_name = parse_bsub_output(bsub_output)
        
        self.rtm_url = get_hostgraph_url(self.job_id)
        print ("Host graphs:")
        print(self.rtm_url + "\n")
    
        if write_log_header:
            self.write_logfile_header()
            if not suspend:
                self.resume()
    
        if wait_for_start:
            self.wait_for_start()

        return self.job_id, self.queue_name, self.hostname

    def construct_bsub_command(self, suspend):
        cmd = "bsub"
        
        if suspend:
            cmd += " -H"
        
        cmd += f" -n {self.num_slots}"

        if self.name:
            cmd += f" -J {self.name}"
        if self.app_env:
            cmd += f" -a '{self.app_env}'"
        if self.max_runtime_minutes:
            cmd += f" -W {self.max_runtime_minutes}"
        if self.estimated_runtime_minutes:
            cmd += f" -We {self.estimated_runtime_minutes}"
        if self.stdout_file:
            cmd += f" -o {self.stdout_file}"
        if self.stderr_file:
            cmd += f" -e {self.stderr_file}"

        cmd += f" '{self.shell_command}'"

        return cmd

    def resume(self):
        subprocess.check_output(f"bresume {self.job_id}", shell=True).decode()
    
    def wait_for_start(self):
        name = self.name or str(self.job_id)
        print(f"Waiting for {name} to start...")
        self.hostname = wait_for_job_start(self.job_id)

    def write_logfile_header(self):
        assert self.rtm_url, "Job not submitted yet."
        assert self.stdout_file, "No output stdout logfile specified."
        with open(self.stdout_file, 'a') as f:
            f.write("Job submitted with the following command:\n")
            f.write(self.bsub_cmd + "\n\n")
            f.write("Host graphs for this job can be found at the following URL:\n")
            f.write(self.rtm_url + "\n\n")

def get_hostgraph_url(job_id):
    """
    Construct a URL that can be used to browse a job's host
    graphs on Janelia's RTM web server.
    """
    submit_time = get_job_submit_time(job_id)
    submit_timestamp = int(submit_time.timestamp())
    rtm_url = ( "http://lsf-rtm/cacti/plugins/grid/grid_bjobs.php"
                "?action=viewjob"
                "&tab=hostgraph"
                "&clusterid=1"
                "&indexid=0"
               f"&jobid={job_id}"
               f"&submit_time={submit_timestamp}" )
    
    return rtm_url

def parse_bsub_output(bsub_output):
    """
    Parse the given output from the 'bsub' command and return the job ID and the queue name.

    Example:
        
        >>> bsub_output = "Job <774133> is submitted to queue <spark>.\n"
        >>> job_id, queue_name = parse_bsub_output(bsub_output)
        >>> assert job_id == '774133'
        >>> assert queue_name == 'spark'
    """
    nonbracket_text = '[^<>]*'
    field_pattern = "{nonbracket_text}<({nonbracket_text})>{nonbracket_text}".format(**locals())

    NUM_FIELDS = 2
    field_matches = re.match(NUM_FIELDS*field_pattern, bsub_output)

    if not field_matches:
        raise RuntimeError("Could not parse bsub output: {}".format(bsub_output))

    job_id = field_matches.groups()[0]
    queue_name = field_matches.groups()[1]
    return job_id, queue_name

def get_job_hostname(job_id):
    """
    For the given job, return the name of the host it's running on.
    If it is running on more than one host, the first hostname listed by bjobs is returned.
    (For 'sparkbatch' jobs, the first host is the master.)
    """
    bjobs_output = subprocess.check_output(f'bjobs -X -noheader -o EXEC_HOST {job_id}', shell=True).decode()
    hostname = bjobs_output.split(':')[0].split('*')[-1].strip()
    return hostname

def get_job_submit_time(job_id):
    """
    Return the job's submit_time as a datetime object.
    """
    bjobs_output = subprocess.check_output(f'bjobs -X -noheader -o SUBMIT_TIME {job_id}', shell=True).strip().decode()
    # Example:
    # Sep  6 13:10:09 2017
    submit_time = datetime.strptime(f"{bjobs_output} {time.localtime().tm_zone}", "%b %d %H:%M:%S %Y %Z")
    return submit_time

def wait_for_job_start(job_id):
    """
    Wait for the job to start and return its hostname when it does.
    If it is running on more than one host, the first hostname listed by bjobs is returned.
    """
    wait_times = [1.0, 5.0, 10.0]
    hostname = get_job_hostname(job_id)
    while hostname == '-':
        time.sleep(wait_times[0])
        if len(wait_times) > 1:
            wait_times = wait_times[1:]
        hostname = get_job_hostname(job_id)
    return hostname