### Notes #####

# 1. spark-submit should be in the runtime PATH
# 2. must specify directory to the integration tests
# 3. integration tests must be in the same directory as the workflow scripts

import sys
import subprocess
import os
import json
import time
import string

def run_test(test_name, plugin, test_dir, uuid1, uuid2):
    start = time.time()
    print "Starting test: ", test_name

    job_command = ('spark-submit --master local[8] %s/../workflows/launchworkflow.py %s -c %s' % (test_dir, plugin, test_dir+"/"+test_name+"/temp_data/config.json")).split()

    fin = open(test_dir+"/"+test_name+"/config.json")
    data = fin.read()
    fin.close()
    data = data.replace("UUID1", uuid1)
    data = data.replace("UUID2", uuid2)
    data = data.replace("DIR", test_dir)
    fout = open(test_dir+"/"+test_name+"/temp_data/config.json", 'w')
    fout.write(data)
    fout.close()

    p = subprocess.Popen(job_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    results, err = p.communicate()
    
    # write results out
    fout = open(test_dir+"/"+test_name+"/temp_data/results.txt", 'w')
    fout.write(results)
    fout.close()
    
    # compare results to results in output
    result_lines = results.splitlines()
    correct = True

    with open(test_dir+"/"+test_name+"/outputs/results.txt") as fin:
        correctoutput = fin.read()
        correct_lines = correctoutput.splitlines()
        debug1 = []
        debug2 = []

        for line in result_lines:
            if string.find(line, "DEBUG:") != -1:
                debug1.append(line)
        for line in correct_lines:
            if string.find(line, "DEBUG:") != -1:
                debug2.append(line)
       
        if len(debug1) != len(debug2):
            correct = False
        else:
            for iter1 in range(0, len(debug1)):
                if debug1[iter1] != debug2[iter1]:
                    correct = False
                    break
    
    # verify output using custom python script if it exists
    if os.path.exists(test_dir+"/"+test_name+"/checkoutput.py"):
        checkoutput = ("python " + test_dir + "/" + test_name + "/checkoutput.py " + test_dir+"/"+test_name).split()
        
        p = subprocess.Popen(checkoutput, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        dummy, err = p.communicate()

        if p.returncode != 0:
            correct = False

    if not correct:
        print "FAILED"
    else:
        print "SUCCESS"

    finish = time.time()

    print "Finished test: ", test_name, " in ", finish-start, " seconds"


print "Initializing DVID Database"

# supply path to test directory
testpath = sys.argv[1]

# initialize DVID datastore and call tests 
# Curl must be available

create_repo_command = "curl -X POST 127.0.0.1:8000/api/repos".split()

# create first UUID
p = subprocess.Popen(create_repo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
repoinfo, err = p.communicate()
data = json.loads(repoinfo)
uuid1 = data["root"]


# create second UUID
p = subprocess.Popen(create_repo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
repoinfo, err = p.communicate()
data = json.loads(repoinfo)
uuid2 = data["root"]


# create labelblk instance for two uuids
create_instance = 'curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d'
typedata = "{\"typename\": \"labelblk\", \"dataname\" : \"labels\"}"

create_instance1_command = (create_instance % uuid1).split()
create_instance2_command = (create_instance % uuid2).split()

create_instance1_command.append(typedata)
create_instance2_command.append(typedata)

p = subprocess.Popen(create_instance1_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

p = subprocess.Popen(create_instance2_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# load binary label data into uuid1
load_data1_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/labels/raw/0_1_2/512_512_512/0_0_0 --data-binary @%s/resources/labels.bin' % (uuid1, testpath)).split()
p = subprocess.Popen(load_data1_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# load binary label data into uuid2
load_data2_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/labels/raw/0_1_2/512_512_512/0_0_0 --data-binary @%s/resources/labels_comp.bin' % (uuid2, testpath)).split()
p = subprocess.Popen(load_data2_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# create ROI datatype
create_roi_command = ('curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d' % uuid1).split()
create_roi_command.append("{\"typename\": \"roi\", \"dataname\" : \"temproi\"}")
p = subprocess.Popen(create_roi_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# load ROI
load_roi_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/temproi/roi --data-binary @%s/resources/500roi.json' % (uuid1, testpath)).split()
p = subprocess.Popen(load_roi_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# create synapse key value
create_synapse_command = ('curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d' % uuid1).split()
create_synapse_command.append("{\"typename\": \"keyvalue\", \"dataname\" : \"annotations\"}")
p = subprocess.Popen(create_synapse_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# load synapses
load_synapse_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/annotations/key/syn --data-binary @%s/resources/synapse_small.json' % (uuid1, testpath)).split()
p = subprocess.Popen(load_synapse_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# create grayscale data
create_gray = 'curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d'
typedata = "{\"typename\": \"uint8blk\", \"dataname\" : \"grayscale\"}"
create_gray1_command = (create_gray % uuid1).split()
create_gray1_command.append(typedata)

p = subprocess.Popen(create_gray1_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# load grayscale data
load_gray1_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/grayscale/raw/0_1_2/256_256_256/0_0_0 --data-binary @%s/resources/grayscale.bin' % (uuid1, testpath)).split()
p = subprocess.Popen(load_gray1_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# create 256 ROI datatype
create_roi_command = ('curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d' % uuid1).split()
create_roi_command.append("{\"typename\": \"roi\", \"dataname\" : \"temproi256\"}")
p = subprocess.Popen(create_roi_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()

# load 256 ROI
load_roi_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/temproi256/roi --data-binary @%s/resources/256roi.json' % (uuid1, testpath)).split()
p = subprocess.Popen(load_roi_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
dummy, err = p.communicate()


#####  run tests ####

# test 1 segmentation
run_test("test_seg", "CreateSegmentation", testpath, uuid1, uuid2) 

# test 2 label comparison
run_test("test_comp", "EvaluateSeg", testpath, uuid1, uuid2) 

# test 3 graph compute
run_test("test_graph", "ComputeGraph", testpath, uuid1, uuid2) 

# test 4 grayscale ingestion
run_test("test_ingest", "IngestGrayscale", testpath, uuid1, uuid2) 


