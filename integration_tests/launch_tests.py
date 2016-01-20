### Notes #####

# 1. spark-submit should be in the runtime PATH
# 2. must specify directory to the integration tests
# 3. integration tests must be in the same directory as the workflow scripts

import subprocess
import os
import json
import time
import string

def run_test(test_name, plugin, test_dir, uuid1, uuid2):
    start = time.time()
    print "Starting test: ", test_name

    num_jobs = 8
    config_json = test_dir+"/"+test_name+"/temp_data/config.json"
    job_command = 'spark-submit --master local[{num_jobs}] {test_dir}/../workflows/launchworkflow.py {plugin} -c {config_json}'\
                   .format(**locals())

    print job_command
    with open(test_dir+"/"+test_name+"/config.json") as fin:
        data = fin.read()

    data = data.replace("UUID1", uuid1)
    data = data.replace("UUID2", uuid2)
    data = data.replace("DIR", test_dir)

    with open(test_dir+"/"+test_name+"/temp_data/config.json", 'w') as fout:
        fout.write(data)

    try:
        results = subprocess.check_output(job_command, shell=True)
        
        # write results out
        with open(test_dir+"/"+test_name+"/temp_data/results.txt", 'w') as fout:
            fout.write(results)
    except subprocess.CalledProcessError as ex:
        print "BAD RETURN CODE:", ex.returncode
    else:
        # compare results to results in output
        result_lines = results.splitlines()
        correct = True
    
        with open(test_dir+"/"+test_name+"/outputs/results.txt") as fin:
            correctoutput = fin.read()
            correct_lines = correctoutput.splitlines()
            debug1 = []
            debug2 = []
    
            # Compare only DEBUG lines, ignore everything else.
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
            checkoutput_cmd = ("python " + test_dir + "/" + test_name + "/checkoutput.py " + test_dir+"/"+test_name).split()

            try: 
                subprocess.check_call(checkoutput_cmd)
            except subprocess.CalledProcessError as ex:
                print "FAIL: checkoutput.py returned bad status: {}".format(ex.returncode)
                correct = False
    
        if not correct:
            print "FAILED"
        else:
            print "SUCCESS"
    
    finish = time.time()

    print "Finished test: ", test_name, " in ", finish-start, " seconds"

def init_dvid_database(test_dir):
    print "Initializing DVID Database"
   
    os.system("gunzip -f --keep " + test_dir + "/resources/labels.bin.gz")
    os.system("gunzip -f --keep " + test_dir + "/resources/labels_comp.bin.gz")
    os.system("tar zxvf " + test_dir + "/resources/agglom.xml.tgz")
    os.system("tar zxvf " + test_dir + "/resources/voxels.ilp.tgz")

    # initialize DVID datastore and call tests 
    # Curl must be available
    
    create_repo_command = "curl -X POST 127.0.0.1:8000/api/repos".split()
    
    # create first UUID
    repoinfo = subprocess.check_output(create_repo_command)
    data = json.loads(repoinfo)
    uuid1 = data["root"]
    
    # create second UUID
    repoinfo = subprocess.check_output(create_repo_command)
    data = json.loads(repoinfo)
    uuid2 = data["root"]
    
    
    # create labelblk instance for two uuids
    create_instance = 'curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d'
    typedata = "{\"typename\": \"labelblk\", \"dataname\" : \"labels\"}"
    
    create_instance1_command = (create_instance % uuid1).split()
    create_instance2_command = (create_instance % uuid2).split()
    
    create_instance1_command.append(typedata)
    create_instance2_command.append(typedata)

    subprocess.check_call(create_instance1_command)
    subprocess.check_call(create_instance2_command)
    
    # load binary label data into uuid1
    load_data1_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/labels/raw/0_1_2/512_512_512/0_0_0 --data-binary @%s/resources/labels.bin' % (uuid1, test_dir)).split()
    subprocess.check_call(load_data1_command)
    
    # load binary label data into uuid2
    load_data2_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/labels/raw/0_1_2/512_512_512/0_0_0 --data-binary @%s/resources/labels_comp.bin' % (uuid2, test_dir)).split()
    subprocess.check_call(load_data2_command)
    
    # create ROI datatype
    create_roi_command = ('curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d' % uuid1).split()
    create_roi_command.append("{\"typename\": \"roi\", \"dataname\" : \"temproi\"}")
    subprocess.check_call(create_roi_command)
    
    # load ROI
    load_roi_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/temproi/roi --data-binary @%s/resources/500roi.json' % (uuid1, test_dir)).split()
    subprocess.check_call(load_roi_command)
    
    # create synapse key value
    create_synapse_command = ('curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d' % uuid1).split()
    create_synapse_command.append("{\"typename\": \"keyvalue\", \"dataname\" : \"annotations\"}")
    subprocess.check_call(create_synapse_command)
    
    # load synapses
    load_synapse_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/annotations/key/syn --data-binary @%s/resources/synapse_small.json' % (uuid1, test_dir)).split()
    subprocess.check_call(load_synapse_command)
    
    # create grayscale data
    create_gray = 'curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d'
    typedata = "{\"typename\": \"uint8blk\", \"dataname\" : \"grayscale\"}"
    create_gray1_command = (create_gray % uuid1).split()
    create_gray1_command.append(typedata)
    subprocess.check_call(create_gray1_command)
    
    # load grayscale data
    load_gray1_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/grayscale/raw/0_1_2/256_256_256/0_0_0 --data-binary @%s/resources/grayscale-256-256-256-uint8.bin' % (uuid1, test_dir)).split()
    subprocess.check_call(load_gray1_command)
    
    # create 256 ROI datatype
    create_roi_command = ('curl -X POST 127.0.0.1:8000/api/repo/%s/instance -d' % uuid1).split()
    create_roi_command.append("{\"typename\": \"roi\", \"dataname\" : \"temproi256\"}")
    subprocess.check_call(create_roi_command)
    
    # load 256 ROI
    load_roi_command = ('curl -X POST 127.0.0.1:8000/api/node/%s/temproi256/roi --data-binary @%s/resources/256roi.json' % (uuid1, test_dir)).split()
    subprocess.check_call(load_roi_command)
    
    return uuid1, uuid2

def run_tests(test_dir, uuid1, uuid2):
    #####  run tests ####

    # test 1 segmentation with DefaultGrayOnly Segmentor
    run_test("test_seg", "CreateSegmentation", test_dir, uuid1, uuid2)

    # test 1.5 segmentation with Segmentor (base class)
    run_test("test_Segmentor", "CreateSegmentation", test_dir, uuid1, uuid2)
    
    # test 2 segmentation iteration
    run_test("test_seg_iteration", "CreateSegmentation", test_dir, uuid1, uuid2) 

    # test 3 segmentation rollback
    run_test("test_seg_rollback", "CreateSegmentation", test_dir, uuid1, uuid2) 

    # test 4 label comparison
    run_test("test_comp", "EvaluateSeg", test_dir, uuid1, uuid2) 

    # test 5 graph compute
    run_test("test_graph", "ComputeGraph", test_dir, uuid1, uuid2) 

    # test 6 grayscale ingestion
    run_test("test_ingest", "IngestGrayscale", test_dir, uuid1, uuid2) 

    # test 7: segmentation with ilastik
    # First, verify that ilastik is available
    try:
        import ilastik_main
    except ImportError:
        sys.stderr.write("Skipping ilastik segmentation test")
    else:
        # Voxel prediction with ilastik
        run_test("test_seg_ilastik", "CreateSegmentation", test_dir, uuid1, uuid2)

        print "RUNNING TWO-STAGE TEST"

        # Two-stage voxel prediction with ilastik
        run_test("test_seg_ilastik_two_stage", "CreateSegmentation", test_dir, uuid1, uuid2)

    # test 8: Generate supervoxels with the wsdt module
    try:
        import wsdt
    except ImportError:
        sys.stderr.write("Skipping wsdt supervoxel test")
    else:
        run_test("test_seg_wsdt", "CreateSegmentation", test_dir, uuid1, uuid2)

    # test 9: segmentation with neuroproof
    # test 10: segmentation with neuroproof where pre-existing bodies are preserved
    # First, verify that ilastik and neuroproof is available
    try:
        import neuroproof
        import ilastik_main
    except ImportError:
        sys.stderr.write("Skipping neuroproof segmentation test")
    else:
        run_test("test_seg_neuroproof", "CreateSegmentation", test_dir, uuid1, uuid2)
        run_test("test_seg_replace", "CreateSegmentation", test_dir, uuid1, uuid2)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 1:
        sys.stderr.write("This script takes no arguments.\n")
        sys.exit(1)

    # It is assumed that this script lives in the integration_tests directory
    test_dir = os.path.split(__file__)[0]
    uuid1, uuid2 = init_dvid_database(test_dir)
    run_tests(test_dir, uuid1, uuid2)
