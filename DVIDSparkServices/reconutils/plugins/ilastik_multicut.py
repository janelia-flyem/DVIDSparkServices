import DVIDSparkServices

def ilastik_multicut(grayscale, bounary_volume, supervoxels, ilp_path, LAZYFLOW_THREADS=1, LAZYFLOW_TOTAL_RAM_MB=None, logfile="/dev/null", extra_cmdline_args=[]):
    print 'status=multicut'
    print "Starting ilastik_multicut() ..."
    print "grayscale volume: dtype={}, shape={}".format(str(grayscale.dtype), grayscale.shape)
    print "boundary volume: dtype={}, shape={}".format(str(bounary_volume.dtype), bounary_volume.shape)
    print "supervoxels volume: dtype={}, shape={}".format(str(supervoxels.dtype), supervoxels.shape)

    import os
    from collections import OrderedDict

    import uuid
    import multiprocessing
    import platform
    import psutil
    import vigra

    import ilastik_main
    from ilastik.applets.dataSelection import DatasetInfo

    print "ilastik_multicut(): Done with imports"

    if LAZYFLOW_TOTAL_RAM_MB is None:
        # By default, assume our alotted RAM is proportional 
        # to the CPUs we've been told to use
        machine_ram = psutil.virtual_memory().total
        machine_ram -= 1024**3 # Leave 1 GB RAM for the OS.

        LAZYFLOW_TOTAL_RAM_MB = LAZYFLOW_THREADS * machine_ram / multiprocessing.cpu_count()

    # Before we start ilastik, prepare the environment variable settings.
    os.environ["LAZYFLOW_THREADS"] = str(LAZYFLOW_THREADS)
    os.environ["LAZYFLOW_TOTAL_RAM_MB"] = str(LAZYFLOW_TOTAL_RAM_MB)
    os.environ["LAZYFLOW_STATUS_MONITOR_SECONDS"] = "10"

    extra_cmdline_args += ['--output_axis_order=zyx']
    # Prepare ilastik's "command-line" arguments, as if they were already parsed.
    args, extra_workflow_cmdline_args = ilastik_main.parser.parse_known_args(extra_cmdline_args)
    args.headless = True
    args.debug = True # ilastik's 'debug' flag enables special power features, including experimental workflows.
    args.project = str(ilp_path)
    args.readonly = True

    # The process_name argument is prefixed to all log messages.
    # For now, just use the machine name and a uuid
    # FIXME: It would be nice to provide something more descriptive, like the ROI of the current spark job...
    args.process_name = platform.node() + "-" + str(uuid.uuid1())

    # To avoid conflicts between processes, give each process it's own logfile to write to.
    if logfile != "/dev/null":
        base, ext = os.path.splitext(logfile)
        logfile = base + '.' + args.process_name + ext

    # By default, all ilastik processes duplicate their console output to ~/.ilastik_log.txt
    # Obviously, having all spark nodes write to a common file is a bad idea.
    # The "/dev/null" setting here is recognized by ilastik and means "Don't write a log file"
    args.logfile = logfile

    print "ilastik_multicut(): Creating shell..."

    # Instantiate the 'shell', (in this case, an instance of ilastik.shell.HeadlessShell)
    # This also loads the project file into shell.projectManager
    shell = ilastik_main.main( args, extra_workflow_cmdline_args )

    ## Need to find a better way to verify the workflow type
    #from ilastik.workflows.multicutWorkflow import MulticutWorkflow
    #assert isinstance(shell.workflow, MulticutWorkflow)

    # Construct an OrderedDict of role-names -> DatasetInfos
    # (See MulticutWorkflow.ROLE_NAMES)
    raw_data_array = vigra.taggedView(grayscale, 'zyx')
    probabilities_array = vigra.taggedView(bounary_volume, 'zyxc')
    superpixels_array = vigra.taggedView(supervoxels, 'zyx')
    
    role_data_dict = OrderedDict([ ("Raw Data", [ DatasetInfo(preloaded_array=raw_data_array) ]),
                                   ("Probabilities", [ DatasetInfo(preloaded_array=probabilities_array) ]),
                                   ("Superpixels", [ DatasetInfo(preloaded_array=superpixels_array) ]) ])

    print "ilastik_multicut(): Starting export..."

    # Run the export via the BatchProcessingApplet
    segmentation_list = shell.workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=True)
    assert len(segmentation_list) == 1
    segmentation = segmentation_list[0]

    assert segmentation.ndim == 3
    print 'status=multicut finished'
    return segmentation
