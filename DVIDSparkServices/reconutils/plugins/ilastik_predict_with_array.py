from DVIDSparkServices.reconutils.misc import select_channels, normalize_channels_in_place

def ilastik_predict_with_array(gray_vol, mask, ilp_path, selected_channels=None, normalize=True, 
                               LAZYFLOW_THREADS=1, LAZYFLOW_TOTAL_RAM_MB=None, logfile="/dev/null", extra_cmdline_args=[]):
    """
    Using ilastik's python API, open the given project 
    file and run a prediction on the given raw data array.
    
    Other than the project file, nothing is read or written 
    using the hard disk.
    
    gray_vol: A 3D numpy array with axes zyx

    mask: A binary image where 0 means "no prediction necessary".
         'None' can be given, which means "predict everything".

    ilp_path: Path to the project file.  ilastik also accepts a url to a DVID key-value, which will be downloaded and opened as an ilp
    
    selected_channels: A list of channel indexes to select and return from the prediction results.
                       'None' can also be given, which means "return all prediction channels".
                       You may also return a *nested* list, in which case groups of channels can be
                       combined (summed) into their respective output channels.
                       For example: selected_channels=[0,3,[2,4],7] means the output will have 4 channels:
                                    0,3,2+4,7 (channels 5 and 6 are simply dropped).
    
    normalize: Renormalize all outputs so the channels sum to 1 everywhere.
               That is, (predictions.sum(axis=-1) == 1.0).all()
               Note: Pixels with 0.0 in all channels will be simply given a value of 1/N in all channels.
    
    LAZYFLOW_THREADS, LAZYFLOW_TOTAL_RAM_MB: Passed to ilastik via environment variables.
    """
    print "ilastik_predict_with_array(): Starting with raw data: dtype={}, shape={}".format(str(gray_vol.dtype), gray_vol.shape)

    import os
    from collections import OrderedDict

    import uuid
    import multiprocessing
    import platform
    import psutil
    import vigra

    import ilastik_main
    from ilastik.applets.dataSelection import DatasetInfo

    print "ilastik_predict_with_array(): Done with imports"

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

    print "ilastik_predict_with_array(): Creating shell..."

    # Instantiate the 'shell', (in this case, an instance of ilastik.shell.HeadlessShell)
    # This also loads the project file into shell.projectManager
    shell = ilastik_main.main( args, extra_workflow_cmdline_args )

    ## Need to find a better way to verify the workflow type
    #from ilastik.workflows.pixelClassification import PixelClassificationWorkflow
    #assert isinstance(shell.workflow, PixelClassificationWorkflow)

    # Construct an OrderedDict of role-names -> DatasetInfos
    # (See PixelClassificationWorkflow.ROLE_NAMES)
    raw_data_array = vigra.taggedView(gray_vol, 'zyx')
    role_data_dict = OrderedDict([ ("Raw Data", [ DatasetInfo(preloaded_array=raw_data_array) ]) ])
    
    if mask is not None:
        # If there's a mask, we might be able to save some computation time.
        mask = vigra.taggedView(mask, 'zyx')
        role_data_dict["Prediction Mask"] = [ DatasetInfo(preloaded_array=mask) ]

    print "ilastik_predict_with_array(): Starting export..."

    # Sanity checks
    opInteractiveExport = shell.workflow.batchProcessingApplet.dataExportApplet.topLevelOperator.getLane(0)
    selected_result = opInteractiveExport.InputSelection.value
    num_channels = opInteractiveExport.Inputs[selected_result].meta.shape[-1]
    
    # For convenience, verify the selected channels before we run the export.
    if selected_channels:
        assert isinstance(selected_channels, list)
        for selection in selected_channels:
            if isinstance(selection, list):
                assert all(c < num_channels for c in selection), \
                    "Selected channels ({}) exceed number of prediction classes ({})"\
                    .format( selected_channels, num_channels )
            else:
                assert selection < num_channels, \
                    "Selected channels ({}) exceed number of prediction classes ({})"\
                    .format( selected_channels, num_channels )
                

    # Run the export via the BatchProcessingApplet
    prediction_list = shell.workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=True)
    assert len(prediction_list) == 1
    predictions = prediction_list[0]

    assert predictions.shape[-1] == num_channels
    selected_predictions = select_channels(predictions, selected_channels)

    if normalize:
        normalize_channels_in_place(selected_predictions)
    
    return selected_predictions
