def ilastik_predict_with_array(gray_vol, mask, ilp_path, boundary_channels=[0], LAZYFLOW_THREADS=1, LAZYFLOW_TOTAL_RAM_MB=None):
    """
    Using ilastik's python API, open the given project 
    file and run a prediction on the given raw data array.
    
    Other than the project file, nothing is read or written 
    using the hard disk.
    
    raw_data_array: A 3D numpy array with axes zyx
    """
    print "ilastik_predict_with_array(): Starting with raw data: dtype={}, shape={}".format(str(gray_vol.dtype), gray_vol.shape)

    import os
    from collections import OrderedDict

    import multiprocessing
    import psutil
    import vigra

    import ilastik_main
    from ilastik.applets.dataSelection import DatasetInfo
    from ilastik.workflows.pixelClassification import PixelClassificationWorkflow

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

    # Prepare ilastik's "command-line" arguments, as if they were already parsed.
    args = ilastik_main.parser.parse_args([])
    args.headless = True
    args.project = ilp_path
    args.readonly = True
    args.debug = True # ilastik's 'debug' flag enables special power features, including experimental workflows.

    print "ilastik_predict_with_array(): Creating shell..."

    # Instantiate the 'shell', (in this case, an instance of ilastik.shell.HeadlessShell)
    # This also loads the project file into shell.projectManager
    shell = ilastik_main.main( args )
    assert isinstance(shell.workflow, PixelClassificationWorkflow)

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
    num_channels = opInteractiveExport.Inputs[0].meta.shape[-1]
    
    assert all(c < num_channels for c in boundary_channels), \
        "Specified boundary channels ({}) exceed number of prediction classes ({})"\
        .format( boundary_channels, num_channels )

    # Run the export via the BatchProcessingApplet
    prediction_list = shell.workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=True)
    assert len(prediction_list) == 1
    predictions = prediction_list[0]

    assert predictions.shape[-1] == num_channels

    # Select channels for predictions
    # Elsewhere, this will be aggregated from 4D (with channel) to 3D (channels combined)
    return predictions[..., boundary_channels]
