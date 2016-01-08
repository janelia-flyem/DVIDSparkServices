from DVIDSparkServices.reconutils.misc import select_channels, normalize_channels_in_place

def two_stage_voxel_predictions(gray_vol, mask, stage_1_ilp_path, stage_2_ilp_path, selected_channels=None, normalize=True, 
                                LAZYFLOW_THREADS=1, LAZYFLOW_TOTAL_RAM_MB=None, logfile="/dev/null", extra_cmdline_args=[]):
    """
    Using ilastik's python API, run a two-stage voxel prediction using the two given project files.
    The output of the first stage will be saved to a temporary location on disk and used as input to the second stage.
    
    gray_vol: A 3D numpy array with axes zyx

    mask: A binary image where 0 means "no prediction necessary".
         'None' can be given, which means "predict everything".
         (It will only be used during the second stage.)

    ilp_stage_1_path: Path to the project file for the first stage.  Should accept graystale uint8 data as the input.
                      ilastik also accepts a url to a DVID key-value, which will be downloaded and opened as an ilp

    ilp_stage_1_path: Path to the project file for the second stage.  Should take N input channels (uint8) as input, 
                      where N is the number of channels produced in stage 1.
    
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
    print "two_stage_voxel_predictions(): Starting with raw data: dtype={}, shape={}".format(str(gray_vol.dtype), gray_vol.shape)

    import tempfile
    import numpy as np
    import h5py

    print "hey"

    scratch_dir = tempfile.mkdtemp()

    # Run predictions on the in-memory data.
    stage_1_output_path = scratch_dir + '/stage_1_predictions.h5'
    run_ilastik_stage(1, stage_1_ilp_path, gray_vol, None, stage_1_output_path,
                      LAZYFLOW_THREADS, LAZYFLOW_TOTAL_RAM_MB, logfile, extra_cmdline_args)
    print "hey"

    stage_2_output_path = scratch_dir + '/stage_2_predictions.h5'
    run_ilastik_stage(2, stage_2_ilp_path, stage_1_output_path, mask, stage_2_output_path,
                      LAZYFLOW_THREADS, LAZYFLOW_TOTAL_RAM_MB, logfile, extra_cmdline_args)

    combined_predictions_path = scratch_dir + 'combined_predictions.h5'

    # Sadly, we must rewrite the predictions into a single file, because they might be combined together.
    # Technically, we could avoid this with some fancy logic, but that would be really annoying.
    with h5py.File(combined_predictions_path, 'w') as combined_predictions_file:
        with h5py.File(stage_1_output_path, 'r') as stage_1_prediction_file, \
             h5py.File(stage_2_output_path, 'r') as stage_2_prediction_file:
            stage_1_predictions = stage_1_prediction_file['predictions']
            stage_2_predictions = stage_2_prediction_file['predictions']
    
            stage_1_channels = stage_1_predictions.shape[-1]
            stage_2_channels = stage_2_predictions.shape[-1]
            
            assert stage_1_predictions.shape[:-1] == stage_2_predictions.shape[:-1]
            
            combined_shape = stage_1_predictions.shape[:-1] + ((stage_1_channels + stage_2_channels),)
            combined_predictions = combined_predictions_file.create_dataset('predictions',
                                                                            dtype=stage_1_predictions.dtype,
                                                                            shape=combined_shape,
                                                                            chunks=(64,64,64,1) )
    
            combined_predictions[..., :stage_1_channels] = stage_1_predictions[:]
            combined_predictions[..., stage_1_channels:] = stage_2_predictions[:]

        num_channels = combined_predictions.shape[-1]
    
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

        # This will extract the channels we want, converting from hdf5 to numpy along the way.    
        selected_predictions = select_channels(combined_predictions, selected_channels)
    
    if normalize:
        normalize_channels_in_place(selected_predictions)
    
    return selected_predictions

def run_ilastik_stage(stage_num, ilp_path, input_vol, mask, output_path,
                      LAZYFLOW_THREADS=1, LAZYFLOW_TOTAL_RAM_MB=None, logfile="/dev/null", extra_cmdline_args=[]):
    import os
    from collections import OrderedDict

    import uuid
    import multiprocessing
    import platform
    import psutil
    import vigra

    import ilastik_main
    from ilastik.applets.dataSelection import DatasetInfo

    print "done imports"

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
    args.project = ilp_path
    args.readonly = True

    # By default, all ilastik processes duplicate their console output to ~/.ilastik_log.txt
    # Obviously, having all spark nodes write to a common file is a bad idea.
    # The "/dev/null" setting here is recognized by ilastik and means "Don't write a log file"
    args.logfile = logfile

    # The process_name argument is prefixed to all log messages.
    # For now, just use the machine name and a uuid
    # FIXME: It would be nice to provide something more descriptive, like the ROI of the current spark job...
    args.process_name = platform.node() + "-" + str(uuid.uuid1()) + "-" + str(stage_num)

    print "loading shell"

    # Instantiate the 'shell', (in this case, an instance of ilastik.shell.HeadlessShell)
    # This also loads the project file into shell.projectManager
    shell = ilastik_main.main( args, extra_workflow_cmdline_args )

    ## Need to find a better way to verify the workflow type
    #from ilastik.workflows.pixelClassification import PixelClassificationWorkflow
    #assert isinstance(shell.workflow, PixelClassificationWorkflow)

    opInteractiveExport = shell.workflow.batchProcessingApplet.dataExportApplet.topLevelOperator.getLane(0)
    opInteractiveExport.OutputFilenameFormat.setValue(output_path)
    opInteractiveExport.OutputInternalPath.setValue('predictions')
    opInteractiveExport.OutputFormat.setValue('hdf5')
    
    selected_result = opInteractiveExport.InputSelection.value
    num_channels = opInteractiveExport.Inputs[selected_result].meta.shape[-1]

    print "constructing input dict"

    # Construct an OrderedDict of role-names -> DatasetInfos
    # (See PixelClassificationWorkflow.ROLE_NAMES)
    if isinstance(input_vol, (str, unicode)):
        role_data_dict = OrderedDict([ ("Raw Data", [ DatasetInfo(filepath=input_vol) ]) ])
    else:
        # If given raw data, we assume it's grayscale, zyx order (stage 1)
        raw_data_array = vigra.taggedView(input_vol, 'zyx')
        role_data_dict = OrderedDict([ ("Raw Data", [ DatasetInfo(preloaded_array=raw_data_array) ]) ])
    
    if mask is not None:
        # If there's a mask, we might be able to save some computation time.
        mask = vigra.taggedView(mask, 'zyx')
        role_data_dict["Prediction Mask"] = [ DatasetInfo(preloaded_array=mask) ]

    print "exporting..."

    # Run the export via the BatchProcessingApplet
    export_paths = shell.workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=False)
    assert len(export_paths) == 1
    assert export_paths[0] == output_path + '/predictions', "Output path was {}".format(export_paths[0])

    print "done exporting"