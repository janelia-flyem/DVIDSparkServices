from DVIDSparkServices.reconutils.misc import select_channels, normalize_channels_in_place

def ilastik_simple_predict(gray_vol, mask, classifier_path, filter_specs_path, selected_channels=None, normalize=True, 
                           LAZYFLOW_THREADS=0, LAZYFLOW_TOTAL_RAM_MB=None, logfile="/dev/null"):
    """
    gray_vol: A 3D numpy array with axes zyx

    mask: A binary image where 0 means "no prediction necessary".
         'None' can be given, which means "predict everything".

    classifier_path: Path to a vigra RandomForest classifier, in HDF5.
                     Example: /path/to/myclassifier.h5/classifiers/my_rf

    filter_specs_path: Path to "filter specs" json file.  The json structure is like this:
                       [ ['GaussianSmoothing', 0.3],
                         ['GaussianSmoothing', 0.7],
                         ['LaplacianOfGaussian', 1.6] ]
                       (See ilastik's simple_predict.py for valid filter names.)
     
    selected_channels: A list of channel indexes to select and return from the prediction results.
                       'None' can also be given, which means "return all prediction channels".
                       You may also return a *nested* list, in which case groups of channels can be
                       combined (summed) into their respective output channels.
                       For example: selected_channels=[0,3,[2,4],7] means the output will have 4 channels:
                                    0,3,2+4,7 (channels 5 and 6 are simply dropped).
    
    normalize: Renormalize all outputs so the channels sum to 1 everywhere.
               That is, (predictions.sum(axis=-1) == 1.0).all()
               Note: Pixels with 0.0 in all channels will be simply given a value of 1/N in all channels.
    
    LAZYFLOW_THREADS, LAZYFLOW_TOTAL_RAM_MB: Same meanings as in ilastik_predict_with_array().
                      (although we have to configure them in a different way)
    """
    print "ilastik_simple_predict(): Starting with raw data: dtype={}, shape={}".format(str(gray_vol.dtype), gray_vol.shape)

    import os
    from collections import OrderedDict

    import uuid
    import platform
    import vigra

    from ilastik.utility.simple_predict import load_and_predict
    from lazyflow.request import Request
    
    print "ilastik_simple_predict(): Done with imports"

    _prepare_lazyflow_config(LAZYFLOW_THREADS, LAZYFLOW_TOTAL_RAM_MB, 10)

    Request.reset_thread_pool(LAZYFLOW_THREADS)

    # The process_name argument is prefixed to all log messages.
    # For now, just use the machine name and a uuid
    # FIXME: It would be nice to provide something more descriptive, like the ROI of the current spark job...
    process_name = platform.node() + "-" + str(uuid.uuid1())

    # To avoid conflicts between processes, give each process it's own logfile to write to.
    if logfile != "/dev/null":
        base, ext = os.path.splitext(logfile)
        logfile = base + '.' + process_name + ext

    _init_logging(logfile, process_name)
    
    # Construct an OrderedDict of role-names -> DatasetInfos
    # (See PixelClassificationWorkflow.ROLE_NAMES)
    raw_data_array = vigra.taggedView(gray_vol, 'zyx')
    print "ilastik_simple_predict(): Starting export..."

    predictions = load_and_predict( raw_data_array, classifier_path, filter_specs_path, compute_blockwise=True ) 
    selected_predictions = select_channels(predictions, selected_channels)

    if normalize:
        normalize_channels_in_place(selected_predictions)
    
    return selected_predictions

#
# COPIED FROM ilastik_main.py (with slight modifications)
#
def _prepare_lazyflow_config(n_threads, total_ram_mb, status_interval_secs):
    import logging
    logger = logging.getLogger(__name__)

    # Check environment variable settings.
    #n_threads = os.getenv("LAZYFLOW_THREADS", None)
    #total_ram_mb = os.getenv("LAZYFLOW_TOTAL_RAM_MB", None)
    #status_interval_secs = int( os.getenv("LAZYFLOW_STATUS_MONITOR_SECONDS", "0") )

    # Convert str -> int
    if n_threads is not None:
        n_threads = int(n_threads)
    total_ram_mb = total_ram_mb and int(total_ram_mb)
    
    # Note that n_threads == 0 is valid and useful for debugging.
    if (n_threads is not None) or total_ram_mb or status_interval_secs:
        def _configure_lazyflow_settings():
            import lazyflow
            import lazyflow.request
            from lazyflow.utility import Memory
            from lazyflow.operators.cacheMemoryManager import CacheMemoryManager

            if status_interval_secs:
                memory_logger = logging.getLogger('lazyflow.operators.cacheMemoryManager')
                memory_logger.setLevel(logging.DEBUG)
                CacheMemoryManager().setRefreshInterval(status_interval_secs)

            if n_threads is not None:
                logger.info("Resetting lazyflow thread pool with {} threads.".format( n_threads ))
                lazyflow.request.Request.reset_thread_pool(n_threads)
            if total_ram_mb > 0:
                if total_ram_mb < 500:
                    raise Exception("In your current configuration, RAM is limited to {} MB."
                                    "  Remember to specify RAM in MB, not GB."
                                    .format( total_ram_mb ))
                ram = total_ram_mb * 1024**2
                fmt = Memory.format(ram)
                logger.info("Configuring lazyflow RAM limit to {}".format(fmt))
                Memory.setAvailableRam(ram)
        return _configure_lazyflow_settings
    return None

def _init_logging(logfile_path, process_name):
    from ilastik.ilastik_logging import default_config
    default_config.init(process_name + " ", default_config.OutputMode.BOTH, logfile_path)
