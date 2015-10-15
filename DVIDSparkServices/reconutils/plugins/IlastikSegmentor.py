import os
from DVIDSparkServices.reconutils.Segmentor import Segmentor

class IlastikSegmentor(Segmentor):

    def __init__(self, context, config, options):
        super(IlastikSegmentor, self).__init__(context, config, options)
        settings = config["options"]["plugin-configuration"]
        required_settings = ["ilp-path", "LAZYFLOW_THREADS", "LAZYFLOW_TOTAL_RAM_MB"]
        assert set(required_settings).issubset( set(settings.keys()) ), \
            "Missing required settings in config/options/plugin-configuration: {}"\
            .format( set(required_settings) - set(settings.keys()) )
        
        self.ilp_path = os.path.abspath(settings["ilp-path"])
        self.lazyflow_threads = settings["LAZYFLOW_THREADS"]
        self.lazyflow_total_ram_mb = settings["LAZYFLOW_TOTAL_RAM_MB"]

    def predict_voxels(self, gray_chunks): 
        """Create a dummy placeholder boundary channel from grayscale.

        Takes an RDD of grayscale numpy volumes and produces
        an RDD of predictions (z,y,x,ch) and a watershed mask.
        """
        from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
        
        # Can't use 'self' in the closure for mapValues
        ilp_path = self.ilp_path
        lazyflow_threads = self.lazyflow_threads
        lazyflow_total_ram_mb = self.lazyflow_total_ram_mb
        
        def _predict_chunks(gray_chunk):
            (subvolume, gray) = gray_chunk
            
            predictions = ilastik_predict_with_array( ilp_path, gray, lazyflow_threads, lazyflow_total_ram_mb )
            pred_compressed = CompressedNumpyArray(predictions)

            # FIXME?                        
            #mask_compressed = CompressedNumpyArray(mask)
            
            return (subvolume, pred_compressed, None)
        
        # preserver partitioner
        return gray_chunks.mapValues(_predict_chunks)

def ilastik_predict_with_array(project_file_path, raw_data_array, lazyflow_threads, lazyflow_total_ram_mb):
    """
    Using ilastik's python API, open the given project 
    file and run a prediction on the given raw data array.
    
    Other than the project file, nothing is read or written 
    using the hard disk.
    
    raw_data_array: A 3D numpy array with axes zyx
    """
    print "ilastik_predict_with_array(): Starting with raw data: dtype={}, shape={}".format(str(raw_data_array.dtype), raw_data_array.shape)

    import os
    from collections import OrderedDict
    import vigra

    import ilastik_main
    from ilastik.applets.dataSelection import DatasetInfo
    from ilastik.workflows.pixelClassification import PixelClassificationWorkflow

    print "ilastik_predict_with_array(): Done with imports"

    # Before we start ilastik, prepare the environment variable settings.
    os.environ["LAZYFLOW_THREADS"] = str(lazyflow_threads)
    os.environ["LAZYFLOW_TOTAL_RAM_MB"] = str(lazyflow_total_ram_mb)

    # Prepare ilastik's "command-line" arguments, as if they were already parsed.
    args = ilastik_main.parser.parse_args([])
    args.headless = True
    args.project = project_file_path
    args.readonly = True

    print "ilastik_predict_with_array(): Creating shell..."

    # Instantiate the 'shell', (in this case, an instance of ilastik.shell.HeadlessShell)
    # This also loads the project file into shell.projectManager
    shell = ilastik_main.main( args )
    assert isinstance(shell.workflow, PixelClassificationWorkflow)

    # Obtain the training operator
    opPixelClassification = shell.workflow.pcApplet.topLevelOperator

    # Sanity checks
    assert len(opPixelClassification.InputImages) > 0
    assert opPixelClassification.Classifier.ready()

    # Construct an OrderedDict of role-names -> DatasetInfos
    # (See PixelClassificationWorkflow.ROLE_NAMES)
    raw_data_array = vigra.taggedView(raw_data_array, 'zyx')
    role_data_dict = OrderedDict([ ("Raw Data", [ DatasetInfo(preloaded_array=raw_data_array) ]) ]) 

    print "ilastik_predict_with_array(): Starting export..."

    # Run the export via the BatchProcessingApplet
    prediction_list = shell.workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=True)
    assert len(prediction_list) == 1
    predictions = prediction_list[0]

    # Sanity checks
    label_names = opPixelClassification.LabelNames.value
    assert predictions.shape[-1] == len(label_names)
    return predictions
