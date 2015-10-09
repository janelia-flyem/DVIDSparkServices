import sys
import os
import numpy

# Re-use the code in ilastik/bin/train_headless
import ilastik_main
sys.path.append( os.path.split(ilastik_main.__file__)[0] + '/bin')
from train_headless import generate_trained_project_file, ScalesList, FeatureIds

def generate_test_ilp(output_dir, shape=(128, 256, 256)):
    ilp_path = output_dir + '/test-project.ilp'
    data_path = output_dir + '/test-raw-data.npy'
    labels_path = output_dir + '/test-labels.npy'

    print "Using output directory:" + output_dir
    print "Generating test data..."
    write_test_data(data_path, labels_path, shape)

    feature_selections = prepare_feature_selections()
    
    print "Preparing new project file: {}".format( ilp_path )
    from lazyflow.classifiers import ParallelVigraRfLazyflowClassifierFactory
    classifier_factory = ParallelVigraRfLazyflowClassifierFactory(10)
    generate_trained_project_file(ilp_path, [data_path], [labels_path], feature_selections, classifier_factory)

def write_test_data(raw_data_path, label_data_path, shape):
    """
    Produce a simple test volume of grid lines, and a corresponding 
    sparse label volume that goes with it.
    """
    DARK = 0
    GRAY = 100
    
    test_raw_data = GRAY*numpy.ones(shape, dtype=numpy.uint8)

    block_size = 64
    # Just draw 'membranes as a grid of planes cutting through the volume    
    for dim in range(len(shape)):
        for pos in range(0, shape[dim], block_size):
            line_slice = (slice(None),) * dim + (slice(pos, pos+4),) + (slice(None),)*(len(shape)-dim-1)
            test_raw_data[line_slice] = DARK    
    numpy.save(raw_data_path, test_raw_data)

    label_data = numpy.zeros(shape, dtype=numpy.uint8)
    # Provide some labels, too.
    # First, non-membrane
    for dim in range(len(shape)):
        for pos in range(0, shape[dim], 2*block_size):
            if pos+12 < shape[dim]:
                line_slice = (slice(None),) * dim + (slice(pos+4, pos+12),) + (slice(None),)*(len(shape)-dim-1)
                label_data[line_slice] = 2
            if pos-8 < shape[dim]:
                line_slice = (slice(None),) * dim + (slice(pos-8, pos),) + (slice(None),)*(len(shape)-dim-1)
                label_data[line_slice] = 2

    # Make sure membranes are not labeled yet.
    for dim in range(len(shape)):
        for pos in range(0, shape[dim], block_size):
            line_slice = (slice(None),) * dim + (slice(pos, pos+4),) + (slice(None),)*(len(shape)-dim-1)
            label_data[line_slice] = 0

    # Next, membrane
    for dim in range(len(shape)):
        for pos in range(0, shape[dim], 2*block_size):
            line_slice = (slice(None),) * dim + (slice(pos, pos+4),) + (slice(None),)*(len(shape)-dim-1)
            label_data[line_slice] = 1

    numpy.save(label_data_path, label_data)

def prepare_feature_selections():
    """
    Returns a matrix of hard-coded feature selections.
    See below.
    """
    # #                    sigma:   0.3    0.7    1.0    1.6    3.5    5.0   10.0
    # selections = numpy.array( [[False, False, False, False, False, False, False],
    #                            [False, False, False, False, False, False, False],
    #                            [False, False, False, False, False, False, False],
    #                            [False, False, False, False, False, False, False],
    #                            [False, False, False, False, False, False, False],
    #                            [False, False, False, False, False, False, False]] )
    
    # Start with an all-False matrix and apply the features we want.
    selections = numpy.zeros( (len(FeatureIds), len(ScalesList)), dtype=bool )
    def set_feature(feature_id, scale):
        selections[ FeatureIds.index(feature_id), ScalesList.index(scale) ] = True
    
    set_feature('GaussianSmoothing',         0.3)
    set_feature('GaussianSmoothing',         1.0)
    set_feature('GaussianGradientMagnitude', 1.0)

    return selections

if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")    
    parsed_args = parser.parse_args()
    
    generate_test_ilp(parsed_args.output_dir)
    print "DONE."
