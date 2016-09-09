Segmentor Plugins
=================

The `DVIDSparkServices.reconutils.plugins` package provides some "batteries-included" functions for overriding the default behavior of the `Segmentor` class.  As described on the [DVIDSparkServices wiki](https://github.com/janelia-flyem/DVIDSparkServices/wiki/Create-Segmentation),
the `Segmentor.segment()` method divides divides the work into the following steps, each of which can be customized with an arbtirary python function:

1. `background-mask`: Detect large 'background' regions that lie outside the area of interest for segmentation.
2. `predict-voxels`: Predict voxel classes for every grayscale pixel (return an N-channel volume, `float32`).
3. `create-supervoxels`: Create a label volume of supervoxels (`uint32`).
4. `agglomerate-supervoxels`: Aggregate supervoxels into final segments (`uint32`).


Background Masking Functions
----------------------------

So far, we use a simple method for detecting regions without valid image data.

- `DVIDSparkServices.reconutils.plugins.misc.find_large_empty_regions()`
  
   Finds large contiguous blobs of 0-valued pixels, and as mask volume wehere 1 means 'valid data here' and 0 means 'background'.
   By convention, this function is permitted to return `None`, which means 'everything is valid, no background'.


Voxel Classification Functions
------------------------------

These functions can be used for the `predict-voxels`

**Standard ilastik voxel prediction**

- [`DVIDSparkServices.reconutils.plugins.ilastik_predict_with_array.ilastik_predict_with_array()`](./ilastik_predict_with_array.py)

   See docstring for details.  This method (though not this exact function) was used for the FIB-25 7-column dataset.

   Example config:
   ```json
   ...
    "predict-voxels" : {
        "function": "DVIDSparkServices.reconutils.plugins.IlastikSegmentor.ilastik_predict_with_array",
        "parameters": {
            "ilp_path": "/groups/flyem/data/scratchspace/classifiers/fib25-multicut/dots_ilastik_1.0_upgraded-4d-retrained.ilp",
            "selected_channels": [0],
            "normalize": false,
            "LAZYFLOW_THREADS": 2,
            "LAZYFLOW_TOTAL_RAM_MB": 16000,
            "logfile": "/groups/flyem/data/scratchspace/classifiers/fib25-multicut/logs/fib25-multicut-segmentation-2-log.txt"
        }
    },
   ...
   ```

**Two-stage ilastik voxel prediction (a.k.a. 2-stage "Autocontext")**

- [`DVIDSparkServices.reconutils.plugins.two_stage_voxel_predictions.two_stage_voxel_predictions()`](./two_stage_voxel_predictions.py)

   Runs ilastik voxel prediction, and then run it again on the results.  See docstring for details.  Was used for FIB-19.

   Example config:
   
   ```json
   ...
    "predict-voxels" : {
        "function": "DVIDSparkServices.reconutils.plugins.two_stage_voxel_predictions.two_stage_voxel_predictions",
        "parameters": {
            "stage_1_ilp_path": "/groups/flyem/data/scratchspace/classifiers/fib19_experimental/two-stage-ilp/ordishc_closing_caps_5_8channels-linked-input.ilp",
            "stage_2_ilp_path": "/groups/flyem/data/scratchspace/classifiers/fib19_experimental/two-stage-ilp/noAC_ordishc_closing_caps_5_8channels_2pass.ilp",
            "selected_channels": [8,1,2,3],
            "normalize": true,
            "LAZYFLOW_THREADS": 1,
            "LAZYFLOW_TOTAL_RAM_MB": 4000,
            "logfile" : "/groups/flyem/data/scratchspace/sparkjoblogs/prod-substacks.txt"
        }
    },
   ...
   ```

**"Simple" ilastik prediction**

- [`DVIDSparkServices.reconutils.plugins.ilastik_simple_predict.ilastik_simple_predict()`](./ilastik_simple_predict.py)

   This one is embarrassing.  Toufiq has *his own* scripts for generating his own pixel classifiers.  Fortunately, they were designed to use the *same features* and random forest format that ilastik uses. So, in theory it would be possible to import the feature choices and classifier directly into an ilastik project file.  But unfortunately, for historical reasons, ilastik's voxel features are slightly falsely advertised.  Therefore, the features Toufiq uses are not quite the same as ilastik's features, and therefore his custom-produced classifiers do not quite produce the same results when used within ilastik.

   The "simple predict" plugin re-implements pixel prediction without using ilastik per se.  It is compatible with the pixel classifiers produced using Toufiq's scripts (and *not* compatible with the classifier in a `.ilp` file).  This plugin function calls ilastik's [`simple_predict.py`](https://github.com/ilastik/ilastik/blob/master/ilastik/utility/simple_predict.py) utility script to perform the prediction.  Instead of providing an `.ilp` file to the plugin, you must provide (1) a saved vigra RF classifer in `.h5` format, and (2) a special JSON file describing the order of the training features that the classifier was generated with.  See the `ilastik_simple_predict()` docstring for details.
   
   Was used for the CX/PB.
   
   Example config:
   
   ```json
   ...
    "predict-voxels": {
        "function": "DVIDSparkServices.reconutils.plugins.ilastik_simple_predict.ilastik_simple_predict",
        "parameters": {
            "classifier_path": "/groups/flyem/data/scratchspace/classifiers/fib25-july/pixel_classifier_4class_2.5_600000_10_800_1000_1.0_6.h5/rf",
            "filter_specs_path": "/groups/flyem/data/scratchspace/classifiers/fib25-july/filter-specs-pixel_classifier_4class_2.5_600000_10_800_1000_1.0_6.json",
            "normalize": false,
            "LAZYFLOW_THREADS": 2,
            "LAZYFLOW_TOTAL_RAM_MB": 8000,
            "logfile": "/groups/flyem/data/scratchspace/classifiers/fib25-july/logs/fib25-july2016-ilastik-log.txt"
        }
    },
   ...
   ```
   
   Example filter specs file:
   
   ```json
    [
        [ "GaussianSmoothing", 0.3 ],
        [ "GaussianSmoothing", 0.7 ],
        [ "LaplacianOfGaussian", 0.7 ],
        [ "LaplacianOfGaussian", 1.6 ],
        [ "LaplacianOfGaussian", 3.5 ],
        [ "GaussianGradientMagnitude", 0.7 ],
        [ "GaussianGradientMagnitude", 1.6 ],
        [ "GaussianGradientMagnitude", 3.5 ],
        [ "DifferenceOfGaussians", 0.7 ],
        [ "DifferenceOfGaussians", 1.6 ],
        [ "DifferenceOfGaussians", 3.5 ],
        [ "StructureTensorEigenvalues", 0.7 ],
        [ "StructureTensorEigenvalues", 1.6 ],
        [ "StructureTensorEigenvalues", 3.5 ],
        [ "HessianOfGaussianEigenvalues", 0.7],
        [ "HessianOfGaussianEigenvalues", 1.6],
        [ "HessianOfGaussianEigenvalues", 3.5]
    ]
   ```

- `DVIDSparkServices.reconutils.plugins.misc.naive_membrane_predictions()`

   Not a real prediction function; used for testing only.  See docstring for details.


Watershed Functions
-------------------

**Seeded Watershed**

- `DVIDSparkServices.reconutils.plugins.misc.seeded_watershed()`

   Generates supervoxels via a seeded watershed.  See docstring for details.  This method (though not this exact function) was used for the FIB-25 7-column dataset.

   Example config:
   
   ```json
   ...
    "create-supervoxels": {
        "function": "DVIDSparkServices.reconutils.misc.seeded_watershed",
        "parameters": {
            "boundary_channel": 1,
            "seed_threshold": 0,
            "seed_size": 1,
            "min_segment_size": 300
        }
    },
   ...
   ```

**Watershed over Distance-Transform, a.k.a "wsdt"**
   
- [`DVIDSparkServices.reconutils.plugins.create_supervoxels_with_wsdt.create_supervoxels_with_wsdt()`](./create_supervoxels_with_wsdt.py)

   First computes a distance transform to the thresholded membrane probabilities, then performs watershed on top of that.  See the [wsdt source code](https://github.com/ilastik/wsdt) and [unit tests](https://github.com/ilastik/wsdt/blob/master/tests/testWsDtSegmentation.py) for algorithm details.  The ilastik "Edge Training With Multicut" workflow includes a UI for computing WSDT superpixels, which is useful for experimentating with the parameters (on small volumes).   The most important parameter is the probability threshold.  For the other parameters, it's generally easy to guess reasonable values.
   
   Helps close holes in bad membranes. Was used for FIB-19.

   Example config:
   
   ```json
   ...
    "create-supervoxels" : {
        "function": "DVIDSparkServices.reconutils.plugins.create_supervoxels_with_wsdt.create_supervoxels_with_wsdt",
        "parameters": {
            "pmin": 0.5,
            "minMembraneSize": 0,
            "minSegmentSize": 200,
            "sigmaMinima": 3.0,
            "sigmaWeights": 0.0
        }
    },
   ...
   ```

Agglomeration Functions
-----------------------

**NeuroProof Agglomeration**

- [`DVIDSparkServices.reconutils.plugins.NeuroProofAgglom.neuroproof_agglomerate()`](./NeuroProofAgglom.py)

   Agglomerate supervoxels with `NeuroProof`.  Requires a saved NeuroProof classifier in either `.xml` or `.h5` format.  See docstring for details.  Used for all production segmentations to date.
  
   Example config:
   
   ```json
   ...
    "agglomerate-supervoxels": {
        "function": "DVIDSparkServices.reconutils.plugins.NeuroProofAgglom.neuroproof_agglomerate",
        "parameters": {
            "classifier": {
                "path": "/groups/flyem/data/scratchspace/classifiers/fib25-july/int_classifier2_600000_1000_800_e10000_456.xml"
            },
            "threshold": 0.36,
            "mitochannel": 2
        }
    }
   ...
   ```

**Multicut Agglomeration**

- [`DVIDSparkServices.reconutils.plugins.ilastik_multicut.ilastik_multicut()`](./ilastik_multicut.py)

   Agglomerate supervoxels using ilastik's Multicut workflow.  Not production-ready at the time of this writing.  Requires a trained ilastik project file using workflow type "Edge Training With Multicut".

