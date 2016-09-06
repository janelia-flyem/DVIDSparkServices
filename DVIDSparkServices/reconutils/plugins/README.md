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

**Two-stage ilastik voxel prediction (a.k.a. 2-stage "Autocontext")**

- [`DVIDSparkServices.reconutils.plugins.two_stage_voxel_predictions.two_stage_voxel_predictions()`](./two_stage_voxel_predictions.py)

   Runs ilastik voxel prediction, and then run it again on the results.  See docstring for details.  Was used for FIB-19.

**"Simple" ilastik prediction**

- [`DVIDSparkServices.reconutils.plugins.ilastik_simple_predict.ilastik_simple_predict()`](./ilastik_simple_predict.py)

   This one is embarrassing.  Toufiq has scripts for generating his own pixel classifiers, but they were designed to use the same features and random forest format that ilastik uses. In theory it would be possible to import the feature choices and classifier directly into an ilastik project file.  But for historical reasons, ilastik's voxel predictions are slightly off.  Therefore, Toufiq's custom-produced classifiers are not compatible with ilastik.

   The "simple predict" plugin re-implements pixel prediction without using ilastik per se.  It is compatible with the pixel classifiers produced using Toufiq's scripts (and *not* compatible with the classifier in a `.ilp` file).  This plugin function calls ilastik's [`simple_predict.py`](https://github.com/ilastik/ilastik/blob/master/ilastik/utility/simple_predict.py) utility script to perform the prediction.  Instead of providing an `.ilp` file to the plugin, you must provide (1) a saved vigra RF classifer in `.h5` format, and (2) a special JSON file describing the order of the training features that the classifier was generated with.  See the `ilastik_simple_predict()` docstring for details.
   
   Was used for the CX/PB.

- `DVIDSparkServices.reconutils.plugins.misc.naive_membrane_predictions()`

   Not a real prediction function; used for testing only.  See docstring for details.


Watershed Functions
-------------------

**Seeded Watershed**

- `DVIDSparkServices.reconutils.plugins.misc.seeded_watershed()`

   Generates supervoxels via a seeded watershed.  See docstring for details.  This method (though not this exact function) was used for the FIB-25 7-column dataset.

**Watershed over Distance-Transform, a.k.a "wsdt"**
   
- [`DVIDSparkServices.reconutils.plugins.create_supervoxels_with_wsdt.create_supervoxels_with_wsdt()`](./create_supervoxels_with_wsdt.py)

   First computes a distance transform to the thresholded membrane probabilities, then performs watershed on top of that.  See the [wsdt source code](https://github.com/ilastik/wsdt) and [unit tests](https://github.com/ilastik/wsdt/blob/master/tests/testWsDtSegmentation.py) for algorithm details.  The ilastik "Edge Training With Multicut" workflow includes a UI for computing WSDT superpixels, which is useful for experimentating with the parameters (on small volumes).   The most important parameter is the probability threshold.  For the other parameters, it's generally easy to guess reasonable values.
   
   Helps close holes in bad membranes. Was used for FIB-19.


**Agglomeration Functions**

- `DVIDSparkServices.reconutils.plugins.NeuroProofAgglom.neuroproof_agglomerate()`

   Agglomerate supervoxels with `NeuroProof`.  Requires a saved NeuroProof classifier in either `.xml` or `.h5` format.  See docstring for details.  Used for all production segmentations to date.
  
- `DVIDSparkServices.reconutils.plugins.ilastik_multicut.ilastik_multicut()`

   Agglomerate supervoxels using ilastik's Multicut workflow.  Not production-ready at the time of this writing.  Requires a trained ilastik project file using workflow type "Edge Training With Multicut".

