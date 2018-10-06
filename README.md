# Spark Implemented EM Reconstruction Workflows  [![Picture](https://raw.github.com/janelia-flyem/janelia-flyem.github.com/master/images/HHMI_Janelia_Color_Alternate_180x40.png)](http://www.janelia.org)

This package provides python Spark utilities for interacting with EM data stored in [DVID](https://github.com/janelia-flyem/dvid).
Several workflows are provided, such as large-scale image segmentation, region-adjacency-graph building, and evaluating the similarity between two image segmentations.  DVIDSparkServices provides an infrastructure for custom workflow and segmentation plugins and a library for accessing DVID through Spark RDDs.

The primary goal of this package is to better analyze and manipulate large EM datasets used in Connectomics (such as those needed for the [the Fly EM project](https://www.janelia.org/project-team/fly-em)).  Other applications that leverage DVID might also benefit from this infrastructure.

Please consult the corresponding wiki for more details on the implemented plugins and other architectural discussions: [https://github.com/janelia-flyem/DVIDSparkServices/wiki](https://github.com/janelia-flyem/DVIDSparkServices/wiki)

## Installation
To simplify the build process, we now use the [conda-build](http://conda.pydata.org/docs/build.html) tool.
The resulting binary is uploaded to the [flyem binstar channel](https://binstar.org/flyem),
and can be installed using the [conda](http://conda.pydata.org/) package manager (instructions below).  The installation
will install all of the DVIDSparkServices dependencies including python.

If one desires to build all the dependencies outside of conda, please consult the recipe found in [Fly EM's conda recipes](https://github.com/janelia-flyem/flyem-build-conda.git) under *dvidsparkservices*/meta.yaml.

*Note: other dependencies might be required for custom plugin workflows.*

### CONDA Installation
The [Miniconda](http://conda.pydata.org/miniconda.html) tool first needs to installed:

```
# Install miniconda to the prefix of your choice, e.g. /my/miniconda

# LINUX:
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh

# MAC:
wget https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
bash Miniconda-latest-MacOSX-x86_64.sh

# Activate conda
CONDA_ROOT=`conda info --root`
source ${CONDA_ROOT}/bin/activate root
```
Once conda is in your system path, call the following to install dvidsparkservices:

    % conda create -n <NAME> -c flyem dvidsparkservices
    
Conda allows builders to create multiple environments (< NAME >).  To use DVIDSparkServices
set your executable path to PREFIX/< NAME >/ bin.

### Installation Part 2

The conda package builder is needed primarily to build DVIDSparkServices' dependencies.  Once the package is installed, it is recommended that users clone this repo and then install the latest python files:

    % python setup.py build
    % python setup.py install
    
One needs to make sure their path is set (mentioned in the previous section) to use the python installed with Conda.

### PySpark

To use the libraries, you need to [download][spark-downloads] Spark.
To facilitate debugging, use a pre-built version of Spark.
The "[Pre-built for Hadoop 2.6 and later][spark-tarball]" version is recommended if you aren't sure which version to download.

[spark-downloads]: http://spark.apache.org/downloads.html
[spark-tarball]: http://www.apache.org/dyn/closer.cgi/spark/spark-1.4.1/spark-1.4.1-bin-hadoop2.6.tgz

## General Usage

Once installed, the workflows can be called even with a local spark context.  This is very useful for debugging and analyzing the performance of small jobs.

Example command:

    % spark-submit --master local[4]  DVIDSparkServices/workflow/launchworkflow.py ComputeGraph -c <configfile>.json

This calls the module ComputeGraph with the provided configuration in JSON using 4 spark workers.  Each plugin defines its own configuration.
One can supply the flag '-d' instead of '-c' and the config file to retrieve a JSON schema describing the expected input.  

For examples of how to run the various workflows, please consult the integration_tests and below.

## Testing

To test the correctness of the package, integration tests are supplied for most of the available workflows.  To run the regressions, one must have a local version of the DVID server running on port 8000 and spark-submit must be in the runtime path.  Then the following command initializes DVID datastructures and runs different workflows:

    # Prerequisite: Unzip the test data
    % gunzip integration_tests/resources/*.gz
    
    # Run tests 
    % python integration_tests/launch_tests.py

## Workflow Plugins

This section describes some of the workflows available in DVIDSparkServices and the plugin architecture.  For more details, please consult the corresponding [wiki](https://github.com/janelia-flyem/DVIDSparkServices/wiki).

**Plugin Architecture**

DVIDSparkServices python workflows all have the same pattern:

    % spark-submit DVIDSparkServices/workflow/launchworkflow.py WORKFLOWNAME -c CONFIG

Where WORKFLOWNAME should exist as a python file in the module *workflows* and define a class with the same name that inherits from
the python object *DVIDSparkServices.workflow.Workflow* or *DVIDSparkServices.workflow.DVIDWorkflow*.  launchworkflow.py acts as the entry point that invokes the provided module.

### Evaluate Segmentation (plugin: EvaluateSeg)

This workflow takes the location of two DVID segmentations and computes quality metrics over them.
An ROI is provided and is split up between spark workers.  Each worker fetches two volumes corresponding to the baseline/ground truth volume and
the comparison volume. 

Custom metric plugins can be created and added to DVIDSparkServices/reconutils/metrics/plugins.  There are many example plugins in this directory that implement different subsets of the base class interface defined at DVIDSparkServices/reconutils/metrics/plugins/stat.py.

### Segmentation (plugin: CreateSegmentation)

Performs segmentation on overlapping grayscale subvolumes fetched from DVID using a supplied segmentation plugin (or falls back to a simple default watershed implementation).  The subvolumes are then stitched together using conservative overlap rules to form a global segmentation.  This segmentation is then read back into DVID.  The default segmentation algorithm
is essentially a no-op and requires the implementation of custom segmentation plugins.  Please see the wiki.

TODO:

* Add options for anisotropic volumes
* Add NeuroProof segmentation


### Compute Graph (plugin: ComputeGraph)

This workflow takes a DVID labelsblock volume and ROI and computes a DVID labelgraph.
The ROI is split into several substacks.  Spark will process each subvolume (plus a 1 pixel overlap)
to compute the size of each vertex and its overlap with neighboring vertices.  This graph information
is aggregated by key across all substacks and written to DVID.  The actual graph construction per subvolume is done
using [NeuroProof](https://github.com/janelia-flyem/NeuroProof).


### DVID Block Ingest (plugin: IngestGrayscale)

This workflow takes a stack of 2D images and produces binary chunks of DVID blocks.  The script does not
actually communicate with DVID and can be called independent of sparkdvid.
A separate script is necessary to write the DVID blocks to DVID.  An example of one such script is provided,
in the example directory.  The blocks will be padded with black (0) pixels, so that all blocks are 32x32x32
in size.

TODO: Add option to write blocks directly to DVID.

## TODO

* Change compute graph to have idempotent edge and vertex insertion to guard against any Spark task failure.
* Increase modularity of the segmentation workflow to allow for easier plugins.
* Expand the sparkdvid API to handle more DVID datatypes
