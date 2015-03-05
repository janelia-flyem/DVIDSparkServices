# Spark Implemented EM Reconstruction Workflows  [![Picture](https://raw.github.com/janelia-flyem/janelia-flyem.github.com/master/images/HHMI_Janelia_Color_Alternate_180x40.png)](http://www.janelia.org)

Provides python utitlies for interacting with EM data stored in DVID.
Several sample workflows are provided as well as infrastructure for custom
plugin modules.

## Installation

Python dependences: jsonschema, pydvid, argparse, importlib

    % python setup build
    % python setup install

*Other dependencies might be required for certain workflows.*

A pre-built spark binary can be downloaded from [http://spark.apache.org/downloads.html](http://spark.apache.org/downloads.html)

Example command:

    % spark-submit --master local[4]  workflows/launchworkflow.py ComputeGraph -c example/config_example.json

This calls the module ComputeGraph with the provide configuration in JSON.  One can supply the flag '-d' instead of '-c' and the config file to retrieve a JSON schema describing the expected input. 


## Workflow Plugins

This section describes some of the workflows available in recospark and the plugin architecture.

**Plugin Architecture**

Recospark python workflows all have the same pattern:

    % workflows/launchworkflow.py WORKFLOWNAME -c CONFIG

Where WORKFLOWNAME should exist as a python file in the module *workflows* and define a class with the same name that inherits from
the python object *recospark.reconutils.workflow.Workflow* or *recospark.reconutils.workflow.DVIDWorkflow*.  launchworkflow.py acts as the entry point that invokes the provide module.

### Compute Graph

This workflow takes a DVID labelsblock volume and ROI and computes a DVID labelgraph.
The ROI is split into several substacks.  Spark will process each substack (plus a 1 pixel overlap)
to compute the size of each vertex and its overlap with neighboring vertices.  This graph information
is aggregated by key across all substacks and written to DVID.  The actual graph construction can be
done with a third-party binary.  In this workflow, it is recommended that [NeuroProof](https://github.com/janelia-flyem/NeuroProof) is installed.

Compute Graph is limited by the throughput of databases behind DVID.  If the DVID backend is not clustered, the accesses
will be throttled and slow.  The algorithm has linear complexity and should otherwise scale to very large datasets.

Example configuration JSON (commens added for convenience but is not valid JSON)

    {
        "dvid-server": "127.0.0.1:8000", # DVID server
        "uuid": "UUID", # DVID uuid
        "label-name": "label-name",
        "roi": "roi-name",
        "graph-name": "graph-name",
        "graph-builder-exe": "neuroproof_graph_build_stream" # binary that builds the graph
    }

### DVID Block Ingest

This workflow takes a stack of 2D images and produces binary chunks of DVID blocks.  The script does not
actually communicate with DVID and can be called independent of some of the recospark libraries.
A separate script is necessary to write the DVID blocks to DVID.  An example of one such script is provided,
in the example directory.  The blocks will be padded with black (0) pixels, so that all blocks are 32x32x32
in size.

Example configuration JSON:

    {
        "minslice" : MINZ, # minimum stack slice number examined
        "maxslice" : MAXZ, # maximum stack slice number examined
        "basename": IMAGEPATH, # image name template
        "output-dir": OUTDIR # directory where the blocks are written to
    }
