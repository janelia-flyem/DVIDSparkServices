# Spark Implemented EM Reconstruction Workflows [![Picture](https://raw.github.com/janelia-flyem/janelia-flyem.github.com/master/images/jfrc_grey_180x40.png)](http://www.janelia.org) 

(Not ready for use)

Provides python utitlies for interacting with EM data stored in DVID.
Several sample workflows are provided as well as infrastructure for custom
plugin modules.

## Installation

Python dependences: jsonschema and pydvid.

    % python setup build
    % python setup install

Example command:

    % spark-submit --master local[4]  workflows/ComputeGraph.py example/config.json
