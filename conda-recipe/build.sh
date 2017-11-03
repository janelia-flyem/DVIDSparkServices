#!/bin/bash

# https://github.com/conda-forge/staged-recipes/issues/528
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
