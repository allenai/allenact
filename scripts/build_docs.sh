#!/usr/bin/env bash

set -e

# Add allenact to the python path
export PYTHONPATH=$PYTHONPATH:$PWD

# Alter the relative path of the README image for the docs.
#sed -i '1s/docs/./' docs/README.md
python scripts/build_docs.py

