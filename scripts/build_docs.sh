#!/usr/bin/env bash

set -e

# Alter the relative path of the README image for the docs.
#sed -i '1s/docs/./' docs/README.md
python scripts/build_docs.py

