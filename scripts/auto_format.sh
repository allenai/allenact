#!/bin/bash

# Move to the directory containing the directory that this file is in
cd "$( cd "$( dirname "${BASH_SOURCE[0]}/.." )" >/dev/null 2>&1 && pwd )" || exit

echo RUNNING BLACK
black . --exclude src --exclude external_projects
echo BLACK DONE
echo ""

echo RUNNING DOCFORMATTER
find . -name "*.py" | grep -v ^./src | grep -v used_configs | xargs docformatter --in-place -r
echo DOCFORMATTER DONE

echo ALL DONE