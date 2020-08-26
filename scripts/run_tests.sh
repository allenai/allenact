#!/usr/bin/env bash

echo RUNNING PYTEST WITH COVERAGE
pipenv run coverage run -m --source=. pytest tests/
echo DONE
echo ""

echo GENERATING COVERAGE HTML
coverage html
echo HTML GENERATED

if [ "$(uname)" == "Darwin" ]; then
    echo OPENING COVERAGE INFO
    open htmlcov/index.html
fi