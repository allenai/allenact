#!/bin/bash

echo RUNNING BLACK
pipenv run black *.py */*.py */*/*.py
echo BLACK DONE
echo ""

echo RUNNING DOCFORMATTER
pipenv run docformatter --in-place -r *.py */*.py */*/*.py
echo DOCFORMATTER DONE

echo ALL DONE