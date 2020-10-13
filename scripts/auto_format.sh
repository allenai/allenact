#!/bin/bash

echo RUNNING BLACK
black . --exclude src
echo BLACK DONE
echo ""

echo RUNNING DOCFORMATTER
docformatter --in-place -r --exclude src .
echo DOCFORMATTER DONE

echo ALL DONE