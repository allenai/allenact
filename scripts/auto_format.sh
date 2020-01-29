#!/bin/bash

echo RUNNING BLACK
black .
echo BLACK DONE
echo ""

echo RUNNING DOCFORMATTER
docformatter --in-place -r .
echo DOCFORMATTER DONE

echo ALL DONE