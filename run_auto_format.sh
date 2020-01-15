#!/bin/bash

pipenv run black *.py **/*.py
pipenv run docformatter --in-place -r *.py */*.py */*/*.py