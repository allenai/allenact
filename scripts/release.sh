#!/usr/bin/env bash

version=$(cat .VERSION)
echo BUILDING sdists for $version
python3 scripts/release.py

echo UPLOADING sdist for allenact-$version
python3 -m twine upload --repository testpypi dist/allenact-$version.tar.gz

echo UPLOADING sdist for allenact_plugins-$version
python3 -m twine upload --repository testpypi dist/allenact_plugins-$version.tar.gz

echo DONE
