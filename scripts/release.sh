#!/usr/bin/env bash

version=$(cat .VERSION)
echo BUILDING sdists for $version
python3 scripts/release.py

if [ "$1" = "live" ]
then
  TWINE_COMMAND="twine upload"
else
  TWINE_COMMAND="twine upload --repository testpypi"
  echo "NOTE: Use  scripts/release.sh live  to upload to main PyPI!"
fi

echo "UPLOADING sdists for"
echo "allenact-$version and allenact_plugins-$version"
echo "the username is __token__"
python3 -m $TWINE_COMMAND dist/allenact*-$version.tar.gz

echo "DONE $version"
