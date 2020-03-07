#!/usr/bin/env bash

exec nvidia-docker run --privileged -it -v ~/work:/root/work -p 4711:6006 jordisai2/embodiedrl:latest
