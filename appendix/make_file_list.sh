#!/bin/sh

find data/train/ -name "*.jpg" > data/train.txt
find data/test/ -name "*.jpg" | sort > data/test.txt
