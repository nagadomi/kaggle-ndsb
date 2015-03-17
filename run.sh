#!/bin/sh

if test 3 -ne $# ; then
    echo "${0} <model size> <split method> <random seed>"
    exit 1
fi
th train.lua -model $1 -split $2 -seed $3
th predict.lua -model $1 -split $2 -seed $3
