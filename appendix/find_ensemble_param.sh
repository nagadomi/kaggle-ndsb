#!/bin/sh

th cv_predict.lua -model 48 -seed 101
th cv_predict.lua -model 48 -seed 102
th cv_predict.lua -model 48 -seed 103
th cv_predict.lua -model 48 -seed 104
th cv_predict.lua -model 48 -seed 105
th cv_predict.lua -model 48 -seed 106
th cv_predict.lua -model 48 -seed 107
th cv_predict.lua -model 48 -seed 108

th cv_predict.lua -model 72 -seed 101
th cv_predict.lua -model 72 -seed 102
th cv_predict.lua -model 72 -seed 103
th cv_predict.lua -model 72 -seed 104
th cv_predict.lua -model 72 -seed 105
th cv_predict.lua -model 72 -seed 106
th cv_predict.lua -model 72 -seed 107
th cv_predict.lua -model 72 -seed 108

th cv_predict.lua -model 96 -seed 101
th cv_predict.lua -model 96 -seed 102
th cv_predict.lua -model 96 -seed 103
th cv_predict.lua -model 96 -seed 104
th cv_predict.lua -model 96 -seed 105
th cv_predict.lua -model 96 -seed 106
th cv_predict.lua -model 96 -seed 107
th cv_predict.lua -model 96 -seed 108

th cv_find_param.lua
