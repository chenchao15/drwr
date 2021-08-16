#!/bin/sh

python ../drwr/densify/downsample_gt.py \
--inp_dir=gt/dense \
--out_dir=gt/downsampled \
--synth_set=$1