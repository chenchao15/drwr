#!/bin/sh

python ../drwr/densify/densify.py \
--shapenet_path=dataset/ShapeNetCore.v1 \
--python_interpreter=python3 \
--synth_set=$1 \
--subset=val \
--output_dir=gt/dense
