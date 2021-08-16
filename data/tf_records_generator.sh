#!/bin/sh

python ../drwr/scripts/tf_records_generator.py \
--split_dir=splits/ \
--inp_dir_renders=renders \
--out_dir=tf_records/ \
--tfrecords_gzip_compressed=True \
--synth_set=$1 \
--image_size=128 \
--store_camera=True \
--store_voxels=False \
--store_depth=True \
--num_views=5
