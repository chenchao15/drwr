#!/usr/bin/env python

import startup

import os
import numpy as np
import tensorflow as tf

from util.app_config import config as app_config
from util.train import get_path

from scripts.predict import compute_predictions as compute_predictions_pc
from scripts.eval_chamfer_distance import run_eval


def compute_eval():
    cfg = app_config

    dataset = compute_predictions_pc()
    res = run_eval(dataset)

    return res


def test_one_step(index):
    cfg = app_config
    train_dir = get_path(cfg)
    name = os.path.join(train_dir, 'result.txt')
    cfg.test_step = index
    result = compute_eval()
    with open(min_name, 'a+') as f:
        f.write(str(cfg.test_step) + ': ' + str(result) + '\n')


def main(_):
    cfg = app_config
    train_dir = get_path(cfg)
    res = []
    index = [100000]
    for i in index:    
        cfg.test_step = i
        a1, a2 = compute_eval()
        res.append(i)
        res.append(a1)
        with open(os.path.join(train_dir, 'result.txt'), 'w') as f:
            for i in res:
                f.write(str(i) + '\n')
    


if __name__ == '__main__':
    tf.app.run()
