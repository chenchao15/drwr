import tensorflow as tf
import tensorflow.contrib.slim as slim


def model(inputs, outputs_all, cfg, is_training):
    num_points = cfg.pc_num_points

    init_stddev = cfg.pc_decoder_init_stddev
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)
    msra_init = tf.contrib.layers.variance_scaling_initializer()
    z_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)
    b_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)

    pts_raw = slim.fully_connected(inputs, num_points * 3,
                                   activation_fn=None,
                                   weights_initializer=w_init)

    z_raw = slim.fully_connected(inputs, num_points,activation_fn=tf.nn.relu,weights_initializer=z_init)
    z_raw = tf.add(z_raw, 0.01)
    #z_raw = tf.reshape(z_raw, [z_raw.shape[0], num_points, num_points])
    b_raw = slim.fully_connected(inputs,num_points, activation_fn=tf.nn.relu,weights_initializer=b_init)
    b_raw = tf.add(b_raw, 3.00)

    pred_pts = tf.reshape(pts_raw, [pts_raw.shape[0], num_points, 3])
    pred_pts = tf.tanh(pred_pts)
    if cfg.pc_unit_cube:
        pred_pts = pred_pts / 2.0

    out = dict()
    out["xyz"] = pred_pts
    out['z'] = z_raw
    out['b'] = b_raw

    if cfg.pc_rgb:
        if cfg.pc_rgb_deep_decoder:
            inp = outputs_all["conv_features"]
            for _ in range(3):
                inp = slim.fully_connected(inp, cfg.fc_dim,
                                           activation_fn=tf.nn.leaky_relu,
                                           weights_initializer=msra_init)
        else:
            inp = inputs
        rgb_raw = slim.fully_connected(inp, num_points * 3,
                                       activation_fn=None,
                                       weights_initializer=w_init)
        rgb = tf.reshape(rgb_raw, [rgb_raw.shape[0], num_points, 3])
        rgb = tf.sigmoid(rgb)
    else:
        rgb = None
    out["rgb"] = rgb

    return out
