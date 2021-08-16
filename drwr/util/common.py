import tensorflow as tf

def build_command_line_args(pairs, as_string=True):
    if as_string:
        s = ""
    else:
        s = []
    for p in pairs:
        arg = None
        if type(p[1]) == bool:
            if p[1]:
                arg = f"--{p[0]}"
        else:
            arg = f"--{p[0]}={p[1]}"
        if arg:
            if as_string:
                s += arg + " "
            else:
                s.append(arg)
    return s


def parse_lines(filename):
    f = open(filename, "r")
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

	
def en_distances(arr, max_size):	
    batch_size, n_points, _ = arr.shape
    res = tf.tile(arr, [1,n_points,1])
    arr = tf.expand_dims(arr, [2])
    res2 = tf.tile(arr, [1,1,n_points, 1])
    res2 = tf.reshape(res2, [batch_size, n_points * n_points, 2])

    en_distance_liner = tf.reduce_sum(tf.square(tf.subtract(res,res2)),2)
    en_distance = tf.reshape(en_distance_liner, [batch_size, n_points, n_points])
   
    temp = tf.cast(max_size, 'float32') * tf.ones([batch_size, n_points], dtype=tf.float32)
    temp = tf.matrix_diag(temp)

    en_distance_temp = en_distance + temp
    en_distance_min  = tf.reduce_min(en_distance_temp, 1)
    return en_distance_min, en_distance
