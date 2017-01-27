# A. Lons
# Jan. 2017
#
# DESCRIPTION
# Here we generate a full model of the network we want to use.
#
# MODEL-00 Properties. (18-layer Model)
#
#  Input-Image (128x128)
#           |
#           V
#     {[3X3, 64]
#      [3X3, 64]} x2
#           |
#           V
#  size: [n, 128, 128, 64]
#           |
#           V
#     Max-Pool [2,2]
#           |
#           V
#     {[3X3, 128]
#      [3X3, 128]} x2
#           |
#           V
#  size: [n, 64, 64, 128]
#           |
#           V
#     Max-Pool [2,2]
#           |
#           V
#     {[3X3, 256]
#      [3X3, 256]} x2
#           |
#           V
#  size: [n, 32, 32, 256]
#           |
#           V
#     Max-Pool [2,2]
#           |
#           V
#     {[3X3, 512]
#      [3X3, 512]} x2
#           |
#           V
#  size: [n, 16, 16, 512]
#           |
#           V
#     Max-Pool [2,2]
#           |
#           V
#     {[3X3, 1064]
#      [3X3, 1064]} x2
#           |
#           V
#  size: [n, 8, 8, 1064]
#           |
#           V
#     Max-Pool [2,2]
#           |
#           V
#  size: [n, 4, 4, 1064]
#           |
#           V
#     Avg-Pool [2,2]
#           |
#           V
#  size: [n, 1, 1, 1064]
#           |
#           V
# fc-layer-2128 (hidden layer-ReLu?)
#

import tensorflow as tf
import experimental_results.exper_0.exp_0_network_layers as nl

def generate_res_network(images, batch_size, n_classes, batch_norm=True, is_training=True, on_cpu=True, gpu=0,
                         regulizer=0.0, keep_prob=0.0):
    """
    DESCRIPTION
    Creates teh TF graph/model to use.
    :param images:
    :param batch_size:
    :param n_classes:
    :param batch_norm:
    :param is_training:
    :param on_cpu:
    :param gpu:
    :param regulizer:
    :return:
    """

    # Setup my network layers, where the images are essentially the first input
    net_layers = []
    net_layers.append(images)

    # No down-sampling in first image please
    layer_1_str = 'first_layer'
    layer_1_depth = 32
    layer_1_num_res = 1
    layer_1_down_sample = False

    for i in range(layer_1_num_res):
        var_name = layer_1_str + "_res_%d" % len(net_layers)
        with tf.variable_scope(var_name):
            net_layers.append(nl.res_block(net_layers[-1], layer_1_depth, down_sample=layer_1_down_sample,
                                           batch_norm=batch_norm, is_training=is_training, on_cpu=on_cpu, gpu=gpu,
                                           regulizer=regulizer, keep_prob=keep_prob))

    # Yes down sampling
    # Input at this point should be [n, 128, 128, 32]
    layer_2_str = 'second_layer'
    layer_2_depth = 64
    layer_2_num_res = 1
    layer_2_down_sample = True

    for i in range(layer_2_num_res):
        var_name = layer_2_str + "_res_%d" % len(net_layers)
        with tf.variable_scope(var_name):
            if i == 0:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_2_depth, down_sample=layer_2_down_sample, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))
            else:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_2_depth, down_sample=False, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))

    # Yes down sampling
    # Input at this point should be [n, 64, 64, 64]
    layer_3_str = 'third_layer'
    layer_3_depth = 128
    layer_3_num_res = 1
    layer_3_down_sample = True

    for i in range(layer_3_num_res):
        var_name = layer_3_str + "_res_%d" % len(net_layers)
        with tf.variable_scope(var_name):
            if i == 0:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_3_depth, down_sample=layer_3_down_sample, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))
            else:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_3_depth, down_sample=False, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))

    # Yes down sampling
    # Input at this point should be [n, 32, 32, 128]
    layer_4_str = 'fourth_layer'
    layer_4_depth = 256
    layer_4_num_res = 1
    layer_4_down_sample = True

    for i in range(layer_4_num_res):
        var_name = layer_4_str + "_res_%d" % len(net_layers)
        with tf.variable_scope(var_name):
            if i == 0:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_4_depth, down_sample=layer_4_down_sample, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))
            else:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_4_depth, down_sample=False, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))

    # Yes down sampling
    # Input at this point should be [n, 16, 16, 256]
    layer_5_str = 'fifth_layer'
    layer_5_depth = 512
    layer_5_num_res = 1
    layer_5_down_sample = True

    for i in range(layer_5_num_res):
        var_name = layer_5_str + "_res_%d" % len(net_layers)
        with tf.variable_scope(var_name):
            if i == 0:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_5_depth, down_sample=layer_5_down_sample, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))
            else:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_5_depth, down_sample=False, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))

    # Yes down sampling
    # Input at this point should be [n, 8, 8, 512]
    layer_6_str = 'sixth_layer'
    layer_6_depth = 1012
    layer_6_num_res = 1
    layer_6_down_sample = True

    for i in range(layer_6_num_res):
        var_name = layer_6_str + "_res_%d" % len(net_layers)
        with tf.variable_scope(var_name):
            if i == 0:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_6_depth, down_sample=layer_6_down_sample,
                                 batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))
            else:
                net_layers.append(
                    nl.res_block(net_layers[-1], layer_6_depth, down_sample=False, batch_norm=batch_norm,
                                 is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                 keep_prob=keep_prob))

    # Input at this point should be [n, 4, 4, 1024]
    # apply a max pool operation
    with tf.variable_scope('seventh_layer_avg_pool'):
        net_layers.append(tf.nn.avg_pool(net_layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                         name='max_pool_op'))

    # Input at this point should be [n, 1, 1, 1024]
    # Now add a fully connected layer please!
    with tf.variable_scope('eight_layer_fully_connected'):
        reshaped_input = tf.reshape(net_layers[-1], [batch_size, -1], name='reshape_input')
        # I am getting the shape of the output, simply following what is done in cifar10.inference(images)
        flattened_dim = reshaped_input.get_shape()[1].value
        net_layers.append(nl.gen_output_layer(reshaped_input, [flattened_dim, n_classes], [n_classes]))

    return net_layers[-1]


def loss(prediction, labels):
    """
    DESCRIPTION
    Seperately setup the loss function.
    :param prediction:
    :param labels:
    :return: loss
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(prediction, labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='mean_cross_entropy_loss')

    # We do it this way because this is how it was done in multi-gpu CIFAR-10 example, which allows us to easily add
    # other values to the loss as well!
    tf.add_to_collection("losses", cross_entropy_mean)

    #return cross_entropy_mean
