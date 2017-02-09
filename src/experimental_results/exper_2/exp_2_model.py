# A. Lons
# Jan. 2017
#
# DESCRIPTION
# Here we generate a full model of the network we want to use.


import tensorflow as tf
import experimental_results.exper_2.exp_2_network_layers as nl

def generate_res_network(images, batch_size, n_classes, batch_norm=True, is_training=True, on_cpu=False, gpu=0,
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

    # 46 layers
    #list_of_layers = [[False, 2, 512, 512, 64],
    #                  [True, 3, 256, 256, 128],
    #                  [True, 4, 128, 128, 256],
    #                  [True, 5, 64, 64, 512],
    #                  [True, 6, 32, 32, 1024],
    #                  [True, 6, 16, 16, 1024],
    #                  [True, 6, 8, 8, 1024],
    #                  [True, 6, 4, 4, 1024]]

    #list_of_layers = [[False, 1, 128, 128, 32],
    #                  [False, 1, 128, 128, 64],
    #                  [False, 1, 128, 128, 64],
    #                  [True, 2, 64, 64, 128],
    #                  [True, 2, 32, 32, 128],
    #                  [True, 2, 16, 16, 256],
    #                  [True, 1, 8, 8, 512],
    #                  [True, 1, 4, 4, 1024]]

    list_of_layers = [[False, 1, 128, 128, 32],
                      [False, 1, 128, 128, 64],
                      [False, 1, 128, 128, 64],
                      [True, 1, 64, 64, 128],
                      [True, 1, 32, 32, 128],
                      [True, 1, 16, 16, 256],
                      [True, 1, 8, 8, 512],
                      [True, 1, 4, 4, 1024]]

    net_layers = []
    net_layers.append(images)
    layer_str_base = 'layer_'
    counter = 0

    # Now fill in the layers!
    for sub_list in list_of_layers:

        # Fill in over the number of sub-layers
        for i in range(sub_list[1]):
            print(sub_list)

            var_name = layer_str_base + "%d" % counter
            counter += 1

            with tf.variable_scope(var_name):
                if i==0 and sub_list[0]:
                    net_layers.append(
                                      nl.res_block(net_layers[-1], sub_list[4], down_sample=True,
                                      batch_norm=batch_norm, is_training=is_training, on_cpu=on_cpu, gpu=gpu,
                                      regulizer=regulizer, keep_prob=keep_prob))
                else:
                    net_layers.append(
                                      nl.res_block(net_layers[-1], sub_list[4], down_sample=False, batch_norm=batch_norm,
                                      is_training=is_training, on_cpu=on_cpu, gpu=gpu, regulizer=regulizer,
                                      keep_prob=keep_prob))

    # Input at this point should be [n, 4, 4, 1024]
    var_name = layer_str_base + "%d" % counter
    counter += 1
    if on_cpu:
        device_str = '/gpu:0'
        print("\nFUCK... last layer is on the cpu")
        with tf.device(device_str):
            with tf.variable_scope(var_name):
                net_layers.append(tf.nn.avg_pool(net_layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                                 name='max_pool_op'))
                net_layers.append(tf.nn.max_pool(net_layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                                 name='max_pool_op'))
    else:
        device_str = '/gpu:%d' % gpu
        with tf.device(device_str):
            with tf.variable_scope(var_name):
                net_layers.append(tf.nn.avg_pool(net_layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                                 name='max_pool_op'))
                net_layers.append(tf.nn.max_pool(net_layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                                 name='max_pool_op'))

    # Input at this point should be [n, 1, 1, 1024]
    # Now add a fully connected layer please!
    var_name = layer_str_base + "%d" % counter
    counter += 1
    if on_cpu:
        device_str = '/gpu:0'
        print("\nFUCK... last layer is on the cpu")
        with tf.device(device_str):
            with tf.variable_scope(var_name):
                reshaped_input = tf.reshape(net_layers[-1], [batch_size, -1], name='reshape_input')
                # I am getting the shape of the output, simply following what is done in cifar10.inference(images)
                flattened_dim = reshaped_input.get_shape()[1].value

                net_layers.append(nl.gen_output_layer(reshaped_input, [flattened_dim, n_classes], [n_classes],
                                                      on_cpu=on_cpu, gpu=gpu, regulizer=regulizer))
    else:
        device_str = '/gpu:%d' % gpu
        with tf.device(device_str):
            with tf.variable_scope(var_name):
                reshaped_input = tf.reshape(net_layers[-1], [batch_size, -1], name='reshape_input')
                # I am getting the shape of the output, simply following what is done in cifar10.inference(images)
                #flattened_dim = reshaped_input.get_shape()[1].value
                flattened_dim = 1024
                net_layers.append(nl.gen_output_layer(reshaped_input, [flattened_dim, n_classes], [n_classes],
                                                      on_cpu=on_cpu, gpu=gpu, regulizer=regulizer))

    return net_layers[-1]


def loss(prediction, labels):
    """
    DESCRIPTION
    Seperately setup the loss function.
    :param prediction:
    :param labels:
    :return: loss
    """
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(prediction, labels, name='cross_entropy')
    # Change for tf r1.0
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=prediction, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='mean_cross_entropy_loss')

    # We do it this way because this is how it was done in multi-gpu CIFAR-10 example, which allows us to easily add
    # other values to the loss as well!
    #tf.add_to_collection("losses", cross_entropy_mean)

    return cross_entropy_mean