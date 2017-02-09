# A. lons
# Jan. 2017
#
# Here are all the layers that can be put together to form a res-net.

import tensorflow as tf



def _variable_on_cpu(name, shape, initializer, trainable=True, regulizer=0.1):
    """
    DESCRIPTION
    Taken from "../cifar10.py", where even though the ops may run on the gpu, it seems they put the variables on the
    CPU.
    :param name: name of the variable
    :param shape: list of ints
    :param initializer: initializer for the tf.Variable
    :param trainable: boolean, True means that this variable can be trained or altered by TF's optimizers
    :param regulizer: float over [0,1], this is the coefficient we use to penalize weight growth
    :return: Variable tensor
    """
    with tf.device('/gpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        # Add regularization to weights
        #if regulizer > .0001:
        #    # We add small value for stability (also have the absolute value)
        #    weight_reg = tf.mul(tf.nn.l2_loss(tf.add(tf.abs(var), .001)), regulizer, name='weight_reg')
        #    tf.add_to_collection('losses', weight_reg)
    return var


def _variable_on_gpu(name, shape, initializer, trainable=True, regulizer=0.1, gpu=0):
    """
    DESCRIPTION
    I have found that striktly the variables onto the gpu is a bit faster, but it might not be in all circumstances
    especially if I am dividing training across GPUs, though
    :param name: name of the variable
    :param shape: list of ints
    :param initializer: initializer for the tf.Variable
    :param trainable: boolean, True means that this variable can be trained or altered by TF's optimizers
    :param regulizer: float over [0,1], this is the coefficient we use to penalize weight growth
    :return: Variable object (ie, biases)
    """
    gpu_str = '/gpu:%d' % gpu
    with tf.device(gpu_str):
    #with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        # Add regularization to weights
        if regulizer > .0001:
            # We add small value for stability (also have the absolute value)
            weight_reg = tf.multiply(tf.nn.l2_loss(tf.add(tf.abs(var), .001)), regulizer, name='weight_reg')
            tf.add_to_collection('losses', weight_reg)
    return var


def batch_normalization_wrapper(inputs, ten_shape, is_training, decay=.999, decay_minus = .001, epsilon=.0000000001, on_cpu=True, gpu=0):
    """
    DESCRIPTION
    This function is meant to take the batch norm, and switch how this is used from training to evaluation. During
    evaluation we need to use the average values found over training. We assume we are normalizing over the first
    dimension of the inputs!!!!!!
    This is my version of that in http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    Also see https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412 to see how to handle convolution types
    :param inputs: Tensors before going through activation, so simply weights*prior-outputs
    :param ten_shape: integer, depth of input map
    :param is_training: boolean, very important, must be set to False if doing an evaluation
    :param decay: The is for exponential moving average which we use to keep track of the gloabl values of the mean
           and variance
    :param epsilon: small number to divide by, if batch-norm values end up being zero, so we do not deivide by zero
    :param on_cpu: boolean, do we put this operation of the CPU or GPU
    :param gpu: which gpu are we on if more than one is available
    :return: batch-normed tensor, should be same size as input tensor
    """
    if on_cpu:

        device_str = '/cpu:0'

        # The variables that will be used during during training to hold mean and var or a particular input batch.
        # These are used only during training epochs.
        bn_scale = _variable_on_cpu('bn_scaling', ten_shape,
                                    initializer=tf.constant_initializer(value=1.0, dtype=tf.float32), regulizer=0.0)
        bn_beta = _variable_on_cpu('bn_beta', ten_shape,
                                   initializer=tf.constant_initializer(value=0.0, dtype=tf.float32), regulizer=0.0)

        # The variables that get updated during learning, and are actually used however in testing. So these are
        # used and updated differently depending on if used suring evaluation or training.
        pop_bn_mean = _variable_on_cpu('pop_bn_mean', ten_shape,
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32), regulizer=0.0,
                                       trainable=False)
        pop_bn_var = _variable_on_cpu('pop_bn_var', ten_shape,
                                      initializer=tf.constant_initializer(value=1.0, dtype=tf.float32), regulizer=0.0,
                                      trainable=False)

    else:

        device_str = '/gpu:%d' % gpu

        # The variables that will be used during during training to hold mean and var or a particular input batch.
        # These are used only during training epochs.
        bn_scale = _variable_on_gpu('bn_scaling', ten_shape,
                                    initializer=tf.constant_initializer(value=1.0, dtype=tf.float32), regulizer=0.0,
                                    gpu=gpu)
        bn_beta = _variable_on_gpu('bn_beta', ten_shape,
                                   initializer=tf.constant_initializer(value=0.0, dtype=tf.float32), regulizer=0.0,
                                   gpu=gpu)

        # The variables that get updated during learning, and are actually used however in testing. So these are
        # used and updated differently depending on if used suring evaluation or training.
        pop_bn_mean = _variable_on_gpu('pop_bn_mean', ten_shape,
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32),
                                       trainable=False, regulizer=0.0, gpu=gpu)
        pop_bn_var = _variable_on_gpu('pop_bn_var', ten_shape,
                                      initializer=tf.constant_initializer(value=1.0, dtype=tf.float32), trainable=False,
                                      regulizer=0.0, gpu=gpu)

    # Add summaries (all of these will be on the CPU by default)
    #variable_summaries(pop_bn_mean, 'pop_bn_mean')
    #variable_summaries(pop_bn_var, 'pop_bn_var')

    decay_tf = tf.constant(decay, dtype=tf.float32)
    decay_minus_tf = tf.constant(decay_minus, dtype=tf.float32)

    # Set the deivce
    with tf.device(device_str):
    #with tf.device('/gpu:0'):

        if is_training:

            with tf.name_scope('batch_norm_training'):

                # We normalize over depth. Thus if we have a output tensor like [100, 32, 32, 64], we want to
                # normalize over the depth of, for example 64, so we hand the tf.nn.moments a list like [0, 1, 2]
                # to say which dimensions we want to operate over. To calculate this list, we have this simple loop.
                calc_moments_over_which_dimensions = []
                count = 0
                for i in range(len(inputs.get_shape())-1):
                    calc_moments_over_which_dimensions.append(count)
                    count += 1
                b_mean, b_var = tf.nn.moments(inputs, calc_moments_over_which_dimensions)

                # Track with an exponential moving average. Because I am naming this "train_pop_bn_mean" I have a
                # seperate op that can be run in training, and thus I can still use the variable "pop_bn_mean"
                # later in evaluation.
                train_pop_bn_mean = tf.assign(pop_bn_mean, pop_bn_mean * decay_tf + b_mean * decay_minus_tf,
                                              name='poulation_mean_calc')
                train_pop_bn_var = tf.assign(pop_bn_var, pop_bn_var * decay_tf + b_var * decay_minus_tf,
                                             name='poulation_var_calc')

                # Run batch norm (the built in version)
                with tf.control_dependencies([train_pop_bn_mean, train_pop_bn_var]):
                    return tf.nn.batch_normalization(inputs, b_mean, b_var, bn_beta, bn_scale, epsilon,
                                                     name='batch_normalization_training')
        else:
            with tf.name_scope('batch_norm_evaluation'):
                return tf.nn.batch_normalization(inputs, pop_bn_mean, pop_bn_var, bn_beta, bn_scale, epsilon,
                                                 name='batch_normalization_testing')


def gen_2dconv(input_tensor, conv_shape, strides, bias_shape, keep_prob=.85, batch_norm=True, is_training=True,
               on_cpu=True, gpu=0, regulizer=0.0):
    """
    DESCRIPTION
    Creates a 2D-convolution operation, where within we will be creating weights, biases, etc. Note we can select this
    to have a batch-norm here, the activation function is not here but rather is outside to make creating Res-Nets
    more like the original paper
    :param input_tensor: input tensor from previous op or simply the input to a graph.
    :param conv_shape: list, [n, m, x, y] where n and m are the kernal sizes, and x is the number of inputs, and y is
           the number of outputs
    :param strides: list [n, m, x, y] see tf.nn.conv2d to see what strides does.
    :param bias_shape: 1D list, should be equal to the number of outputs
    :param keep_prob: float over [0,1] the probability we keep an output and do not drop it.
    :param batch_norm: boolean, True means we apply a batch_norm before every ReLU
    :param is_training: boolean, needs to be set to true if training, and false is evaluating, this is for setting up
           the batch_norm correctly.
    :param on_cpu: boolean, tells us if this op will be run on the cpu or the gpu
    :param gpu: int, which gpu to dump the operation on
    :param regulizer: float over [0,1] how much to penalize each variable
    :return: tensor output of the convolution layer (ReLU(BN(W*A + Bias))) where BN is batch-norm if called for.
    """

    if on_cpu:
        device_str = '/cpu:0'
    else:
        device_str = '/gpu:%d' % gpu

    if on_cpu:
        kernel = _variable_on_cpu("weights", conv_shape, initializer=tf.random_normal_initializer(),
                                  regulizer=regulizer)
        # Biases do NOT have regulization https://plus.google.com/+IanGoodfellow/posts/QUaCJfvDpni
        biases = _variable_on_cpu("biases", bias_shape, initializer=tf.random_normal_initializer(),
                                  regulizer=0.0)
    else:
        kernel = _variable_on_gpu("weights", conv_shape, initializer=tf.random_normal_initializer(), gpu=gpu,
                                  regulizer=regulizer)
        # Biases do NOT have regulization https://plus.google.com/+IanGoodfellow/posts/QUaCJfvDpni
        biases = _variable_on_gpu("biases", bias_shape, initializer=tf.random_normal_initializer(), gpu=gpu,
                                  regulizer=0.0)

    #variable_summaries(kernel, 'weights')
    #variable_summaries(biases, 'biases')

    with tf.device(device_str):

        conv_op = tf.nn.conv2d(input_tensor, kernel, strides=strides, padding='SAME', name='conv2d_op')

        if keep_prob <.99999:

            if is_training:
                drop_out = tf.nn.dropout(conv_op, keep_prob=keep_prob, name='drop_out')
            else:
                drop_out = tf.nn.dropout(conv_op, keep_prob=1.00, name='drop_out')

            # Apply (or not apply) batch normalization with trainable parameters
            if batch_norm:
                with tf.variable_scope('batch_norm'):
                    pre_activ = batch_normalization_wrapper(tf.nn.bias_add(drop_out, biases, name='add_biases_op'),
                                                            bias_shape, is_training, decay=.999, decay_minus=.001, epsilon=.0000000001,
                                                            on_cpu=on_cpu, gpu=gpu)
            else:
                pre_activ = tf.nn.bias_add(drop_out, biases, name='add_biases_op')

        else:

            # Apply (or not apply) batch normalization with trainable parameters
            if batch_norm:
                with tf.variable_scope('batch_norm'):
                    pre_activ = batch_normalization_wrapper(tf.nn.bias_add(conv_op, biases, name='add_biases_op'),
                                                            bias_shape, is_training, decay=.999, decay_minus=.001, epsilon=.0000000001,
                                                            on_cpu=on_cpu, gpu=gpu)
            else:
                pre_activ = tf.nn.bias_add(conv_op, biases, name='add_biases_op')

        return pre_activ


def res_block(input_tensor, output_depth, down_sample=True, batch_norm=True, is_training=True, on_cpu=True, gpu=0,
              regulizer=0.0, keep_prob=.85):
    """
    DESCRIPTION
        A typical res-net block
    :param input_tensor:
    :param output_depth:
    :param batch_norm: boolean, do we apply batch norm
    :param is_training: boolean, changes how batch norm is applied
    :param down_sample:
    :param on_cpu: boolean, if operation and variables go on CPU
    :param gpu: int, which GPU to use IF we are not running on the CPU
    :param keep_prob: float over [0, 1], the drop out factor
    :return: tensor output from the residual block
    """

    if on_cpu:
        device_str = '/cpu:0'
    else:
        device_str = '/gpu:%d' % gpu

    with tf.device(device_str), tf.name_scope("res_block"):

        # If there is a down-sampling, meaning a max-pooling step, we do this first! The max pooling here is teh
        # standard operation that cuts down the image in half.
        if down_sample:
            with tf.variable_scope("max_pooling"):
                input_tensor = tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                              name='max_pool_op')

        # Now find the input depth, and the output depths, this will be helpful and
        input_depth = input_tensor.get_shape().as_list()[3]

        # Apply the first convolution
        with tf.variable_scope("conv_1"):
            conv_1 = gen_2dconv(input_tensor, [3, 3, input_depth, output_depth], [1, 1, 1, 1], [output_depth],
                                batch_norm=batch_norm, is_training=is_training, on_cpu=on_cpu, gpu=gpu,
                                regulizer=regulizer, keep_prob=keep_prob)

        # Apply the ReLU
        with tf.name_scope("relu_1"):
            relu_1 = tf.nn.relu(conv_1, name='relu_1')

        # Apply the second convolution
        with tf.variable_scope("conv_2"):
            conv_2 = gen_2dconv(relu_1, [3, 3, output_depth, output_depth], [1, 1, 1, 1], [output_depth],
                                batch_norm=batch_norm, is_training=is_training, on_cpu=on_cpu, gpu=gpu,
                                regulizer=regulizer, keep_prob=keep_prob)

        # No apply the addition operator to combine input and output!

        # First we have to pad the if necessary
        if input_depth != output_depth:
            with tf.name_scope('pad_input_tensor'):
                input_tensor = tf.pad(input_tensor, [[0, 0], [0, 0], [0, 0], [0, output_depth-input_depth]],
                                      name='pad_input')

        # Add together
        res_pre_activ= tf.add(conv_2, input_tensor, name='add_block_to_skip')

        # Apply Relu output
        res_output = tf.nn.relu(res_pre_activ, name='relu_output')

    return res_output


def gen_hidden_layer(input_tensor, weight_shape, bias_shape, batch_norm=True, is_training=True, on_cpu=True, gpu=0,
                     regulizer=0.0):
    """
    DESCRIPTION
    Generate a hidden layer (non convolution and fully connected).
    NOTES
    Assume the input is already in the correct [nBatchs, mFlattened] size.
    :param input_tensor: input tensor, ie [batch_size, length of flattened input] <- if the former layer was a conv layer flattened!
    :param weight_shape: the shape of the weights, ie, [length of flattened input, number of hidden neurons]
    :param bias_shape: the shape of the bias, ie [32] if we have 32 hidden neuons.
    :param batch_norm: boolean, True means we apply a batch_norm before every ReLU
    :param is_training: boolean, needs to be set to true if training, and false is evaluating, this is for setting up
           the batch_norm correctly.
    :param on_cpu: boolean
    :param gpu: int
    :return:
    """

    if on_cpu:
        device_str = '/cpu:0'
        weights = _variable_on_cpu("weights", weight_shape, initializer=tf.random_normal_initializer(),
                                   regulizer=regulizer)
        # Biases do NOT have regulization https://plus.google.com/+IanGoodfellow/posts/QUaCJfvDpni
        biases = _variable_on_cpu("biases", bias_shape, initializer=tf.random_normal_initializer(),
                                   regulizer=0.0)
    else:
        device_str = '/gpu:%d' % gpu
        weights = _variable_on_gpu("weights", weight_shape, initializer=tf.random_normal_initializer(), gpu=gpu,
                                   regulizer=regulizer)
        # Biases do NOT have regulization https://plus.google.com/+IanGoodfellow/posts/QUaCJfvDpni
        biases = _variable_on_gpu("biases", bias_shape, initializer=tf.random_normal_initializer(), gpu=gpu,
                                   regulizer=0.0)

    #variable_summaries(weights, 'weights')
    #variable_summaries(biases, 'biases')

    with tf.device(device_str):

        add_bias_op = tf.nn.bias_add(tf.matmul(input_tensor, weights, name='mat_mul'), biases, name='add_biases_op')

        if batch_norm:
            with tf.name_scope('batch_norm'):
                pre_activ = batch_normalization_wrapper(add_bias_op, bias_shape, is_training, decay=.999,
                                                        epsilon=.0000000001, on_cpu=on_cpu, gpu=gpu)
                output = tf.nn.relu(pre_activ, name='relu_op')
        else:
            output = tf.nn.relu(add_bias_op, name='relu_op')

    return output


def gen_output_layer(input, weight_shape, bias_shape, on_cpu=True, gpu=0, regulizer=0.0):
    """
    DESCRIPTION
    We are assuming all the reshaping has been done outside already!
    :param input: input tensor, ie [batch_size, length of flattened input] <- if the former layer was a conv layer
    flattened!
    :param kernel_shape: the shape of the weights, ie, [length of flattened input, number of hidden neurons]
    :param bias_shape: the size of the bias, ie [64] if there are 64 outputs
    :return:
    """
    if on_cpu:
        device_str = '/cpu:0'
        weights = _variable_on_cpu("weights", weight_shape, initializer=tf.random_normal_initializer(),
                                   regulizer=regulizer)
        # Biases do NOT have regulization https://plus.google.com/+IanGoodfellow/posts/QUaCJfvDpni
        biases = _variable_on_cpu("biases", bias_shape, initializer=tf.random_normal_initializer(),
                                   regulizer=0.0)
    else:
        device_str = '/gpu:%d' % gpu
        weights = _variable_on_gpu("weights", weight_shape, initializer=tf.random_normal_initializer(), gpu=gpu,
                                   regulizer=regulizer)
        # Biases do NOT have regulization https://plus.google.com/+IanGoodfellow/posts/QUaCJfvDpni
        biases = _variable_on_gpu("biases", bias_shape, initializer=tf.random_normal_initializer(), gpu=gpu,
                                   regulizer=0.0)

    #variable_summaries(weights, 'weights')
    #variable_summaries(biases, 'biases')

    #with tf.device(device_str):
    with tf.device('/gpu:0'):
        output = tf.nn.bias_add(tf.matmul(input, weights, name='mat_mul'), biases, name='add_biases_op')

    return output