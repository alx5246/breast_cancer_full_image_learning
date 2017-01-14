# A.Lons
# Jan. 2017
#
# DESCRIPTION
# This will be the input-pipeline I use to feed data into the train and evaluation methods.

import tensorflow as tf
import matplotlib.pyplot as plt


def read_and_decode_TFR(filename_queue, image_width, image_height, image_channels=3, gray_scale=False):
    """
    DESCRIPTION
    Read and decode TFRecord format data.
    NOTES
    There are some things hard-coded in here. For example the size of the "one-hot" option is set to 2 because we have
    two possible outputs.
    :param filename_queue:
    :return: image, label
    """

    # I use the reader specific to "TFRecordds" format
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={'image/height': tf.FixedLenFeature([], tf.int64),
                                                                     'image/width': tf.FixedLenFeature([], tf.int64),
                                                                     'image/colorspace': tf.FixedLenFeature([], tf.string),
                                                                     'image/channels': tf.FixedLenFeature([], tf.int64),
                                                                     'image/class/label': tf.FixedLenFeature([], tf.int64),
                                                                     'image/class/text': tf.FixedLenFeature([], tf.string),
                                                                     'image/format': tf.FixedLenFeature([], tf.string),
                                                                     'image/filename': tf.FixedLenFeature([], tf.string),
                                                                     'image/encoded': tf.FixedLenFeature([], tf.string)}, name='parse_single_example')

    with tf.name_scope('format_image'):

        # The image/encodes was originally stored as string, we have to decode this into a the jpeg, I could not get the
        # decode_raw to work. I am using tf.image.decode_jpeg instead, which decodes a jpeg into a uint8 tensor. This
        # means that all the image values range from [0,255]
        # image = tf.decode_raw(features['image/encoded'], tf.uint8)
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3, name='decode_jpeg')
        height = image_height
        width = image_width
        depth = image_channels
        image = tf.reshape(image, [height, width, depth], name='reshape')
        image.set_shape([height, width, depth])

        # Convert images to correct input scaling.
        if gray_scale:
            image = tf.image.rgb_to_grayscale(image, name='convert_to_grayscale')
            with tf.name_scope('normalize_im'):
                image = tf.cast(image, tf.float32, name='cast_image') * (1. / 255.) - 0.5  # Convert to better input range.
                # IF I want to print this image out later with imshow, I need to dump the last dimension!
                #image = tf.reshape(image, [height, width]) # I need to get rid of the last dimension
        else:
            with tf.name_scope('normalize_im'):
                image = tf.cast(image, tf.float32, name='cast_image') * (1. / 255.) - 0.5  # Convert to better input range.

    with tf.name_scope('format_label'):
        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['image/class/label'], tf.int32, name='cast_label')
        print(label)
        # We make this scalar label a one-hot type!
        label = tf.one_hot(label, depth=2, name='label_scaler_to_vec')
        # Flatten label to 1D, rather than 2D with 1-row
        label = tf.reshape(label, [-1], name='label_flatten')

    return image, label


def input_pipline(file_names, batch_size, numb_pre_threads, num_epochs = 1, output_type = 'train', im_w=128, im_h=128,
                  im_d=3, gray_scale=False):
    """
    DESCRIPTION
        In accordance with your typical pipeline, we have a seperate method that sets up the data.
    :param file_names: list of file names that have the data
    :param batch_size: the number of examples per batch
    :param numb_pre_threads:
    :return: A tuple (images, labels, keys) where:
    """

    # This will no work if we pin to the GPU!
    with tf.device('/cpu:0'):

        with tf.name_scope('input_pipeline'):

            # Generate the file-name queue from given list of filenames. IMPORTANT, this function can read through
            # strings indefinitely, thus you WANT to give a "num_epochs" parameter, when you reach the limit, the
            # "OutOfRange" error will be thrown.
            filename_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, name='file_name_queue',
                                                            capacity=100)

            # Read the image using method defined above, this will actually take the queue and one its files, and read
            # some data
            #read_input = read_binary_image(filename_queue)
            images, labels = read_and_decode_TFR(filename_queue, im_w, im_w, image_channels=im_d, gray_scale=gray_scale)

            # Use tf.train.shuffle_batch to shuffle up batches. "min_after_dequeue" defines how big a buffer we will
            # randomly sample from -- bigger means better shuffling but slower start up and more memory used. "capacity"
            # must be larger than "min_after_dequeue" and the amount larger determines the maximm we will prefetch. The
            # recommendation: for capacity is min_after_dequeue + (num_threads + saftey factor) * batch_size
            # From cifar10_input.input(), setup min numb of examples in the queue
            min_fraction_of_examples_in_queue = .6
            min_queue_examples = int(batch_size * min_fraction_of_examples_in_queue)
            min_after_dequeue = min_queue_examples
            capacity = min_queue_examples + 10 * batch_size
            print("\nInput-pipe,")
            print("  capacity .....", capacity)
            print("  batch_size ...", batch_size)
            print("  min_after.....", min_after_dequeue)

            if output_type == 'train':
                # If I want to shuffle input!
                images_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                                   batch_size=batch_size,
                                                                   num_threads=numb_pre_threads,
                                                                   capacity=capacity,
                                                                   min_after_dequeue=min_after_dequeue,
                                                                   name='train_shuffle_batch')
            # If I do not wany to shuffle input!
            else:
                images_batch, label_batch = tf.train.batch([images, labels],
                                                           batch_size=batch_size,
                                                           num_threads=numb_pre_threads,
                                                           capacity=capacity,
                                                           name='batch_generator')


        return images_batch, label_batch


if __name__ == '__main__':

    # Here we will run the test! This will test our abilities to set everything correctly! In this case I will test with
    # the data I have converted using convertPng.py.

    # Get file names
    filenames = ['data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00001-of-00010',
                 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00002-of-00010',
                 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00003-of-00010',
                 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00004-of-00010',
                 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00005-of-00010',
                 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00006-of-00010',
                 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00007-of-00010',
                 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00008-of-00010',
                 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test/test_data_-00009-of-00010']

    #
    images, labels = input_pipline(filenames, batch_size=10, numb_pre_threads=1, num_epochs=1, output_type='train',
                                   im_w=128, im_h=128, im_d=3, gray_scale=True)

    # This is done in one how-to example and in cafir-10 example.
    init_op = init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create a session, this is done in how-to and cifar-10 example (in the cifar-10 the also have some configs)
    sess = tf.Session()

    # Run the init, this is done in how-to and cifar-10
    sess.run(init_op)

    # Make a coordinator,
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(2):

        a, b = sess.run([images, labels])
        print("\n")
        print(a.shape)
        print(type(a))
        print(sess.run(tf.reduce_max(a[0, :, :])))
        print(sess.run(tf.reduce_min(a[0, :, :])))
        print(b)
        #plt.imshow(a[0, :, :,0])
        plt.figure()
        plt.imshow(a[0, :, :, 0], cmap='gray')
        plt.figure()
        plt.imshow(a[1, :, :, 0], cmap='gray')
        plt.show()

    # Now I have to clean up
    coord.request_stop()
    coord.join(threads)
    sess.close()