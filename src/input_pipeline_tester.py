# A. Lons
# Deceber 2016
#
# This is here to make sure I am doing all the conversion work correctly and making sure my images come out as
# expected.


import tensorflow as tf
import matplotlib.pyplot as plt

# There are some variables I need to set!
# Not yet sure how to set this yet, so this number right now is a bit arbitrary
NUMB_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30
# In order for TF loading part to work, TF needs to know the size of the images, thus I set that here.
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

def read_and_decode_TFR(filename_queue):
    """
    This is a modified version of read_and_decode from ...fully_connected_reader.py. NOTE: here I am assigning image
    size on the fly, which is not good for tensorflow to build graph, thus I have to give the size of the image here.
    NOTE: I could not get this to work with just .decode_raw, I had to use the decode_jpeg.
    NOTE: I did not unpack most of the information from teh TFrecond file, just the image, and the label
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
                                                                     'image/encoded': tf.FixedLenFeature([], tf.string)} )

    # The image/encodes was originally stored as string, we have to decode this into a the jpeg, I could not get the
    # decode_raw to work. I am using tf.image.decode_jpeg instead, which decodes a jpeg into a uint8 tensor. This means
    # that all the image values range from [0,255]
    # image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    depth = 3
    image = tf.reshape(image, [height, width, depth])
    image.set_shape([height, width, depth])

    # Convert iamges to gray-scales. NOTE: in order to see image with pyplot.imshow() this must also be
    # transformed into float values with the last dimension (which shoudl be 1 because we reduced to gray-scale)
    # trimmed off as well.
    image = tf.image.rgb_to_grayscale(image, name='convert_to_grayscale')
    image = tf.cast(image, tf.float32) * (1. / 255.) - 0.5 # Convert to better input range.
    # I need to get rid of the last dimension
    image = tf.reshape(image, [height, width])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['image/class/label'], tf.int64)

    return image, label


def input_pipline(filenames, batch_size, numb_pre_threads):
    """
    DESCRIPTION
        In accordance with your typical pipeline that I have denoted, we have a seperate method that sets up the
        data.
    :param filenames: the list of filenames, where each file has examples (TFRecords type for the exmaples here)
    :param batch_size: how many files in each bath
    :param numb_pre_threads: the number of threads to use to read and decode
    :return:
    """

    # Generate the file-name queue from given list of filenames. IMPORTANT, this function can read through strings
    # indefinitely, thus you WANT to give a "num_epochs" parameter, when you reach the limit, the "OutOfRange" error
    # will be thrown.
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    # Read the image using method defined above
    image, label = read_and_decode_TFR(filename_queue)

    # Use tf.train.shuffle_batch to shuffle up batches. "min_after_dequeue" defines how big a buffer we will randomly
    # sample from -- bigger means better shuffling but slower start up and more memory used. "capacity" must be larger
    # than "min_after_dequeue" and the amount larger determines the maximm we will prefetch. The recommendation:
    # for capacity is min_after_dequeue + (num_threads + saftey factor) * batch_size
    # From cifar10_input.input(), setup min numb of exampels in teh queue
    min_fraction_of_examples_in_queue = .6
    min_queue_examples = int(NUMB_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    min_after_dequeue = min_queue_examples
    capacity = min_queue_examples + 3 * batch_size
    images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=numb_pre_threads,
                                                 capacity=capacity, min_after_dequeue=min_after_dequeue)

    return images, tf.reshape(label_batch, [batch_size])


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
    images, labels = input_pipline(filenames, batch_size=10, numb_pre_threads=1)

    # This is done in one how-to example and in cafir-10 example.
    init_op = init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create a session, this is done in how-to and cifar-10 example (in the cifar-10 the also have some configs)
    sess = tf.Session()

    # Run the init, this is done in how-to and cifar-10
    sess.run(init_op)

    # Make a coordinator,
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(5):

        a, b = sess.run([images, labels])
        print("\n")
        print(a.shape)
        print(type(a))
        print(sess.run(tf.reduce_max(a[0, :, :])))
        print(sess.run(tf.reduce_min(a[0, :, :])))
        print(b)
        #plt.imshow(a[0, :, :,0])
        plt.figure()
        plt.imshow(a[0], cmap='gray')
        plt.figure()
        plt.imshow(a[1], cmap='gray')
        plt.show()

    # Now I have to clean up
    coord.request_stop()
    coord.join(threads)
    sess.close()