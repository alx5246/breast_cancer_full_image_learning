# A.Lons
# Jan. 2017
#
# DESCRIPTION
# Take a set of PNG files and transfer into TFRecord type for fast data movement.
#
# We are here looking for .png or .jpeg pictures to be sorted in folders, where each folder name is a specific label.
# For example, if we have a folder named "planes", then all the images within said folder are accordingly pictures of
# planes. A test file is also given that has a list of the classes that we are going to be looking for, one class per
# line.


import tensorflow as tf
import os
import random
import numpy as np
from datetime import datetime
import threading
import sys



def _int64_feature(value):
    """
    (AJL) Taken and modified from tensorflow/models/inception/data/build_image_data.py.
    Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """
    (AJL) Taken and modified from tensorflow/models/inception/data/build_image_data.py. I had to add the bits with the
    .encode, because stings where not encoding automatically!
    Wrapper for inserting bytes features into Example proto.
    """
    if isinstance(value, str):
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """
    (AJL) Taken and modified from tensorflow/models/inception/data/build_image_data.py.
    Build an Example proto for an example.
    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
    Returns:
        Example proto
    """

    colorspace = 'RGB'
    #channels = 3
    channels = 3
    image_format = 'JPEG'
    # Here we are make the features, and turning everyting to a int64 or bytes
    example = tf.train.Example(features=tf.train.Features(feature={
                               'image/height': _int64_feature(height),
                               'image/width': _int64_feature(width),
                               'image/colorspace': _bytes_feature(colorspace),
                               'image/channels': _int64_feature(channels),
                               'image/class/label': _int64_feature(label),
                               'image/class/text': _bytes_feature(text),
                               'image/format': _bytes_feature(image_format),
                               'image/filename': _bytes_feature(os.path.basename(filename)),
                               'image/encoded': _bytes_feature(image_buffer)}))
    return example



class ImageCoder(object):
    """
    (AJL) Taken and modified from tensorflow/models/inception/data/build_image_data.py. One of the importnat methods
    here is the tf.image.decode_jpeg() which will a tf.placeholer(dtype=tf.sting) and make into a tf.unit8 tensor.

    DESCRIPTION
    Helper class that provides TensorFlow image coding utilities. This will simply hold functions and methods!
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        #
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """
    (AJL) Taken and modified from tensorflow/models/inception/data/build_image_data.py. NOTE, the
    tf.gfile.FastFile.read() in the first with statement below actually returns 'bytes'!
    Process a single image file.
    ARGS:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    RETURNS:
        image_buffer: string, JPEG encoding of RGB image. <------------------------Okay a sting
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
        #print(type(image_data))
    # Convert any PNG to JPEG's for consistency (AJL, I may want to dumb this later)
    if '.png' in filename:
        #print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG. We ONLY do this in order to get the size of the image! We will NOT output the decoded image
    # rather we will output the image_data. NOTE, the coder.decode_jpeg retruns an numpy.ndarry type!
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    # Return image as string that would have to be dcoded!
    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, texts, labels, num_shards):
    """
    (AJL) Taken and modified from tensorflow/models/inception/data/build_image_data.py.
    DESCRIPTION
        Processes and saves list of images as TFRecord in 1 thread.
    ARGS:
        coder: instance of ImageCoder class to provide TensorFlow image coding utils/methods.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to
            analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads). For instance, if num_shards = 128, and the
    # num_threads = 2, then the first thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    # (AJL) this whole shard thing is... strange. But it seems to be that we are going to divide up the entire dataset
    # into a set of smaller data pieces. If there are 1000 examples, and 100 shardes, then we are going to end up with
    # 10 examples per shard.

    # Find the indicies, 'shard_ranges', of the files in particular we are going to here parse. That is we want to find
    # the range of example indicies that will go into each shard.
    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
    # The number of files to parse in this thread
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    # Iterate over each shard (sub-set of examples) that we do here in this thread
    counter = 0
    for s in range(num_shards_per_batch):

        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s                     # The shard number, ie 2 of 128
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)     # The output file name
        output_file = os.path.join(OUTPUT_DIRECTOY, output_filename)        # The actual output file
        writer = tf.python_io.TFRecordWriter(output_file)

        #Iterate over all teh files that will go into this shard
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]   # The filename
            label = labels[i]         # the label (and int i think)
            text = texts[i]           # the text (the class name like 'dog')

            # Run the process-image method above, which will also need the 'coder' object
            image_buffer, height, width = _process_image(filename, coder)

            # Take the processed things and now make an example, write example with TFRexordWriter object
            example = _convert_to_example(filename, image_buffer, label, text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 100:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' % (datetime.now(), thread_index, counter, num_files_in_thread))
            sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %(datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %(datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()



def _process_image_files(name, filenames, texts, labels, num_shards):
    """
    (AJL) Taken and modified from tensorflow/models/inception/data/build_image_data.py.

    Process and save list of images as TFRecord of Example protos.
    Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]], the following line will make a 1D numpy array
    # with the ranges of the examples that will go into each thread.
    spacing = np.linspace(0, len(filenames), NUM_THREADS + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (NUM_THREADS, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.request_stop()
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %(datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    """
    (AJL) Taken and modified from tensorflow/models/inception/data/build_image_data.py.

    Build a list of all images files and labels in the data set.
    Args:
        data_dir: string, path to the root directory of images.
            Assumes that the image data set resides in JPEG files located in
            the following directory structure.
                data_dir/dog/another-image.JPEG
                data_dir/dog/my-image.jpg
            where 'dog' is the label associated with these images.
        labels_file: string, path to the labels file.
            The list of valid labels are held in this file. Assumes that the file
            contains entries as such:
                dog
                cat
                flower
            where each line corresponds to a label. We map each label contained in
            the file to an integer starting with the integer 0 corresponding to the
            label contained in the first line.
    Returns:
        filenames: list of strings; each string is a path to an image file.
        texts: list of strings; each string is the class, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth.
    """

    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []

    # Leave label index 0 empty as a background class.
    label_index = 0

    # Construct the list of JPEG files and labels. # Find all the sub-files (png) in each label directoy.
    for text in unique_labels:
        png_file_path = '%s/%s/*' % (data_dir, text)   # so %s I think means insert string
        matching_files = tf.gfile.Glob(png_file_path)
        #print(len(matching_files)) # Trying to see what we find
        #print(matching_files[0])   # Trying to see what we find
        # This .extend (list method) seems almost the same as .append but by an iterable instead
        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (label_index, len(labels)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    #shuffled_index = range(len(filenames))
    shuffled_index = np.arange(0, len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d PNG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
    return filenames, texts, labels


if __name__ == '__main__':

    # This is tell how many threads to use and directoreis to work in. Some of this code is hardcoded.
    NUM_THREADS = 1
    OUTPUT_DIRECTOY = 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/test'
    #OUTPUT_DIRECTOY = 'data_files/tfr_files/tfr_set_00/set_16bit_128x128/training'

    # Give the data directories
    data_dir = 'data_files/png_files/png_set_00/set_16bit_128x128/Test'
    #data_dir = 'data_files/png_files/png_set_00/set_16bit_128x128/Training'

    # Give label text files, these are the folder labels we will be looking for!
    labels_file = 'data_files/png_files/png_set_00/set_16bit_128x128/classes.txt'

    filenames, texts, labels = _find_image_files(data_dir, labels_file)
    print(filenames)
    name ='test_data_'
    #name = 'train_data_'
    print(set(labels))
    print(set(texts))
    num_shards = 10
    #num_shards = 1
    _process_image_files(name, filenames, texts, labels, num_shards)


