# A.Lons
# Jan. 2017
#
# DESCRIPTION
# This is where we run the whole thing, but also where I mess with time-lines and what not to figure out how to best
# run and optimize everything. Thus this is really not for training by testing. For training and evaluation see
# both train_model and eval_model. Those are designed to run independently and trainging and evaluate the results in
# separate python instantiations.


import tensorflow as tf
from tensorflow.python.client import timeline
import time
import input_pipeline as rd
import full_trail_test.network_model_0 as rnm0
import os

# Make sure we set the visable CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#To make sure this is actually working
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# There is a set number of examples in the CIFAR-10
NUM_OF_TRAINING_EXAMPLES = 220


def run_training(train_filenames, batch_size, n_classes, n_epochs=1):

    with tf.Graph().as_default() as g:

        # Get images and labels,
        # Get file names by setting up my readers and queues and pin them to the CPU
        #   see, (https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py)
        #   in method, inputs(), I think I can "Force all teh input processing onto the CPU" by calling the tf.device here
        #   as long as I have "allow_soft_placement=False" in the session settings.
        with tf.device('/cpu:0'):
            images, labels = rd.input_pipline(train_filenames, batch_size=batch_size, numb_pre_threads=4, num_epochs=n_epochs+1, output_type='train')

        with tf.device('/gpu:0'):
            # Create the network graph
            prediction = rnm0.generate_res_network(images, batch_size, n_classes, batch_norm=True, is_training=False,
                                                   on_cpu=False, gpu=0, regulizer=0.05, keep_prob=1.0)
            # Find accuracy
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    pred_arg_max = tf.argmax(prediction, 1)
                    labl_arg_max = tf.argmax(labels, 1)
                    correct_prediction = tf.equal(pred_arg_max, labl_arg_max)
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.device('/cpu:0'):
            with tf.name_scope('global_stepping'):
                global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.device('/cpu:0'):

            # Now prepare all summaries (these following lines will be be based on the tensorflow version!)
            # Tensor Flow r0.12
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('full_trail_test/net_0/smryTest', g)

            # I need to run meta-data which will help for 'time-lines' and if I want to output more info
            #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()

            # Create saver for writing training checkpoints
            saver = tf.train.Saver(name='graph_loader')

            # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the
            # tf.local_variables_init() because I set the num_epoch in the string producer in the other python file.
            # Tensor Flow r0.12

            # Run training for a specific number of training examples.
            counter = 0

            for i in range(n_epochs):

                start_time = time.time()

                # Create a session
                sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))

                init_op = tf.group(tf.local_variables_initializer(),
                                   name='initialize_op')

                # Run the init operation
                sess.run(init_op)

                # Restore the check point
                #chkp = tf.train.get_checkpoint_state('full_trail_test/net_0/chk_pt')
                chkp = tf.train.latest_checkpoint('full_trail_test/net_0/chk_pt')
                #print(sess.run(chkp))
                #saver.restore(sess, chkp.model_checkpoint_path)
                saver.restore(sess, chkp)

                print("\nThe global step value is %d" % sess.run(global_step))

                # Make a coordinator,
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # Train over 90% of examples, save the others for testing
                num_training_batches = int(NUM_OF_TRAINING_EXAMPLES/batch_size)
                acc_list = []


                for j in range(num_training_batches-3):

                    acc, summary = sess.run([accuracy, merged])
                    acc_list.append(acc)
                    summary_writer.add_summary(summary, counter)
                    counter += 1

                coord.request_stop()
                coord.join(threads)
                sess.close()

                # Test over last batch!
                # summary, acc, predi = sess.run([merged, accuracy, key])
                print("Epoch ", i)
                print('  Accuracy:', sum(acc_list) / float(len(acc_list)))
                print('  Epoch run-time:', time.time() - start_time)

                print("\nSleeping for ", 95, " (secs)")
                time.sleep(95)

            # Now I have to clean up
            summary_writer.close()


if __name__ == '__main__':
    filenames = ['data_files/tfr_files/augmented_sets/set_00/test/test_data_-00000-of-00004',
                 'data_files/tfr_files/augmented_sets/set_00/test/test_data_-00001-of-00004',
                 'data_files/tfr_files/augmented_sets/set_00/test/test_data_-00002-of-00004',
                 'data_files/tfr_files/augmented_sets/set_00/test/test_data_-00003-of-00004']
    batch_size = 30
    n_classes = 7
    n_epochs = 600
    run_training(filenames, batch_size, n_classes, n_epochs)