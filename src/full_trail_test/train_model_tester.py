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
NUM_OF_TRAINING_EXAMPLES = 11000


def run_training(train_filenames, batch_size, n_classes, n_epochs=1):

    with tf.Graph().as_default():

        # Get images and labels,
        # Get file names by setting up my readers and queues and pin them to the CPU
        #   see, (https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py)
        #   in method, inputs(), I think I can "Force all teh input processing onto the CPU" by calling the tf.device here
        #   as long as I have "allow_soft_placement=False" in the session settings.
        with tf.device('/cpu:0'):
            images, labels = rd.input_pipline(train_filenames, batch_size=batch_size, numb_pre_threads=4, num_epochs=n_epochs+1, output_type='train')

        with tf.device('/gpu:0'):
            # Create the network graph
            prediction = rnm0.generate_res_network(images, batch_size, n_classes, batch_norm=True, is_training=True,
                                                   on_cpu=False, gpu=0, regulizer=0.05, keep_prob=.80)
            # Now we generate a cost function (so tf knows what this is)
            with tf.name_scope('calc_loss'):
                #losses = rnm0.loss(prediction, labels)
                _ = rnm0.loss(prediction, labels)
                losses = tf.get_collection('losses')
                total_loss = tf.add_n(losses, name='sum_total_loss')
            # Now generate optimizer!
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=.0005).minimize(total_loss, name='adam_optim_min')
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(total_loss, name='grad_dec_optim')
            # Now with global_steps

        with tf.device('/cpu:0'):
            with tf.name_scope('global_stepping'):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                increment_global_step_op = tf.assign(global_step, global_step+1)

        with tf.device('/gpu:0'):
            # Find accuracy
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    pred_arg_max = tf.argmax(prediction, 1)
                    labl_arg_max = tf.argmax(labels, 1)
                    correct_prediction = tf.equal(pred_arg_max, labl_arg_max)
                    #with tf.device('/cpu:0'):
                    #    tf.summary.histogram('predicted_label_numb', pred_arg_max)
                    #    tf.summary.histogram('actual_label_numb', labl_arg_max)
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.device('/cpu:0'):

            # Create a session, this is done in how-to and cifar-10 example (in the cifar-10 the also have some configs).
            # I also found a resource to help specifiy which GPUs to use and how to label them
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
                                                    #device_count={'GPU': 1}))

            # Now prepare all summaries (these following lines will be be based on the tensorflow version!)
            # Tensor Flow r0.12
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('full_trail_test/net_0/smry', sess.graph)

            # I need to run meta-data which will help for 'time-lines' and if I want to output more info
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Create saver for writing training checkpoints
            saver = tf.train.Saver()

            # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the
            # tf.local_variables_init() because I set the num_epoch in the string producer in the other python file.
            # Tensor Flow r0.12
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='initialize_op')

            # Run the init operation
            sess.run(init_op)

            # Make a coordinator,
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Run training for a specific number of training examples.
            counter = 0

            for i in range(n_epochs):

                # Train over 90% of examples, save the others for testing
                num_training_batches = int(NUM_OF_TRAINING_EXAMPLES/batch_size)
                #num_training_batches = 50

                # Track time for batches
                run_batch_times = []

                for j in range(num_training_batches-3):

                    start_time = time.time()

                    #summary, _ = sess.run([merged, optimizer], options=run_options, run_metadata=run_metadata)
                    #summary_writer.add_summary(summary, counter)

                    #_ = sess.run([optimizer], options=run_options, run_metadata=run_metadata)

                    _ = sess.run([optimizer])

                    counter += 1
                    run_batch_times.append(time.time()-start_time)

                # Run global-stepping
                sess.run(increment_global_step_op)

                # Time-line save Run
                start_time = time.time()
                _, summary = sess.run([optimizer, merged], options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(summary, counter)
                meta_run_time = time.time() - start_time
                t1 = timeline.Timeline(run_metadata.step_stats)
                ctf = t1.generate_chrome_trace_format()
                with open('full_trail_test/net_0/tline/timeline1.json', 'w') as f:
                    f.write(ctf)
                counter += 1


                # Test over last batchs!
                acc = sess.run([accuracy])          #Accuracy
                counter += 1
                prd = sess.run([correct_prediction])#Prediction
                counter += 1

                print(prd)
                print('Epoch ', i)
                print('  Accuracy:', acc)
                avg_run_time = sum(run_batch_times) / float(len(run_batch_times))
                print('  Avg batch run time :', avg_run_time)
                print('  Number of batch runs: ', len(run_batch_times))
                print('  Time to run with meta-data: ', meta_run_time)

                # Now save the graph!
                path_to_checkpoint = saver.save(sess, 'full_trail_test/net_0/chk_pt/model.ckpt', global_step=global_step)
                print('  Path to check pont: ', path_to_checkpoint)

            # Now I have to clean up
            summary_writer.close()
            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    filenames = ['data_files/tfr_files/augmented_sets/set_00/train/train_data_-00000-of-00004',
                 'data_files/tfr_files/augmented_sets/set_00/train/train_data_-00001-of-00004',
                 'data_files/tfr_files/augmented_sets/set_00/train/train_data_-00002-of-00004',
                 'data_files/tfr_files/augmented_sets/set_00/train/train_data_-00003-of-00004']
    batch_size = 30
    n_classes = 7
    n_epochs = 300
    run_training(filenames, batch_size, n_classes, n_epochs)