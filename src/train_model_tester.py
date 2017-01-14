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
import network_model_0 as rnm0
import os

# Make sure we set the visable CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#To make sure this is actually working
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# There is a set number of examples in the CIFAR-10
NUM_OF_TRAINING_EXAMPLES = 1000


def run_training(train_filenames, batch_size, n_classes, n_epochs=1):

    with tf.Graph().as_default():

        # Get images and labels,
        # Get file names by setting up my readers and queues and pin them to the CPU
        #   see, (https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py)
        #   in method, inputs(), I think I can "Force all teh input processing onto the CPU" by calling the tf.device here
        #   as long as I have "allow_soft_placement=False" in the session settings.
        with tf.device('/cpu:0'):
            images, labels, _ = rd.input_pipline(train_filenames, batch_size=batch_size, numb_pre_threads=4, num_epochs=n_epochs+1, output_type='train')

        with tf.device('/gpu:0'):
            # Create the network graph
            prediction = rnm0.generate_res_network(images, batch_size, n_classes, batch_norm=True, is_training=True)
            # Now we generate a cost function (so tf knows what this is)
            with tf.name_scope('calc_loss'):
                losses = rnm0.loss(prediction, labels)
            # Now generate optimizer!
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=.0005).minimize(losses, name='adam_optim_min')

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
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=False))
                                                    #device_count={'GPU': 1}))

            # Now prepare all summaries (these following lines will be be based on the tensorflow version!)
            # Tensor Flow r0.12
            merged = tf.summary.merge_all()  # <- these work in newer versions of TF
            summary_writer = tf.summary.FileWriter('summaries/train_summary', sess.graph)

            # I need to run meta-data which will help for 'time-lines' and if I want to output more info
            #   a) To get a time-line to work see running meta-data see http://stackoverflow.com/questions/40190510/tensorflow-
            #   how-to-log-gpu-memory-vram-utilization/40197094
            #   b) To get detailed run information to text file see http://stackoverflow.com/questions/40190510/tensorflow-how-t
            #   o-log-gpu-memory-vram-utilization/40197094
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Create saver for writing training checkpoints
            saver = tf.train.Saver()

            # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the tf.local_variables_init()
            # because I set the num_epoch in the string producer in the other python file.
            # Tensor Flow r0.12
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='initialize_op')
            # In tf 011
            # init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables(), name='initialize_ops')

            # Run the init operation
            sess.run(init_op)

            # Make a coordinator,
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Run training for a specific number of training examples.
            counter = 0

            # Start with a time-line object here to try to get a time line for every epoch
            #with open('timeline.json', 'w') as f:


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


                # Time-line save Run
                start_time = time.time()
                _, summary = sess.run([optimizer, merged], options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(summary, counter)
                meta_run_time = time.time() -start_time
                t1 = timeline.Timeline(run_metadata.step_stats)
                ctf = t1.generate_chrome_trace_format()
                with open('timeline3.json', 'w') as f:
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
                path_to_checkpoint = saver.save(sess, 'summaries/chk_pt/model.ckpt', global_step=i)
                print('  Path to check pont: ', path_to_checkpoint)

            # with open("meta_data_run.txt", "w") as out:
            #    out.write(str(run_metadata))

            # Time-line save
            #t1 = timeline.Timeline(run_metadata.step_stats)
            #ctf = t1.generate_chrome_trace_format()
            #with open('timeline.json', 'w') as f:
            #    f.write(ctf)

            # Now I have to clean up
            summary_writer.close()
            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    filenames = ['cifar-10-batches-bin/data_batch_1.bin',
                 'cifar-10-batches-bin/data_batch_2.bin',
                 'cifar-10-batches-bin/data_batch_3.bin',
                 'cifar-10-batches-bin/data_batch_4.bin',
                 'cifar-10-batches-bin/data_batch_5.bin']
    batch_size = 100
    n_classes = 10
    n_epochs = 1
    run_training(filenames, batch_size, n_classes, n_epochs)