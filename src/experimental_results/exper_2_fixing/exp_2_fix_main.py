# A.Lons
# Jan 2017
#
# DESCRIPTION
# I am going to make something to run things iteratively

# Imports to handle multi-processing portions
import random
from time import sleep

# Tensorflow import
import tensorflow as tf
import experimental_results.exper_2_fixing.exp_2_fix_model as model
import input_pipeline as ip

from tensorflow.python.client import timeline

# To track running time
import time
from datetime import datetime


def training_session(data_description_txt_file, save_path,
                     train_filenames, n_examples_per_epoch, n_classes, batch_size, n_epochs, batch_norm, regulizer,
                     keep_prob, learning_rate, decay_rate, decay_steps):
    """
    DESCRIPTION
    Where network training occurs. This method encapsulates the generation of a TF network, and then training for a
    specififed number of iterations. Every so often we write summaries our. After each training epoch we save the
    graph as a check-pt.
    :param data_description_txt_file: a txt file where the data used is described.
    :param finished_first_epoch_event: a multiprocessing event, that is set after one epoch, letting the evalution
           portion to start running.
    :param finished_training_event: a multiprocessing event, that is set after all training epochs are complete, this
           is to let the evaluation/testing portion to stop running.
    :param save_path: string, the path to the main folder where we will store the results of training and evaluation.
    :param train_filenames: list of strings, the training data
    :param n_examples_per_epoch: int, the number of exampels we train with for each epoch, guides the input pipeline
    :param n_classes: how many classes the output of the model has
    :param batch_size: int, how many examples in each batch
    :param n_epochs: int, how many epochs to train the graph on
    :param batch_norm: boolean, set True if we want to batch-norm after convolutions
    :param regulizer: float over [0, 1], weight of the penatly
    :param keep_prob: float over [0, 1], the probability at which activations are dropped
    :param learning_rate: float, for optimizer
    :return: None, saves results.
    """

    # We need to import the os in order to handle files and setting up CUDA correctly
    import os
    # Make sure we set the visable CUDA devices
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # To make sure this is actually working
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # PATH NOTES: we are within the exper_0/ folder, thus everything we save must be relative to that path.
    # Thus if we want to go up one folder we would have to use ../ or two folders ../../

    # Deal with folder creation. For each experiment we want a separate folder. In said folder we will have folders for
    # saving training-check-points, best-evaluation-check-points, and summary-writers.

    list_of_dirs = os.listdir(save_path)
    experiment_numb = len(list_of_dirs) + 1

    # Now we making (or finding if we are restarting) new directories, first by making the path names
    exp_dir = os.path.join(save_path, '_exper_2_' + str(experiment_numb).zfill(4))
    train_chk_pt_dir = os.path.join(exp_dir, 'train_chk_pt')
    test_chk_pt_dir = os.path.join(exp_dir, 'test_chk_pt')
    train_smry_dir = os.path.join(exp_dir, 'train_smry_dir')
    test_smry_dir = os.path.join(exp_dir, 'test_smry_dir')


    os.makedirs(exp_dir)
    os.makedirs(train_chk_pt_dir)
    os.makedirs(test_chk_pt_dir)
    os.makedirs(train_smry_dir)
    os.makedirs(test_smry_dir)

    # Now make the actual path to the 'check-point', because loading and saving need to point differently I guess. When
    # I load I only point to the directory, here I have to point to the actual file-name. DUMB!
    train_chk_pt_dir = os.path.join(train_chk_pt_dir, 'chk_pt')
    test_chk_pt_dir = os.path.join(test_chk_pt_dir, 'chk_pt')

    # Include a log-file that writes out a report so we can figure out why we are crashing!
    run_log_text_path = os.path.join(exp_dir, 'training_log_.txt')

    # Include a compact mini-fail log
    fail_log_test_path = os.path.join(exp_dir, 'training_mini_fail_log.txt')

    # Now we also want to include a text file (not a running log) that just tells us about what is being trained, what
    # it is being trained with, and how long it takes!
    exp_info_text_path = os.path.join(exp_dir, 'training_info.txt')

    # If we are NOT restarting, then we need to start off by writing to the experiment-info file.
    with open(exp_info_text_path, "w") as info_text_file:
        # Lets fill in some important information in here!
        info_text_file.write('A.Lons & S.Picard')
        info_text_file.write('\n\nDeep Learning - Breast Cancer - Benign vs. Malignant Identification')
        info_text_file.write('\n\nExperimental Trial: ' + str(experiment_numb))
        info_text_file.write("\n\n########################################################################################################################")
        info_text_file.write('\n\n                                           EXPERIMENTAL VARIABLES AND PARAMS')
        info_text_file.write('\n\nExamples per Epoch: ' + str(n_examples_per_epoch))
        info_text_file.write('\nBatch size: ' + str(batch_size))
        info_text_file.write('\nNumber of target Epochs: ' + str(n_epochs))
        info_text_file.write('\nBatch normalization: ' + str(batch_norm))
        info_text_file.write('\nRegularization Beta: ' + str(regulizer))
        info_text_file.write('\nDrop out val: ' + str(keep_prob))
        info_text_file.write('\nLearning rate: ' + str(learning_rate))
        info_text_file.write('\nLearning rate decay-steps: ' + str(decay_steps))
        info_text_file.write('\nLearning rate decay-rate: ' + str(decay_rate))
        info_text_file.write('\n\nThe filenames of the data used here during training are ... ')
        for file_name in train_filenames:
            info_text_file.write("\n" + file_name)
        info_text_file.write('\n\nThe description associated with the data is ... ')
        with open(data_description_txt_file, "r") as set_desc:
            # Grab all lines of the description file and put into an array
            lines_of_text = set_desc.readlines()
            # Now put all these lines into our file
            for text_line in lines_of_text:
                info_text_file.write(text_line)


    with tf.Graph().as_default():

        #with open(run_log_text_path, "a") as log_file:
        #    log_file.write("\nGenerate input-pipeline")

        with tf.device('/cpu:0'):
            # Get images and labels,
            images, labels = ip.input_pipline(train_filenames, batch_size=batch_size, numb_pre_threads=2,
                                              num_epochs=n_epochs+1, output_type='train')

        #with open(run_log_text_path, "a") as log_file:
        #    log_file.write("\nGenerate Network")

        with tf.device('/gpu:0'):
        #with tf.device('/cpu:0'):
            # Create the network graph model
            prediction = model.generate_res_network(images, batch_size, n_classes, batch_norm=batch_norm,
                                                    is_training=True, on_cpu=False, gpu=0, regulizer=regulizer,
                                                    keep_prob=keep_prob)

            with tf.name_scope('calc_loss'):
                # Generate losses, where we add losses to a collection, so we can add other losses to minimize as well.
                #_ = model.loss(prediction, labels)
                total_loss = model.loss(prediction, labels)
                # Find losses (error loss)
                #losses = tf.get_collection('losses')
                # Find total loss (including the fact we want to minimize weights)
                #total_loss = tf.add_n(losses, name='sum_total_loss')

        #with open(run_log_text_path, "a") as log_file:
        #    log_file.write("\nGenerate optimizer")
        #
        #with open(run_log_text_path, "a") as log_file:
        #    log_file.write("\nGenerate global-stepper")

        #with tf.device('/cpu:0'):
        with tf.device('/gpu:0'):
            with tf.name_scope('global_stepping'):
                #global_step = tf.Variable(0, name='global_step', trainable=False)
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                              trainable=False, dtype=tf.float32)
        with tf.device('/gpu:0'):
            increment_global_step_op = tf.assign(global_step, global_step + 1)

        with tf.device('/gpu:0'):
        #with tf.device('/gpu:0'):
            with tf.name_scope('some_optimizer_parts'):
                decay_learn_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                              decay_rate=decay_rate, decay_steps=decay_steps)

                #opt_adam = tf.train.MomentumOptimizer(learning_rate=decay_learn_rate, momentum=0.9)
                #opt_adam = tf.train.GradientDescentOptimizer(learning_rate=decay_learn_rate)


        with tf.device('/gpu:0'):
            with tf.name_scope('optimizer'):
                # Now generate optimizer, firstly we will make the learning rate decay!
                #decay_learn_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                #                                              decay_rate=decay_rate, decay_steps=decay_steps)
                # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss,
                #                                                                         name='adam_optimizer')
                # Now make the optimizer with gradient clipping! This should help as the training goes onwards and
                # which in turn makes the gradients larger apparently.
                #opt_adam = tf.train.AdadeltaOptimizer(learning_rate=decay_learn_rate , name='adam_optim')
                #opt_adam = tf.train.AdamOptimizer(learning_rate=decay_learn_rate, name='adam_optim')
                opt_adam = tf.train.MomentumOptimizer(learning_rate=decay_learn_rate, momentum=0.9)
                #opt_adam = tf.train.GradientDescentOptimizer(learning_rate=decay_learn_rate)
                gvs = opt_adam.compute_gradients(total_loss, colocate_gradients_with_ops=False)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                optimizer = opt_adam.apply_gradients(capped_gvs, global_step=global_step)
                #optimizer = opt_adam.apply_gradients(gvs) # Capping does not seem to effect run time!
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate=decay_learn_rate).minimize(total_loss, name='adam_optim_min')

        with tf.device('/cpu:0'):

            #with open(run_log_text_path, "a") as log_file:
            #    log_file.write("\nGenerate tf.Session")

            # Create a session, but make sure we donot keep logs and also do not allow TF to place operations where it
            # wants to, but rather where we want to.
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list="0"), log_device_placement=False, allow_soft_placement=True))

            #with open(run_log_text_path, "a") as log_file:
            #    log_file.write("\nGenerate summary operations")

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(train_smry_dir, sess.graph)

            # I need to run meta-data which will help for 'time-lines' and if I want to output more info
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            #with open(run_log_text_path, "a") as log_file:
            #    log_file.write("\nGenerate saver")

            # Create saver for writing training checkpoints, these checkpoints are necessary for allowing an
            # evaluation function to open the graph and evaluate it's performance.
            #saver = tf.train.Saver()

            #with open(run_log_text_path, "a") as log_file:
            #    log_file.write("\nGenerate initialize variables")

            # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the
            # tf.local_variables_init() because I set the num_epoch in the string producer in the other python file.
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                               name='initialize_op')
            sess.run(init_op)

            #with open(run_log_text_path, "a") as log_file:
            #    log_file.write("\nGenerate coordinator")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Run training for a specific number of training examples.
            num_training_batches = int(n_examples_per_epoch / batch_size)

            # Lets keep track of starting run in the experiment-info file
            with open(exp_info_text_path, "a") as info_text_file:
                # Lets fill in some important information in here!
                info_text_file.write("\n\n##############################################################################"
                                     "##########################################")
                info_text_file.write('\n\n Beginning training at ' + str(datetime.now()))

            with open(run_log_text_path, "a") as log_file:
                log_file.write("\nGenerate loop over training iterations")

            # Keep track of total time
            total_start_time = time.time()

            # Iterate over all training epochs, change from a for-loop in order to handle restarts
            #for epoch_iteration in range(n_epochs):
            while sess.run(global_step) < n_epochs:

                #with open(run_log_text_path, "a") as log_file:
                #    log_file.write("\nEpoch %d: (a) starting training iteration" % sess.run(global_step))

                # Just for testing, we are going to force this thread to die at epoch 2 for multi-processing testing
                #if sess.run(global_step) == 2 and not is_restart:
                #    some_new_value = 2.3/0.


                try:

                    with open(run_log_text_path, "a") as log_file:
                        log_file.write("\nEpoch %d: (b) setting up variables" % sess.run(global_step))

                    run_batch_times = []
                    sess.run(increment_global_step_op) # Putting this now at the end

                    print('\nTRAINING Epoch %d Started...\n' % sess.run(global_step))

                    with open(run_log_text_path, "a") as log_file:
                        log_file.write("\nEpoch %d: (c) starting for-loop" % sess.run(global_step))

                    start_time_full_run = time.time()

                    for j in range(num_training_batches):

                        # start_time = time.time()
                        # _ = sess.run(prediction)
                        # p_run_time = time.time() - start_time
                        # print("\nPrediction run-time = " + str(p_run_time))
                        #
                        # start_time = time.time()
                        # _ = sess.run(total_loss)
                        # l_run_time = time.time() - start_time
                        # print("\nLoss run-time = " + str(l_run_time))
                        #
                        # start_time = time.time()
                        # _ = sess.run(gvs)
                        # g_run_time = time.time() - start_time
                        # print("\nGradient run-time = " + str(g_run_time))

                        #print('\nTRAINING Epoch :', epoch_iteration, ', batch:', j)
                        start_time = time.time()
                        _ = sess.run(optimizer)
                        print('\nNO-Meta-Run-Time = ' + str(time.time() - start_time))

                        start_time = time.time()
                        _ = sess.run(optimizer, options=run_options, run_metadata=run_metadata)
                        #print(run_metadata.step_stats)
                        print('\nMeta-Run-Time = ' + str(time.time()-start_time))

                        summary_str = sess.run(summary_op)
                        step_number = (sess.run(global_step) + 1) * num_training_batches + (j + 1)
                        summary_writer.add_summary(summary_str, step_number)

                        with tf.device('/cpu:0'):
                            run_batch_times.append(time.time()-start_time)
                            t1 = timeline.Timeline(run_metadata.step_stats)
                            ctf = t1.generate_chrome_trace_format(show_dataflow=True, show_memory=True)
                            theFileName = os.path.join(exp_dir, 'timeline.json')
                            with open(theFileName, 'w') as f:
                                f.write(ctf)

                        if j % 100 == 0:
                            with open(run_log_text_path, "a") as log_file:
                                log_file.write("\nEpoch %d: (c-a) saving summary mid-training" % sess.run(global_step))
                            summary_str = sess.run(summary_op)
                            step_number = (sess.run(global_step)+1)*num_training_batches + (j+1)
                            summary_writer.add_summary(summary_str, step_number)

                    run_time = time.time() - start_time_full_run

                    with open(run_log_text_path, "a") as log_file:
                        log_file.write("\nEpoch %d: (d) finished training now output to screen " % sess.run(global_step))

                    print('\nTRAINING Epoch %d Complete,  ' % sess.run(global_step))
                    avg_run_time = run_time / float(num_training_batches)
                    print('  Date: ', datetime.now())
                    print('  Total run time: ', time.time()-total_start_time)
                    print('  Avg batch run time :', avg_run_time)
                    print('  Number of batch runs: ', len(run_batch_times), "\n")

                    with open(run_log_text_path, "a") as log_file:
                        log_file.write("\nEpoch %d: (e) saving to chk-pt  " % sess.run(global_step))

                    # Now save the graph. There are some important parameters we want to keep around here. Important
                    # is "global_step" which will be the epoch count for us. Also we can set the make points kept.
                    # try:
                    #     saver.save(sess, train_chk_pt_dir, global_step=global_step, write_meta_graph=False)
                    # except:
                    #
                    #     print('\nTRAINING, problem with saving check-point, not saving the graph')
                    #
                    #     with open(run_log_text_path, "a") as log_file:
                    #         log_file.write('\n... (e-a) ERROR ERROR problem with saving check-point (unable to '
                    #                        'save graph)')
                    #
                    #     with open(fail_log_test_path, "a") as fail_log:
                    #         fail_log.write('\nERROR ERROR, problem with saving ch-pt (unable to save graph) at '
                    #                        'global-step: ' + str(sess.run(global_step)) + " at time: " +
                    #                        str(datetime.now()))

                except:

                    # Lets keep track of starting run in file
                    with open(run_log_text_path, "a") as log_file:
                        log_file.write('\n\nERROR ERROR, oops, something went wrong in try-except!!\n')

                    with open(fail_log_test_path, "a") as fail_log:
                        fail_log.write('\nERROR ERROR, oops, something went wrong in try-except at global-step: '
                                       + str(sess.run(global_step) + " at time " + str(datetime.now())))

                    print("\nTRAINING: SOMETHING IN THE TESTING WENT WRONG!!, WILL TRY AGAIN, B"
                          "EGIN SLEEP FOR A BIT\n")
                    time.sleep(2.0)

                # Increment the global-step value
                sess.run(increment_global_step_op)

            with open(run_log_text_path, "a") as log_file:
                log_file.write("\nSaving results of training to text file (ie total training time recorded)")

            # Lets keep track of starting run in file
            with open(exp_info_text_path, "a") as info_text_file:
                # Lets fill in some important information in here!
                info_text_file.write('\n Ending training at ' + str(datetime.now()))

            with open(run_log_text_path, "a") as log_file:
                log_file.write("\nCleaning up and closing threads")

            # Clean up all the tensor flow things!
            summary_writer.close()
            coord.request_stop()
            coord.join(threads)
            sess.close()
            time.sleep(1.5)

            print('\nTRAINING Ended and closed out\n')






if __name__ == "__main__":

    # We need to import the os in order to handle files and setting up CUDA correctly
    import os
    # Make sure we set the visable CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # To make sure this is actually working
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    train_filenames_list = []
    test_filenames_list = []
    data_description_list = []
    train_numbexamples_list = []
    test_numbexamples_list = []

    # Grab the augmented data set ("soft" distortions of rotations and translations with caltech images as well)
    train_filenames_list.append(['../../data_files/tfr_files/augmented_sets/set_01/train/train_data_-00000-of-00004',
                                 '../../data_files/tfr_files/augmented_sets/set_01/train/train_data_-00001-of-00004',
                                 '../../data_files/tfr_files/augmented_sets/set_01/train/train_data_-00002-of-00004',
                                 '../../data_files/tfr_files/augmented_sets/set_01/train/train_data_-00003-of-00004'])
    test_filenames_list.append(['../../data_files/tfr_files/augmented_sets/set_01/test/test_data_-00000-of-00004',
                                '../../data_files/tfr_files/augmented_sets/set_01/test/test_data_-00001-of-00004',
                                '../../data_files/tfr_files/augmented_sets/set_01/test/test_data_-00002-of-00004',
                                '../../data_files/tfr_files/augmented_sets/set_01/test/test_data_-00003-of-00004'])
    data_description_list.append('../../data_files/tfr_files/augmented_sets/set_01/set_info.txt')
    #train_numbexamples_list.append(9070)
    train_numbexamples_list.append(1000)
    test_numbexamples_list.append(220)

    save_path = 'saved_results_a'
    n_classes = 7
    batch_size = 100
    n_epochs = 300
    batch_norm = True
    #regulizer_list = [0.0, 0.001, .01, .1, 1.0]
    regulizer_list = [0.001, .000, .00, .0]
    keep_prob_list = [1.0] # Drop out does not seem to be effective here
    decay_rate_list = [.9]
    decay_steps_list = [25, 50, 100, 200]
    learning_rate = .1 # This is now the initial rate

    number_train_sessions = len(regulizer_list) * len(keep_prob_list) * len(train_filenames_list)
    count = 1
    global_start_time = time.time()

    sub_count = 0

    # Now we also want to include a text file along with all of this, so lets do that now!
    run_info_text_path = os.path.join(save_path, 'training_testing_.txt')
    with open(run_info_text_path, "a") as info_text_file:

        for keep_prob in keep_prob_list:

            for regulizer in regulizer_list:

                for decay_rate in decay_rate_list:

                    for decay_steps in decay_steps_list:

                        sub_count += 1
                        if sub_count>0:

                            for i in range(len(train_filenames_list)):

                                info_text_file.write("\nRunning Training-Testing on data-set " + str(i) + ", with reg-val: " + str(
                                    regulizer) + ", with keep_prob:" + str(keep_prob) + " with decay-steps: "
                                    + str(decay_steps) + ", with decay-rate: " + str(decay_rate))

                                print(
                                    "\n\n\n########################################################################################")
                                print("Beginning training session %d of %d" % (count, number_train_sessions))
                                print("Total run time is " + str(time.time() - global_start_time))
                                print("########################################################################################")

                                # Pull out the correct set of training/testing data and the description txt files
                                train_filenames = train_filenames_list[i]
                                test_filenames = test_filenames_list[i]
                                data_description = data_description_list[i]  # The txt files that describe the data


                                # Pull out the number of training examples
                                n_examples_per_epoch = train_numbexamples_list[i]
                                training_session(data_description, save_path, train_filenames, n_examples_per_epoch,
                                                 n_classes, batch_size, n_epochs, batch_norm,
                                                 regulizer, keep_prob, learning_rate,
                                                 decay_rate, decay_steps)
