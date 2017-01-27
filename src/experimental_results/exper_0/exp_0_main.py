# A.Lons
# Jan 2017
#
# DESCRIPTION
# I am going to make something to run things iteratively

# Imports to handle multi-processing portions
import random
from time import sleep
import multiprocessing as mp

# Tensorflow import
import tensorflow as tf
import experimental_results.exper_0.exp_0_model_0 as model
import input_pipeline as ip

# To track running time
import time
from datetime import datetime




def training_session(is_restart, data_description_txt_file, finished_first_epoch_event, finished_training_event,
                     performance_drop_event, save_path,
                     train_filenames, n_examples_per_epoch, n_classes, batch_size, n_epochs, batch_norm, regulizer,
                     keep_prob, learning_rate):
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # To make sure this is actually working
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())

    # PATH NOTES: we are within the exper_0/ folder, thus everything we save must be relative to that path.
    # Thus if we want to go up one folder we would have to use ../ or two folders ../../

    # Deal with folder creation. For each experiment we want a separate folder. In said folder we will have folders for
    # saving training-check-points, best-evaluation-check-points, and summary-writers.

    if not is_restart:
        # We need to make a new directory for our experiment, thus we need to see what directories already exist! To do this
        # we are going to find out how many directories already exist, and then make a new one!
        list_of_dirs = os.listdir(save_path)
        experiment_numb = len(list_of_dirs) + 1
    else:
        # If we are restarting this up again because it turned originally into a Zombie-Process then we need to find
        # the older folder we were writing too.
        list_of_dirs = os.listdir(save_path)
        experiment_numb = len(list_of_dirs)

    # Now we making (or finding if we are restarting) new directories, first by making the path names
    exp_dir = os.path.join(save_path, '_exper_0_' + str(experiment_numb).zfill(4))
    train_chk_pt_dir = os.path.join(exp_dir, 'train_chk_pt')
    test_chk_pt_dir = os.path.join(exp_dir, 'test_chk_pt')
    train_smry_dir = os.path.join(exp_dir, 'train_smry_dir')
    test_smry_dir = os.path.join(exp_dir, 'test_smry_dir')

    # Now we actually make the directories where the check-points and the summaries will be stored!
    if not is_restart:
        os.makedirs(exp_dir)
        os.makedirs(train_chk_pt_dir)
        os.makedirs(test_chk_pt_dir)
        os.makedirs(train_smry_dir)
        os.makedirs(test_smry_dir)

    # Now make the actual path to the 'check-point', because loading and saving need to point differently I guess. When
    # I load I only point to the directory, here I have to point to the actual file-name. DUMB!
    if is_restart:
        train_chk_pt_dir_act = train_chk_pt_dir # the actual directory not a file name
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
    if not is_restart:
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
            info_text_file.write('\nLearning rate: ' + str(learning_rate))
            info_text_file.write('\nDrop out val: ' + str(keep_prob))
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
    else:
        with open(fail_log_test_path, "a") as fail_log:
            fail_log.write('\nForced to RESTART TRAINING at ' + str(datetime.now()) + " ... the thread was shut down for some reason")

    with tf.Graph().as_default():

        if is_restart:
            with open(run_log_text_path, "a") as log_file:
                log_file.write("\n\nRESTARTING TRAINING ... the thread was shut down for some reason, now we have to "
                               "rebuild!!")

        with open(run_log_text_path, "a") as log_file:
            log_file.write("\nGenerate input-pipeline")

        with tf.device('/cpu:0'):
            # Get images and labels,
            images, labels = ip.input_pipline(train_filenames, batch_size=batch_size, numb_pre_threads=4,
                                              num_epochs=n_epochs+1, output_type='train')

        with open(run_log_text_path, "a") as log_file:
            log_file.write("\nGenerate Network")

        with tf.device('/gpu:0'):
            # Create the network graph model
            prediction = model.generate_res_network(images, batch_size, n_classes, batch_norm=batch_norm,
                                                    is_training=True, on_cpu=False, gpu=0, regulizer=regulizer,
                                                    keep_prob=keep_prob)

            with tf.name_scope('calc_loss'):
                # Generate losses, where we add losses to a collection, so we can add other losses to minimize as well.
                _ = model.loss(prediction, labels)
                # Find losses (error loss)
                losses = tf.get_collection('losses')
                # Find total loss (including the fact we want to minimize weights)
                total_loss = tf.add_n(losses, name='sum_total_loss')

        with open(run_log_text_path, "a") as log_file:
            log_file.write("\nGenerate optimizer")

        with tf.device('/gpu:0'):
            with tf.name_scope('optimizer'):
                # Now generate optimizer!
                #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss,
                #                                                                         name='adam_optimizer')
                # Now make the optimizer with gradient clipping! This should help as the training goes onwards and
                # which in turn makes the gradients larger apparently.
                opt_adam = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name='adam_optim')
                gvs = opt_adam.compute_gradients(total_loss)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                optimizer = opt_adam.apply_gradients(capped_gvs)

        with open(run_log_text_path, "a") as log_file:
            log_file.write("\nGenerate global-stepper")

        with tf.device('/cpu:0'):
            with tf.name_scope('global_stepping'):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                increment_global_step_op = tf.assign(global_step, global_step + 1)

        with tf.device('/cpu:0'):

            with open(run_log_text_path, "a") as log_file:
                log_file.write("\nGenerate tf.Session")

            # Create a session, but make sure we donot keep logs and also do not allow TF to place operations where it
            # wants to, but rather where we want to.
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))

            with open(run_log_text_path, "a") as log_file:
                log_file.write("\nGenerate summary operations")

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(train_smry_dir, sess.graph)

            with open(run_log_text_path, "a") as log_file:
                log_file.write("\nGenerate saver")

            # Create saver for writing training checkpoints, these checkpoints are necessary for allowing an
            # evaluation function to open the graph and evaluate it's performance.
            saver = tf.train.Saver()

            with open(run_log_text_path, "a") as log_file:
                log_file.write("\nGenerate initialize variables")

            # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the
            # tf.local_variables_init() because I set the num_epoch in the string producer in the other python file.
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                               name='initialize_op')
            sess.run(init_op)

            # Now we need to reload the graph if necessary
            if is_restart:
                # Fill training-log
                with open(run_log_text_path, "a") as log_file:
                    log_file.write("\n...Restart: load in last graph")
                chkp = tf.train.latest_checkpoint(train_chk_pt_dir_act)
                saver.restore(sess, chkp)
                with open(run_log_text_path, "a") as log_file:
                    log_file.write("\n...Restart: the last chk-pt has taken at global-step %d" % sess.run(global_step))
                # Increment global-step to get to correct value
                sess.run(increment_global_step_op)
                # Fill in fail-log
                with open(fail_log_test_path, "a") as fail_log:
                    fail_log.write('\n...Restart training at global-step %d' % sess.run(global_step))

            with open(run_log_text_path, "a") as log_file:
                log_file.write("\nGenerate coordinator")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Run training for a specific number of training examples.
            num_training_batches = int(n_examples_per_epoch / batch_size)

            # Lets keep track of starting run in the experiment-info file
            if not is_restart:
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

                with open(run_log_text_path, "a") as log_file:
                    log_file.write("\nEpoch %d: (a) starting training iteration" % sess.run(global_step))

                # Just for testing, we are going to force this thread to die at epoch 2 for multi-processing testing
                #if sess.run(global_step) == 2 and not is_restart:
                #    some_new_value = 2.3/0.

                # Multiprocessing: make sure we are not over-training by checking 'performance_drop_event' which is
                # set if the testing (separate thread) is not seeing any improvement over a number of iterations.
                if not performance_drop_event.is_set():

                    try:

                        # Some multiprocessing things, we want to set this event to true after we finish the first
                        # training epoch so we can start calling the evaluation
                        if sess.run(global_step) == 1:
                            finished_first_epoch_event.set()

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write("\nEpoch %d: (b) setting up variables" % sess.run(global_step))

                        run_batch_times = []
                        #sess.run(increment_global_step_op) # Putting this now at the end

                        print('\nTRAINING Epoch %d Started...\n' % sess.run(global_step))

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write("\nEpoch %d: (c) starting for-loop" % sess.run(global_step))

                        for j in range(num_training_batches):

                            #print('\nTRAINING Epoch :', epoch_iteration, ', batch:', j)
                            start_time = time.time()
                            _ = sess.run([optimizer])
                            run_batch_times.append(time.time()-start_time)

                            if j % 100 == 0:
                                with open(run_log_text_path, "a") as log_file:
                                    log_file.write("\nEpoch %d: (c-a) saving summary mid-training" % sess.run(global_step))
                                summary_str = sess.run(summary_op)
                                step_number = (sess.run(global_step)+1)*num_training_batches + (j+1)
                                summary_writer.add_summary(summary_str, step_number)

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write("\nEpoch %d: (d) finished training now output to screen " % sess.run(global_step))

                        print('\nTRAINING Epoch %d Complete,  ' % sess.run(global_step))
                        avg_run_time = sum(run_batch_times) / float(len(run_batch_times))
                        print('  Date: ', datetime.now())
                        print('  Total run time: ', time.time()-total_start_time)
                        print('  Avg batch run time :', avg_run_time)
                        print('  Number of batch runs: ', len(run_batch_times), "\n")

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write("\nEpoch %d: (e) saving to chk-pt  " % sess.run(global_step))

                        # Now save the graph. There are some important parameters we want to keep around here. Important
                        # is "global_step" which will be the epoch count for us. Also we can set the make points kept.
                        try:
                            saver.save(sess, train_chk_pt_dir, global_step=global_step, write_meta_graph=False)
                        except:

                            print('\nTRAINING, problem with saving check-point, not saving the graph')

                            with open(run_log_text_path, "a") as log_file:
                                log_file.write('\n... (e-a) ERROR ERROR problem with saving check-point (unable to '
                                               'save graph)')

                            with open(fail_log_test_path, "a") as fail_log:
                                fail_log.write('\nERROR ERROR, problem with saving ch-pt (unable to save graph) at '
                                               'global-step: ' + str(sess.run(global_step)) + " at time: " +
                                               str(datetime.now()))

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
                else:

                    print('\nTRAINING Ending early at epoch %d due to performance drops in testing\n' %
                          sess.run(global_step))

                    # Set the other event for good measure
                    finished_training_event.set()

                    with open(run_log_text_path, "a") as log_file:
                        log_file.write("\n\nPerformance drop event occurred, leaving training loop")
                    break

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

    # Now do the multi-processing thing... which is .set() the event, which will hopefully tell
    finished_training_event.set()


def evaluation_session(is_restart, data_description_txt_file, finished_training_event, performance_drop_event,
                       performance_drop_count_limit, save_path, test_filenames, n_examples_per_epoch, n_classes,
                       batch_size, n_epochs, batch_norm, regulizer, keep_prob, learning_rate):

    # We need to import the os in order to handle files and setting up CUDA correctly
    import os

    # Make sure we set the visable CUDA devices, if we only want CPU, then set to empty.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # To make sure this is actually working, we want to print out some things to tell us which GPU device this thing
    # is seeing.
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())

    # The first step is going to be to find the directory where we are saving the training data, it should be the
    # directory with the largest number! We will find this one by looping over the directories found.
    list_of_dirs = os.listdir(save_path)
    val = -1
    ind = 0
    for zx, dir_name in enumerate(list_of_dirs):
        last_chars = dir_name[-4:]
        if last_chars.isdigit():
            if val < int(last_chars):
                ind = zx
                val = int(last_chars)

    # Now we know the path name! Now we need to find the path's to the sub-directories
    exp_dir = os.path.join(save_path, list_of_dirs[ind])      # The primary directory we want to be in
    train_chk_pt_dir = os.path.join(exp_dir, 'train_chk_pt')  # Sub-directory, storing training chk-pts
    test_chk_pt_dir = os.path.join(exp_dir, 'test_chk_pt')    # Sub-directory, storing testing chk-pts (best networks)
    #train_smry_dir = os.path.join(exp_dir, 'train_smry_dir')  # Sub-directory, storing training summaries
    test_smry_dir = os.path.join(exp_dir, 'test_smry_dir')    # Sub-directory, storing testing summaries

    # Wait are these right? Only sort of. In order to load a chk_pt we have to point to the directory, in order to save
    # we have to point to the actual file! STUPID FUCKING TENSORFLOW!
    # train_chk_pt_dir = os.path.join(train_chk_pt_dir, 'chk_pt')
    test_chk_pt_dir = os.path.join(test_chk_pt_dir, 'chk_pt')

    # Include a log-file that writes out a report so we can figure out why we are crashing!
    run_log_text_path = os.path.join(exp_dir, 'testing_log_.txt')

    # Include a compact mini-fail log
    fail_log_test_path = os.path.join(exp_dir, 'testing_mini_fail_log.txt')

    # Now we also want to include a text file along with all of this to keep track of overall testing results.
    exp_info_text_path = os.path.join(exp_dir, 'testing_info_.txt')

    # Generate the description data in the testing-info file, so one can determine what was tested.
    if not is_restart:
        with open(exp_info_text_path, "w") as info_text_file:
            # Lets fill in some important information in here!
            info_text_file.write('A.Lons & S.Picard')
            info_text_file.write('\n\nDeep Learning - Breast Cancer - Benign vs. Malignant Identification')
            info_text_file.write("\n\n########################################################################################################################")
            info_text_file.write('\n\n                                           EXPERIMENTAL VARIABLES AND PARAMS')
            info_text_file.write('\n\nExamples per Epoch: ' + str(n_examples_per_epoch))
            info_text_file.write('\nBatch size: ' + str(batch_size))
            info_text_file.write('\nNumber of target Epochs: ' + str(n_epochs))
            info_text_file.write('\nBatch normalization: ' + str(batch_norm))
            info_text_file.write('\nRegularization Beta: ' + str(regulizer))
            info_text_file.write('\nLearning rate: ' + str(learning_rate))
            info_text_file.write('\n\nThe file-names of the data used here during testing are ... ')
            for file_name in test_filenames:
                info_text_file.write("\n" + file_name)
            info_text_file.write('\n\nThe description associated with the data is ... ')
            with open(data_description_txt_file, "r") as set_desc:
                # Grab all lines of the description file and put into an array
                lines_of_text = set_desc.readlines()
                # Now put all these lines into our file
                for text_line in lines_of_text:
                    info_text_file.write(text_line)
            info_text_file.write("\n\n########################################################################################################################")
    else:
        with open(fail_log_test_path, "a") as fail_log:
            fail_log.write('\nForced to RESTART TRAINING at ' + str(datetime.now()) + " ... the thread was shut down for some reason")

    with tf.Graph().as_default() as g:

        # Get images and labels,
        with tf.device('/cpu:0'):
            images, labels = ip.input_pipline(test_filenames, batch_size=batch_size, numb_pre_threads=1,
                                              num_epochs=n_epochs+1, output_type='train')

        with tf.device('/cpu:0'):
            # Create the network graph, making sure to set 'is_training' to False.
            prediction = model.generate_res_network(images, batch_size, n_classes, batch_norm=batch_norm,
                                                    is_training=False, on_cpu=True, gpu=0, regulizer=regulizer,
                                                    keep_prob=keep_prob)
            # Find accuracy
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    pred_arg_max = tf.argmax(prediction, 1)
                    labl_arg_max = tf.argmax(labels, 1)
                    correct_prediction = tf.equal(pred_arg_max, labl_arg_max)
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.device('/cpu:0'):
            with tf.name_scope('accuracy'):
                tf.summary.scalar('accuracy', accuracy)

        with tf.device('/cpu:0'):
            with tf.name_scope('global_stepping'):
                global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.device('/cpu:0'):

            # Create saver for writing training checkpoints
            saver = tf.train.Saver(name='tf_graph_loader_op')

            # Keeping track of best results!
            best_accuracy = 0.0

            # Set up counter so that if we surpass a number of epochs without impriving, we jump out of here and set
            # an event so we can jump out of training as well.
            acc_exit_counter = 0

            # Set up counter so we know about which was the last global_step (epoch) that was used before so we do not
            # re-run testing over and over again.
            last_global_step = -1

            # This is just to help us print to screen.
            gen_count = 0

            # Lets keep track of starting run in file
            if not is_restart:
                with open(run_log_text_path, "a") as log_file:
                    log_file.write('\nBeginning training at ' + str(datetime.now()))
                    log_file.write("\nBegin tracking results ... \n")
            else:
                with open(run_log_text_path, "a") as log_file:
                    log_file.write("\n\nRESTARTING METHOD... the thread was shut down for some reason, now we have to "
                                   "rebuild!!")

            # We want to keep on running the evaluation until the other training python thread step is complete!
            while not finished_training_event.is_set() or performance_drop_event.is_set():

                # JUST FOR TESTING THE MULTI-PROCESSING RESTART THREADS
                #if not is_restart and last_global_step == 3:
                #    some_new_var = 2.0/0.

                # Add try anc excepts to handle things not working, because I am getting processes that run away.
                try:

                    # Lets keep track of starting run in file
                    with open(run_log_text_path, "a") as log_file:
                        log_file.write('\n\nStarting next iteration of loop')
                        log_file.write('\n... (a) starting run-time tracking')

                    start_time = time.time()

                    with open(run_log_text_path, "a") as log_file:
                        log_file.write('\n... (b) creating tf.Session')

                    # Create a session
                    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))

                    # AJL (Jan. 27) Somehow the summaries are super big, so I am going to kill them.
                    #with open(run_log_text_path, "a") as log_file:
                    #    log_file.write('\n... (c) merging summaries')
                    #merged = tf.summary.merge_all()
                    #summary_writer = tf.summary.FileWriter(test_smry_dir, g)

                    with open(run_log_text_path, "a") as log_file:
                        log_file.write('\n... (d) initialize variables')

                    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer(), name='initialize_op')
                    sess.run(init_op)

                    with open(run_log_text_path, "a") as log_file:
                        log_file.write('\n... (e) find chk-pt')

                    # Restore the check point, now try to drop this in the actual while loop!
                    chkp = tf.train.latest_checkpoint(train_chk_pt_dir)

                    # Lets keep track of starting run in file
                    with open(run_log_text_path, "a") as log_file:
                        log_file.write('\n... (f) load chk-pt')

                    # Restore tf.graph from file
                    saver.restore(sess, chkp)

                    # Check global-step, if we are not at a new global-step, we do not need to run calculations again.
                    new_global_step = sess.run(global_step)

                    if new_global_step > last_global_step:

                        gen_count = 0
                        last_global_step = sess.run(global_step)

                        print("\nTESTING, starting evaluation of network at Global-Step %d ... \n" %
                              sess.run(global_step))

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write('\n... (g) generate coordinators and threads')

                        coord = tf.train.Coordinator()
                        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                        # Set up the for-loop to run accuracy calculation over testing-data batches
                        num_training_batches = int(n_examples_per_epoch/batch_size)
                        acc_list = []

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write('\n... (h) iterate over number of training batches')

                        for j in range(num_training_batches):

                            # Every so often to write a summary out. This is really just to be sure if we need to,
                            # look back at what was run as a sanity check.
                            if j % 20 == 0:
                                with open(run_log_text_path, "a") as log_file:
                                    log_file.write('\n... (h-a) run accuracy ')
                                #acc, summary_res = sess.run([accuracy, merged])
                                acc = sess.run(accuracy)
                                acc_list.append(acc)
                                #with open(run_log_text_path, "a") as log_file:
                                #    log_file.write('\n... (h-b) write to summary')
                                #summary_writer.add_summary(summary_res, sess.run(global_step * (j + 1)))
                            else:
                                with open(run_log_text_path, "a") as log_file:
                                    log_file.write('\n... (h-a) run accuracy')
                                acc = sess.run(accuracy)
                                acc_list.append(acc)

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write('\n... (i) find accuracy ')

                        # The list of accuracies is actually a list of lists, so we need to loop through and find total.
                        avg_accuracy = sum(acc_list) / float(len(acc_list))

                        if avg_accuracy > best_accuracy:

                            with open(run_log_text_path, "a") as log_file:
                                log_file.write('\n... (i-a) found good accuracy, save the graph')
                            # We need to save this thing
                            # According to some link (https://github.com/tensorflow/tensorflow/issues/1962), adding the
                            # 'write_meta_graph=False' seems to fix peoples problems of dying during save.
                            try:
                                saver.save(sess, test_chk_pt_dir, global_step=global_step, write_meta_graph=False)
                                best_accuracy = avg_accuracy
                            except:
                                print('\nTESTING, problem with saving check-point, not saving the graph')
                                with open(run_log_text_path, "a") as log_file:
                                    log_file.write('\n... (i-a-1) problem with saving check-point, not saving graph')
                                with open(fail_log_test_path, "a") as fail_log:
                                    fail_log.write('\n\nERROR ERROR, oops, failed to save chk-pt try-loop at global-s'
                                                   'tep: ' + str(sess.run(global_step)) + ', at: '
                                                   + str(datetime.now()))
                            # Reset best counter. If this number gets too big (bigger 'than drop_count_limit'), then
                            # we will exit testing and training.
                            acc_exit_counter = 0

                        else:

                            # Increment best counter. If this number gets too big (bigger 'than drop_count_limit'), then
                            # we will exit testing and training.
                            acc_exit_counter += 1
                            with open(run_log_text_path, "a") as log_file:
                                log_file.write('\n... (i-a) found accuracy is not increasing. Drop count at '
                                               + str(acc_exit_counter) + " of " + str(performance_drop_count_limit))

                            # Check if it is time to exit both the training and testing because we are not improving
                            # in accuracy. We do this by breaking out of this, but also by setting a multiprocesing
                            # event which is how we communicate to the training thread that things are not improving.
                            if performance_drop_count_limit < acc_exit_counter:
                                # Set multiprocessing event.
                                performance_drop_event.set()
                                # Now we need to break out of the for-loop, but before we do this we need to close out
                                # of things and clean up the threads.
                                # summary_writer.close()
                                coord.request_stop()
                                coord.join(threads)
                                sess.close()
                                time.sleep(1.0)
                                break

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write('\n... (j) writing testing accuracy to file')

                        # Record accuracy at every step!
                        with open(exp_info_text_path, "a") as info_text_file:
                            # Lets fill in some important information in here!
                            info_text_file.write('\nAccuracy at global training step: ' + str(sess.run(global_step)) + " is: "
                                                 + str(avg_accuracy) + ", best accuracy is: " + str(best_accuracy))

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write('\n... (k) printing out results to screen from last testing iteration')

                        print("\nTESTING, the results of net evaluation at Global-Step %d are ..." %
                              sess.run(global_step))
                        print('  Accuracy:', avg_accuracy)
                        print('  Best Accuracy: ', best_accuracy)
                        print('  Epoch run-time:', time.time() - start_time)
                        print('  Drop count: ', str(acc_exit_counter), " of ", str(performance_drop_count_limit),
                              " allowed\n")

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write('\n... (l) cleaning up threads and coordinator')

                        coord.request_stop()
                        coord.join(threads)
                        #sess.close()
                        time.sleep(2)

                    else:

                        with open(run_log_text_path, "a") as log_file:
                            log_file.write('\n... (g) global-step not new, starting loop over again')

                        # Use 'gen_count' to know if we need to print out to screen once again.
                        if gen_count < 1:
                            print("\nTESTING, NOT starting evaluation, still same global-step as last test iteration\n")
                            gen_count += 1

                        time.sleep(2.0)

                    # I forgot to add this earlier
                    #summary_writer.close()
                    sess.close()

                except:

                    # Lets keep track of the fail in the generic testing-log file.
                    with open(run_log_text_path, "a") as log_file:
                        log_file.write('\n\nERROR ERROR, oops, something went wrong in try-loop!')

                    # Lets keep track of the fails in particular in the fail-log
                    with open(fail_log_test_path, "a") as fail_log:
                        fail_log.write('\n\nERROR ERROR, oops, something went wrong in try-loop at global-step: ' +
                                       str(sess.run(global_step)) + ', at: ' + str(datetime.now()))

                    print("\nTESTING: SOMETHING IN THE TESTING WENT WRONG!!, WILL TRY AGAIN, BEGIN SLEEP FOR A BIT\n")
                    time.sleep(2.0)

            # We are about to leave this method, before doing so we need to write out more to our logs and summary
            # information txt files.

            if performance_drop_event.is_set():
                with open(run_log_text_path, "a") as log_file:
                    log_file.write('\n\nLEAVING TRAINING ... ... ... limit in epochs with no accuracy improvement hit, exited test loop')
                print("\nTESTING, limit in epochs with no accuracy improvement hit, writing to eval_results.txt \n")

            else:
                with open(run_log_text_path, "a") as log_file:
                    log_file.write('\n\nLEAVING TRAINING ... ... ... training iterations finished, exited test loop')
                print("\nTESTING, training finished, writing to eval_results.txt \n")

            # Add text to the final results of the testing, isolating the best results of the bunch.
            cumm_res_text_path = os.path.join(save_path, 'cumulative_results.txt')
            with open(cumm_res_text_path, "a") as info_text_file:
                # Lets fill in some important information in here!
                info_text_file.write("\n\n########################################################################################################################")
                info_text_file.write('\n                                           EXPERIMENTAL RESULTS')
                info_text_file.write('\nResults for experiment in ' + exp_dir)
                info_text_file.write('\nBest Accuracy of :' + str(best_accuracy) )
                info_text_file.write('\n\nExamples per Epoch: ' + str(n_examples_per_epoch))
                info_text_file.write('\nBatch size: ' + str(batch_size))
                info_text_file.write('\nNumber of target Epochs: ' + str(n_epochs))
                info_text_file.write('\nBatch normalization: ' + str(batch_norm))
                info_text_file.write('\nRegularization Beta: ' + str(regulizer))
                info_text_file.write('\nDrop out val: ' + str(keep_prob))
                info_text_file.write('\nLearning rate: ' + str(learning_rate))
                info_text_file.write('\n\nThe file-names of the data used here during testing are ... ')
                for file_name in test_filenames:
                    info_text_file.write("\n" + file_name)
                info_text_file.write('\n\nThe description associated with the data is ... ')
                with open(data_description_txt_file, "r") as set_desc:
                    # Grab all lines of the description file and put into an array
                    lines_of_text = set_desc.readlines()
                    # Now put all these lines into our file
                    for text_line in lines_of_text:
                        info_text_file.write(text_line)
                info_text_file.write("\n\n########################################################################################################################")


if __name__ == "__main__":

    # I may need to see from what directory I am launching this stuff from.
    import os

    train_filenames_list = []
    test_filenames_list = []
    data_description_list = []
    train_numbexamples_list = []
    test_numbexamples_list = []

    # Grab the altered data set ("soft" distortion of rotations and translations)
    train_filenames_list.append(
        ['../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/train/train_data_-00000-of-00004',
         '../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/train/train_data_-00001-of-00004',
         '../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/train/train_data_-00002-of-00004',
         '../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/train/train_data_-00003-of-00004'])
    test_filenames_list.append(
        ['../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/test/test_data_-00000-of-00004',
         '../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/test/test_data_-00001-of-00004',
         '../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/test/test_data_-00002-of-00004',
         '../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/test/test_data_-00003-of-00004'])
    data_description_list.append('../../data_files/tfr_files/altered_sets/cancer_data_altered_1_128x128/set_info.txt')
    train_numbexamples_list.append(6600)
    test_numbexamples_list.append(220)

    # Grab the original data-set
    train_filenames_list.append(
        ['../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/train/train_data_-00000-of-00004',
         '../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/train/train_data_-00001-of-00004',
         '../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/train/train_data_-00002-of-00004',
         '../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/train/train_data_-00003-of-00004'])
    test_filenames_list.append(
        ['../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/test/test_data_-00000-of-00004',
         '../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/test/test_data_-00001-of-00004',
         '../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/test/test_data_-00002-of-00004',
         '../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/test/test_data_-00003-of-00004'])
    data_description_list.append('../../data_files/tfr_files/orig_sets/cancer_data_orig_128x128/set_info.txt')
    train_numbexamples_list.append(1000)
    test_numbexamples_list.append(220)

    save_path = 'saved_results_1/part_a'
    n_classes = 7
    batch_size = 35
    n_epochs = 300
    batch_norm = True
    regulizer_list = [0.0, 0.001, .01, .1, 1.0]
    keep_prob_list = [1.0, .9, .8, .7]
    learning_rate = .001

    number_train_sessions = len(regulizer_list) * len(keep_prob_list) * len(train_filenames_list)
    count = 1
    global_start_time = time.time()

    # Now we also want to include a text file along with all of this, so lets do that now!
    run_info_text_path = os.path.join(save_path, 'training_testing_.txt')
    with open(run_info_text_path, "a") as info_text_file:

        for keep_prob in keep_prob_list:

            for regulizer in regulizer_list:

                for i in range(len(train_filenames_list)):

                    info_text_file.write("\nRunning Training-Testing on data-set "+ str(i) + ", with reg-val: "+ str(regulizer)+ ", with keep_prob:"+ str(keep_prob)+ "\n")

                    print("\n\n\n########################################################################################")
                    print("Beginning training session %d of %d" % (count, number_train_sessions))
                    print("Total run time is " + str(time.time()-global_start_time))
                    print("########################################################################################")

                    # Pull out the correct set of training/testing data and the description txt files
                    train_filenames = train_filenames_list[i]
                    test_filenames = test_filenames_list[i]
                    data_description = data_description_list[i]  # The txt files that describe the data

                    # Create the Multi-Processing events that will handle when the training and the testing threads
                    # each begin
                    finished_first_epoch_event = mp.Event()
                    finished_training_event = mp.Event()
                    acc_not_improve_event = mp.Event()

                    # Pull out the number of training examples
                    n_examples_per_epoch = train_numbexamples_list[i]
                    # Start the training process
                    p1 = mp.Process(target=training_session, args=(False, data_description, finished_first_epoch_event,
                                                                   finished_training_event, acc_not_improve_event,
                                                                   save_path, train_filenames, n_examples_per_epoch,
                                                                   n_classes, batch_size, n_epochs, batch_norm,
                                                                   regulizer, keep_prob, learning_rate))
                    p1.start()
                    # Wait until the Multi-Processing event is set, this occurs only after the first training iteration
                    # is complete, and is necessary so that the testing evaluation actually has something to load!
                    finished_first_epoch_event.wait()

                    # Pull out the number of testing examples
                    n_examples_per_epoch = test_numbexamples_list[i]
                    # Start the testing
                    p2 = mp.Process(target=evaluation_session, args=(False, data_description, finished_training_event,
                                                                     acc_not_improve_event, 20, save_path,
                                                                     test_filenames,
                                                                     n_examples_per_epoch, n_classes, batch_size, n_epochs,
                                                                     batch_norm,
                                                                     regulizer, keep_prob, learning_rate))
                    p2.start()

                    # Now according to http://stackoverflow.com/questions/22125256/python-multiprocessing-watch-a-proces
                    # s-and-restart-it-when-fails, I can set something up to check if threads are dying!
                    while not finished_training_event.is_set() and not acc_not_improve_event.is_set():
                        # First check the training process!
                        time.sleep(15)
                        #print("")
                        #print("P1, Exit-code: " + str(p1.exitcode))
                        #print("P1, is-alive? " + str(p1.is_alive()))
                        #print("P1, finished training? " + str(finished_training_event.is_set()))
                        #print("")
                        if not finished_training_event.is_set() and not acc_not_improve_event.is_set() and not p1.is_alive():
                            # Do error handling and restart the damn thing!
                            n_examples_per_epoch = train_numbexamples_list[i] # Make sure to get correct number of
                            p1 = mp.Process(target=training_session,
                                            args=(True, data_description, finished_first_epoch_event,
                                                  finished_training_event, acc_not_improve_event,
                                                  save_path, train_filenames, n_examples_per_epoch,
                                                  n_classes, batch_size, n_epochs, batch_norm,
                                                  regulizer, keep_prob, learning_rate))
                            p1.start()

                        # Second look at the other evaluation process
                        #print("")
                        #print("P2, Exit-code: " + str(p2.exitcode))
                        #print("P2, is-alive? " + str(p2.is_alive()))
                        #print("P2, not improving? " + str(acc_not_improve_event.is_set()))
                        #print("")
                        if not finished_training_event.is_set() and not acc_not_improve_event.is_set() and not p2.is_alive():
                            n_examples_per_epoch = test_numbexamples_list[i]
                            # Start the testing
                            p2 = mp.Process(target=evaluation_session, args=(True, data_description, finished_training_event,
                                                                             acc_not_improve_event, 20, save_path,
                                                                             test_filenames,
                                                                             n_examples_per_epoch, n_classes,
                                                                             batch_size, n_epochs,
                                                                             batch_norm,
                                                                             regulizer, keep_prob, learning_rate))
                            p2.start()

                    # Wait until training is finished.
                    finished_training_event.wait()

                    count += 1

                    p1.join()
                    p2.join()




