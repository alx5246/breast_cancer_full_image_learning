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




def training_session(data_description_txt_file, finished_first_epoch_event, finished_training_event, save_path,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # To make sure this is actually working
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())

    # PATH NOTES: we are within the exper_0/ folder, thus everything we save must be relative to that path.
    # Thus if we want to go up one folder we would have to use ../ or two folders ../../

    # Deal with folder creation. For each experiment we want a separate folder. In said folder we will have folders for
    # saving training-check-points, best-evaluation-check-points, and summary-writers.

    # We need to make a new directory for our experiment, thus we need to see what directories already exist! To do this
    # we are going to find out how many directories already exist, and then make a new one!
    list_of_dirs = os.listdir(save_path)
    experiment_numb = len(list_of_dirs) + 1
    # Now we make new directories, first by making the path names
    exp_dir = os.path.join(save_path, '_exper_0_' + str(experiment_numb).zfill(4))
    train_chk_pt_dir = os.path.join(exp_dir, 'train_chk_pt')
    test_chk_pt_dir = os.path.join(exp_dir, 'test_chk_pt')
    train_smry_dir = os.path.join(exp_dir, 'train_smry_dir')
    test_smry_dir = os.path.join(exp_dir, 'test_smry_dir')

    # Now we actually make the directories where the check-points and the summaries will be stored!
    os.makedirs(exp_dir)
    os.makedirs(train_chk_pt_dir)
    os.makedirs(test_chk_pt_dir)
    os.makedirs(train_smry_dir)
    os.makedirs(test_smry_dir)

    # Now make the actual path to the check-point, because loading and saving need to point differently I guess.
    train_chk_pt_dir = os.path.join(train_chk_pt_dir, 'chk_pt')
    test_chk_pt_dir = os.path.join(test_chk_pt_dir, 'chk_pt')

    # Now we also want to include a text file along with all of this, so lets do that now!
    exp_info_text_path = os.path.join(exp_dir, 'training_info.txt')
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

        with tf.device('/cpu:0'):
            # Get images and labels,
            images, labels = ip.input_pipline(train_filenames, batch_size=batch_size, numb_pre_threads=4,
                                              num_epochs=n_epochs+1, output_type='train')

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

        with tf.device('/gpu:0'):
            with tf.name_scope('optimizer'):
                # Now generate optimizer!
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss,
                                                                                         name='adam_optimizer')

        with tf.device('/cpu:0'):
            with tf.name_scope('global_stepping'):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                increment_global_step_op = tf.assign(global_step, global_step + 1)

        with tf.device('/cpu:0'):

            # Create a session, but make sure we donot keep logs and also do not allow TF to place operations where it
            # wants to, but rather where we want to.
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))

            # We want to save some summaries for this training-portion
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(train_smry_dir, sess.graph)

            # Create saver for writing training checkpoints, these checkpoints are necessary for allowing an
            # evaluation function to open the graph and evaluate it's performance.
            saver = tf.train.Saver()

            # This is done in one how-to example and in cafir-10 example. NOTE, i have to add the
            # tf.local_variables_init() because I set the num_epoch in the string producer in the other python file.
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                               name='initialize_op')

            # Run the init operation
            sess.run(init_op)

            # Make a coordinator,
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Run training for a specific number of training examples. It seems the 'int()' acts like a 'floor' type
            # function.
            num_training_batches = int(n_examples_per_epoch / batch_size)

            # Lets keep track of starting run in file
            with open(exp_info_text_path, "a") as info_text_file:
                # Lets fill in some important information in here!
                info_text_file.write("\n\n########################################################################################################################")
                info_text_file.write('\n\n Beginning training at ' + str(datetime.now()))

            # Keep track of total time
            total_start_time = time.time()

            for epoch_iteration in range(n_epochs):

                # Some multiprocessing things, we want to set this event to true after we finish the first training
                # epoch so we can start calling the evaluation
                if epoch_iteration == 1:
                    finished_first_epoch_event.set()

                # Track time for batches, and increment global_step
                run_batch_times = []
                sess.run(increment_global_step_op)

                print('\nTRAINING Epoch %d Started...\n' % epoch_iteration)

                for j in range(num_training_batches):

                    #print('\nTRAINING Epoch :', epoch_iteration, ', batch:', j)
                    # Track time to run each epoch, and then run optimizer, and then record run-time
                    start_time = time.time()
                    _ = sess.run([optimizer])
                    run_batch_times.append(time.time()-start_time)

                    if j % 100 == 0:
                        summary_str = sess.run(summary_op)
                        step_number = (epoch_iteration+1)*num_training_batches + (j+1)
                        summary_writer.add_summary(summary_str, step_number)

                print('\nTRAINING Epoch %d Complete,  ' % epoch_iteration)
                avg_run_time = sum(run_batch_times) / float(len(run_batch_times))
                print('  Date: ', datetime.now())
                print('  Total run time: ', time.time()-total_start_time)
                print('  Avg batch run time :', avg_run_time)
                print('  Number of batch runs: ', len(run_batch_times), "\n")

                # Now save the graph. There are some important parameters we want to keep around here. Most important
                # is "global_step" which will be the epoch count for us. Also we can set the make points kept.
                path_to_checkpoint = saver.save(sess, train_chk_pt_dir, global_step=global_step)

            # Lets keep track of starting run in file
            with open(exp_info_text_path, "a") as info_text_file:
                # Lets fill in some important information in here!
                info_text_file.write('\n Ending training at ' + str(datetime.now()))

            # Clean up all the tensor flow things!
            summary_writer.close()
            coord.request_stop()
            coord.join(threads)
            sess.close()

    # Now do the multi-processing thing... which is .set() the event, which will hopefully tell
    finished_training_event.set()


def evaluation_session(data_description_txt_file, finished_training_event, save_path, test_filenames, n_examples_per_epoch, n_classes, batch_size,
                       n_epochs, batch_norm, regulizer, keep_prob, learning_rate):

    # We need to import the os in order to handle files and setting up CUDA correctly
    import os
    # Make sure we set the visable CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # To make sure this is actually working
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())

    # The first step is going to be to find the directory where we are saving the training data, it should be the
    # directory with the largest number! We will find this one by looping over
    list_of_dirs = os.listdir(save_path)
    val = -1
    ind = 0
    for zx, dir_name in enumerate(list_of_dirs):
        last_chars = dir_name[-4:]
        if last_chars.isdigit():
            if val < int(last_chars):
                ind = zx
                val = int(last_chars)
    # Now we know the path name!
    exp_dir = os.path.join(save_path, list_of_dirs[ind])
    train_chk_pt_dir = os.path.join(exp_dir, 'train_chk_pt')
    test_chk_pt_dir = os.path.join(exp_dir, 'test_chk_pt')
    train_smry_dir = os.path.join(exp_dir, 'train_smry_dir')
    test_smry_dir = os.path.join(exp_dir, 'test_smry_dir')

    # Wait are these right? Only sort of. In order to load a chk_pt we have to point to the directory, in order to save
    # we have to point to the actual file! STUPID
    # train_chk_pt_dir = os.path.join(train_chk_pt_dir, 'chk_pt')
    test_chk_pt_dir = os.path.join(test_chk_pt_dir, 'chk_pt')

    # Now we also want to include a text file along with all of this, so lets do that now!
    exp_info_text_path = os.path.join(exp_dir, 'testing_info_.txt')
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

    with tf.Graph().as_default() as g:

        # Get images and labels,
        with tf.device('/cpu:0'):
            images, labels = ip.input_pipline(test_filenames, batch_size=batch_size, numb_pre_threads=4,
                                              num_epochs=n_epochs+1, output_type='train')

        with tf.device('/gpu:0'):
            # Create the network graph, making sure to set 'is_training' to False.
            prediction = model.generate_res_network(images, batch_size, n_classes, batch_norm=batch_norm,
                                                    is_training=False, on_cpu=False, gpu=0, regulizer=regulizer,
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

            # Lets keep track of starting run in file
            with open(exp_info_text_path, "a") as info_text_file:
                info_text_file.write('\n\n Beginning training at ' + str(datetime.now()))
                info_text_file.write("\n\n Begin tracking results ... ")

            # We want to keep on running the evaluation until the other training python thread step is complete!
            while not finished_training_event.is_set():
            #while best_accuracy < .0001:

                start_time = time.time()

                # Create a session
                sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False))

                merged = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(test_smry_dir, g)

                init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer(), name='initialize_op')

                sess.run(init_op)

                # Restore the check point
                chkp = tf.train.latest_checkpoint(train_chk_pt_dir)
                saver.restore(sess, chkp)

                print("\nTESTING, the global step value is %d ... \n" % sess.run(global_step))

                # Make a coordinator,
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                num_training_batches = int(n_examples_per_epoch/batch_size)
                acc_list = []

                for j in range(num_training_batches-3):

                    # Every so often to write a summary out. This is really just to be sure if we need to, look back
                    # at what was run as a sanity check.
                    if j % 20 == 0:
                        acc, summary_res = sess.run([accuracy, merged])
                        acc_list.append(acc)
                        summary_writer.add_summary(summary_res, sess.run(global_step * (j + 1)))
                    else:
                        acc = sess.run(accuracy)
                        acc_list.append(acc)

                # The list of accuracies is actually a list of lists, so we need to loop through and find total.
                avg_accuracy = sum(acc_list) / float(len(acc_list))

                if avg_accuracy > best_accuracy:
                    # We need to save this thing
                    saver.save(sess, test_chk_pt_dir, global_step=global_step)
                    best_accuracy = avg_accuracy
                    #
                    # Lets keep track of starting run in file
                    with open(exp_info_text_path, "a") as info_text_file:
                        # Lets fill in some important information in here!
                        info_text_file.write('\nAccuracy at global step: ' + str(sess.run(global_step)) + " is: " + str(best_accuracy))

                print("\nTESTING, Global-Step ", sess.run(global_step))
                print('  Accuracy:', avg_accuracy)
                print('  Best Accuracy: ', best_accuracy)
                print('  Epoch run-time:', time.time() - start_time, "\n")

                coord.request_stop()
                coord.join(threads)
                sess.close()

                #break

            # Once we are kicked out and forced to quit evaluating, now we want to record our final results in the
            # cumulative evaluation form. That is where we will record all of the results.
            print("\nTESTING, Exiting evaluation loop, writing to 'eval_results.txt' \n")

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
                info_text_file.write('\nLearning rate: ' + str(learning_rate))
                info_text_file.write('\n\nThe file-names of the data used here during testing are ... ')
                for file_name in test_filenames:
                    info_text_file.write("\n" + file_name)
                info_text_file.write("\n\n########################################################################################################################")


if __name__ == "__main__":

    import os
    print(os.path.dirname(__file__))

    train_filenames = ['../../data_files/tfr_files/augmented_sets/set_00/train/train_data_-00000-of-00004',
                       '../../data_files/tfr_files/augmented_sets/set_00/train/train_data_-00001-of-00004',
                       '../../data_files/tfr_files/augmented_sets/set_00/train/train_data_-00002-of-00004',
                       '../../data_files/tfr_files/augmented_sets/set_00/train/train_data_-00003-of-00004']

    data_description = '../../data_files/tfr_files/augmented_sets/set_00/set_info.txt'

    test_filenames = ['../../data_files/tfr_files/augmented_sets/set_00/test/test_data_-00000-of-00004',
                      '../../data_files/tfr_files/augmented_sets/set_00/test/test_data_-00001-of-00004',
                      '../../data_files/tfr_files/augmented_sets/set_00/test/test_data_-00002-of-00004',
                      '../../data_files/tfr_files/augmented_sets/set_00/test/test_data_-00003-of-00004']


    finished_first_epoch_event = mp.Event()
    finished_training_event = mp.Event()
    save_path = 'saved_results'
    n_examples_per_epoch = 100
    n_classes = 7
    batch_size = 20
    n_epochs = 10
    batch_norm = True
    regulizer = .8
    keep_prob = .8
    learning_rate = .001

    n_examples_per_epoch = 1000
    p1 = mp.Process(target=training_session, args=(data_description, finished_first_epoch_event, finished_training_event, save_path,
                                                   train_filenames, n_examples_per_epoch, n_classes, batch_size,
                                                   n_epochs, batch_norm, regulizer, keep_prob, learning_rate))
    p1.start()
    finished_first_epoch_event.wait()

    n_examples_per_epoch = 200
    p2 = mp.Process(target=evaluation_session, args=(data_description, finished_training_event, save_path, test_filenames,
                                                     n_examples_per_epoch, n_classes, batch_size, n_epochs, batch_norm,
                                                     regulizer, keep_prob, learning_rate))
    p2.start()
    finished_training_event.wait()

    p1.join()
    p2.join()




