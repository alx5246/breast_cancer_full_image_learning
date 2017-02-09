A.Lons
Feb 2017

This is a simplified version of the code in experiment-2. Here I am tyring to figure out where my bottle-necks are, why
my run-times SUCK!

1.) I changed to tensorflow r1.0 from rc.12, and this might have made only a small difference in run time.

2.) I put the 'global_step' on the GPU by making the variabel a tf.float32, which I did not before, this made a huge
    timing difference.

3.) I put everyting on the GPU, which some of my last layers were originally not, because I forgot to explicitly pin
    to the GPU.

4.) I put the batch-normalization of the GPU, which I was able to do when I made all the values in the calculations
    tf.float32 values.

5.) The different optimizers do not make much of a difference.

6.) I put the titan X inteh box, which makes things much fast, and the time-lines seem to actually work better, I am
    not sure why.

7.) Capped gradients, along with regularization do not seem to slow things down nearly at all.

8.) Batch-normalization, is THE SLOWEST PART, even after fixing and dropping onto the GPU.

########################################################################################################################
1) Going from rc.12 to r1.0, I have this error Couldn't open CUDA library libcupti.so.8.0. LD_LIBRARY_PATH:, I found
    a way to help this out
    https://github.com/tensorflow/tensorflow/issues/7110
    But it did not work, I have to go from,
    LD_LIBRARY_PATH=”$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64"
    to instead
    LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

    I also wondered if I need to be pointing to cuda 8.0, so lets try changing,
    CUDA_HOME=/usr/local/cuda and LD_LIBRARY_PATH=”$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
    to instead CUDA_HOME=/usr/local/cuda-8.0 and LD_LIBRARY_PATH=”$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64"

