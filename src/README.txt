# A.Lons
# Jan. 2017
#
#

########################################################################################################################
Information that very much helped me out.

1) "The Road to TensorFlow - Part 11: Generalization and Overfitting", this reminds us of some simple ways to avoid
    these problems. In particular using weight penalization and drop out.

2) Stack Over Flow - "Does the convolutional layer in TensorFlow support droput?", shows us how to use drop out
    correctly, which I tested here.
    http://stackoverflow.com/questions/38983963/does-the-convolutional-layer-in-tensorflow-support-dropout

3) I am having problems with Nans showing up. Apparently this can happen for a number of reasons. (1) is taking the
    the square root in my loss, which can be very unstable. An option here is not using the square root or adding a
    small number before taking the square-root. Another approach is to use gradient clipping. Also "Gradients tend to
    grow in magnitude as training progresses, and sometimes that introduces stability problems", this is why people end
    up decaying the learning rate.
    https://github.com/tensorflow/tensorflow/issues/323

4) I am having problems with Nans showing up. As the network converges the graients grow in size! This is why people
    change the learning rate as they go.
    https://github.com/tensorflow/tensorflow/issues/323

5) Going from rc.12 to r1.0, I have this error Couldn't open CUDA library libcupti.so.8.0. LD_LIBRARY_PATH:, I found
    a way to help this out
    https://github.com/tensorflow/tensorflow/issues/7110
    But it did not work, I have to go from,
    LD_LIBRARY_PATH=”$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64"
    to instead
    LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

    I also wondered if I need to be pointing to cuda 8.0, so lets try changing,
    CUDA_HOME=/usr/local/cuda and LD_LIBRARY_PATH=”$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
    to instead CUDA_HOME=/usr/local/cuda-8.0 and LD_LIBRARY_PATH=”$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64"

