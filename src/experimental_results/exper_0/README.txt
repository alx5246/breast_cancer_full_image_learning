A. Lons
Jan 2017

In experiment 1, we will play with a number of different things, while keeping the residual network the same. We will
try altering the following and observe the results, essentailly doing a hyper-parameter search!
    (a) drop-out rate: [1.0, .9, .8, .7]
    (b) regularization rate: [0.0, .001, .01, .1, 1.0]
    (c) different data-sets:
        (1) orig-128X128
        (2) altered 128X128
        (3) altered & augmented 128X128

Why do we have two different exp_0_main python scripts? I have not made an elegant way to run two different simulations
at once. Thus what I do is have one script that runs half of the training-testing, and the other script runs the
other half of the training-testing. Currently the difference is each script uses different input data.

What we have learned here is how to use multi-processing to run multi things at the same time, and restart them if
they fail out. I will move onto Experiment 1 to actually collect some data.