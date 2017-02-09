# A.Lons
# Jan. 2017
#
# Here is my second experiment where I am coolecting lots of data, having picked a data-set I like to work with.
#
# In this case I am ONLY running data from ...augmented_sets/set_01... which is the altered and then augmented data-set.
# These tests are here to try out different drop-out rates, regularization rates, and decay-rates for learning. I am
# using the ADAM optimizer.
#
# PROBLEM, I collected a bunch of data and AdadeltaOptimizer NOT the ADAM optimzer, so these experiments might not
# be super useful, however these have shown that picking reg. vals and decayign learning rate are dependent on
# eachother.
#
# I am moving onto experiment 2 in order to make a larger res-net, something that can run 128, 256, and 512 images on
# the same network, as well use the momentum based optimzer as suggested originally with the res-net strucutre.