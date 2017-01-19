# A.Lons
# Jan. 2017
#
# DESCRIPTION
# I am trying to figure out how to write results to a .txt file, so we can accumulate results over multiple runs and
# trials.


with open("src/tutorial_test/some_test_0.txt") as text_file:
    text_file.write("Param 1 %d and Param 2 %d" % (23, 43))
