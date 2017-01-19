import sys
import  os
print(os.path.dirname(__file__))

# Now try reading from the file
with open("../../README.txt", "r") as text_file:
    for line in text_file:
        print(line, end='')


# Check directories and shit
d = os.path.dirname('saved_results/some_dir')
if not os.path.exists(d):
    os.makedirs(d)