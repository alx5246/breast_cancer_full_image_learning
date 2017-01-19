# A.Lons
# Jan 2017
#
# DESCRIPTION
# Trying to figure out how to use the multi-processing toolbox so I can control my scripts.
#
# IMPORTANT HELP
#
# 1) This first link is most helpful. It tells me that I can use events. Essentially I use set one event when a process
#   completes. Then I use event.wait() which waits until the event flag is true, and then the second event flag is set
#   as well. I added join() at the end to make sure I quit all the threads together.
#   http://stackoverflow.com/questions/36962462/terminate-a-python-multiprocessing-program-once-a-one-of-its-workers-meets-a-cer
#
# 2) http://stackoverflow.com/questions/33447055/python-multiprocess-pool-how-to-exit-the-script-when-one-of-the-worker-process/33450972#33450972
#

import random
from time import sleep
import multiprocessing as mp

def worker(i, quit, foundit):
    print("%d started" % i)
    while not quit.is_set():
        x = random.random()
        if x > 0.95:
            print('%d found %g' % (i, x))
            foundit.set()
            break
        sleep(0.1)

    print("%d is done" % i)

if __name__ == "__main__":


    epochs = 3
    for epoch in range(epochs):
        print("\nStarting Epoch %d" % epoch)

        foundit = mp.Event()
        quit = mp.Event()
        processes = []
        for i in range(4):
            p = mp.Process(target=worker, args=(i, quit, foundit))
            p.start()
            processes.append(p)
        # According to the Event() documentation: This "wait" Blocks until the internal flag is true. If the internal
        # flag is true on entry it will return immediately. Othersie it blocks until another thread calls set() to
        # set the flag to true
        foundit.wait()
        quit.set()

        # Added this so that we wait for all the processes to end.
        for proc in processes:
            proc.join()


