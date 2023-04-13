import sys
import time
import os
import psutil
import subprocess


if __name__ == "__main__":
    # provide the path to the executable that you wish to check
    # print(sys.argv[1:])
    try:
        proc = subprocess.Popen(sys.argv[1:])
    except subprocess.TimeoutExpired:
        pass
    # leave time to start up
    pid = proc.pid
    time.sleep(10.0)

    l = (p.pid for p in psutil.process_iter())

    if pid not in l:
        raise RuntimeError("Motofit did not seem to be able to start.")
    else:
        # program started
        proc.kill()
