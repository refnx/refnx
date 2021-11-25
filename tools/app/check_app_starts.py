import sys
import time
import psutil
import subprocess


if __name__ == "__main__":
    # provide the path to the executable that you wish to check
    # print(sys.argv[1:])
    try:
        proc = subprocess.run(sys.argv[1:], timeout=(10.))
    except subprocess.TimeoutExpired:
        pass
    # leave time to start up
    time.sleep(50.0)

    l = {p.name():p.pid for p in psutil.process_iter()}

    if 'motofit' not in l:
        raise RuntimeError("Motofit did not seem to be able to start.")
    else:
        # motofit started
        p = psutil.Process(l['motofit'])
        p.terminate()
