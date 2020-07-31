"""
Auto reduce reflectometry files
"""
#!/usr/bin/env python
import os
import sys
import time
import argparse
from refnx.reduce import AutoReducer


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-reduce reflectometry files.")
    parser.add_argument("file_list", nargs='+')
    parser.add_argument("-p", "--path", type=dir_path, default="./", help="path to datafiles")
    parser.add_argument("-s", "--scale", help="scale factor",
                        type=float, default=1.0)
    args = parser.parse_args()
    files = args.file_list
    pth = args.path
    files = [os.path.join(pth, file) for file in files]
    print(f"Path: {pth}")
    print(f"Reducing against: {files}, with scale: {args.scale}")
    ar = AutoReducer(files, data_folder=pth, scale=args.scale)

    while True:
        time.sleep(10.)
