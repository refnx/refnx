#!/bin/bash
conda create -n app -y python=3.9
conda activate app
pip install pyqt6 numpy==1.15.2 scipy cython matplotlib periodictable pyinstaller
pip install ../../.
rm -rf build dist
pyinstaller motofit.spec