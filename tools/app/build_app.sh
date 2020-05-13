#!/bin/bash
conda create -n app -y python=3.7
conda activate app
pip install pyqt5 numpy==1.15.2 scipy cython matplotlib periodictable pyinstaller
pip install ../../.
rm -rf build dist
pyinstaller motofit.spec