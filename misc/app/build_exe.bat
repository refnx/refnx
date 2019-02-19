call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"
call conda env remove -y -n app
call conda create -y -n app python=3.7 pip
call activate app
pip install pyqt5 numpy==1.15.2 scipy cython matplotlib periodictable pyinstaller
pip install ../../.
if exist dist( rmdir /s/q dist)
if exist dist( rmdir /s/q build)
pyinstaller motofit.spec
