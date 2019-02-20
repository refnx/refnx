call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"
call conda env remove -y -n app
call conda create -y -n app python=3.7 pip
call activate app
call conda install -y -c conda-forge numpy
REM pyinstaller should be installed from git repo, the pypi version doesn'
pip install pyqt5 cython matplotlib periodictable pyinstaller scipy
pip install ../../.
if exist dist ( rmdir /s/q dist)
if exist build ( rmdir /s/q build)
pyinstaller motofit.spec
pause