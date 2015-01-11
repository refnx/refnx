rm -rf build
rm -rf dist
python qt/setup.py py2app
rm -rf dist/Motofit.app/Contents/Resources/lib/python2.7/matplotlib/tests
dropdmg -g Motofit.app dist/Motofit.app
