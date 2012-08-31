rm -rf build
rm -rf dist
python setup.py py2app
touch ./dist/motofit.app/Contents/Resources/qt.conf
