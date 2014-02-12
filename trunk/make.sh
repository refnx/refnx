rm -rf build
rm -rf dist
python qt/setup.py py2app
touch ./dist/motofit.app/Contents/Resources/qt.conf
