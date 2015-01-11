rm -rf build
rm -rf dist
python qt/setup.py py2app
hdiutil create -volname Motofit -srcfolder dist -ov -format UDZO name.dmg
touch ./dist/motofit.app/Contents/Resources/qt.conf
