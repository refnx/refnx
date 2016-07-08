python setup.py install
cd ./doc

make clean
make html

cd ..

cp -r ./doc/_build/html/* ../refnx.github.io