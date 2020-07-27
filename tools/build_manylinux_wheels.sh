#!/bin/bash
set -e -x

# docker run --rm -v $(pwd):/io quay.io/pypa/manylinux2010_x86_64 /bin/bash /io/tools/build_manylinux_wheels.sh
# bash build_manylinux_wheels.sh

# you need to export PLAT so that auditwheel works
# export PLAT=manylinux2010_x86_64
cd /io

# Compile wheels
for PYBIN in /opt/python/cp3[7-8]-cp*/bin; do
  "${PYBIN}/pip" install numpy cython
  "${PYBIN}/pip" wheel --no-deps -w wheelhouse/ .
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
    rm $whl
done

# Install packages and test
# yum install -y hdf5-devel

# need to exit source directory to run tests
cd /home
for PYBIN in /opt/python/cp3[7-8]-cp*/bin; do
  "${PYBIN}/pip" install scipy matplotlib pytest corner
  "${PYBIN}/pip" install --pre --only-binary refnx --no-index --find-links /io/wheelhouse refnx
  "${PYBIN}/pytest" --pyargs refnx.reflect.test.test_reflect refnx.analysis
done
cd /io

mkdir -p /io/dist
cp wheelhouse/refnx*"$PLAT"*.whl dist/