#!/bin/bash
set -e -x

# docker run --rm -it -v $(pwd):/io quay.io/pypa/manylinux2010_x86_64 /bin/bash
# you need to export PLAT so that auditwheel works
# export PLAT=manylinux2010_x86_64
# cd io
# bash build_manylinux_wheels.sh

# Compile wheels
for PYBIN in /opt/python/cp3[5-9]-cp*/bin; do
  "${PYBIN}/pip" install numpy cython
  "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/cp3[5-9]-cp*/bin; do
  "${PYBIN}/pip" install refnx --no-index -f /io/wheelhouse
  "${PYBIN}/pip" install scipy matplotlib pytest corner uncertainties h5py xlrd periodictable pandas
  ${PYBIN}/python setup.py test -a refnx/analysis
done
