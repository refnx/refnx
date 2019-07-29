#!/bin/bash

# This script is able to make and test macOS wheels. To do this change
# MAKE_OSX_WHEELS=1.

# However, the main purpose of the script is to act as a test script for
# Linux wheels. It assumes that the wheels are in /io/wheelhouse.
# The following docker command mounts PWD as /io in the docker image.
# docker run --rm -it -v $(pwd):/io continuumio/miniconda3 /bin/bash

pythons=(
    3.7
    3.6
    3.5
)

MAKE_OSX_WHEELS=0

# test wheels
for PY in "${pythons[@]}"; do
    echo $PY
    env_name="test_wheel$PY"
    conda create -q -y -n $env_name python=$PY numpy scipy matplotlib h5py xlrd pandas tqdm pyqt jupyter cython pytest &> /dev/null
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    if [ $? -eq 1 ]; then
        echo "Couldn't activate $env_name"
        conda deactivate
        conda env remove -y -n $env_name
        exit 1
    fi
    pip install periodictable corner uncertainties pytest-qt
    
    # make OSX wheels
    if [ $MAKE_OSX_WHEELS -eq 1 ]; then
        pip wheel ./ -w io/wheelhouse/
    fi

    pip install refnx --no-index -f io/wheelhouse
    if [ $? -eq 1 ]; then
        echo "Couldn't find wheel for $PY"
        conda deactivate
        conda env remove -y -n $env_name
        continue
    fi
    pytest --pyargs refnx
#     python -c 'import refnx;refnx.test()'
    if [ $? -eq 0 ]; then
        echo "Tests passed for $PY"
    else
        echo "Tests failed for $PY"
        conda deactivate
        conda env remove -y -n $env_name
        exit 1
    fi
    conda deactivate
    conda env remove -y -n $env_name
done