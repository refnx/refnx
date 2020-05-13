# docker build -t andyfaff/refnx-dev
# docker run -it -v ~/Documents/Andy/programming/refnx:/home/refnx andyfaff/refnx-dev

FROM ubuntu:latest

RUN apt-get update

RUN apt-get install -y python3 git build-essential python3-pip vim libcr-dev mpich
# libgl1-mesa-dev

RUN pip3 install numpy cython scipy matplotlib pytest uncertainties h5py xlrd periodictable pandas pytest-qt pyqt5 mpi4py schwimmbad joblib
