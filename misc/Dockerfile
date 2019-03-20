FROM ubuntu:latest

RUN apt-get update

RUN apt-get install -y python3 git build-essential python3-pip libgl1-mesa-dev

RUN pip3 install numpy cython scipy matplotlib pytest uncertainties h5py xlrd periodictable pandas pytest-qt pyqt5

RUN cd home; git clone https://github.com/refnx/refnx.git