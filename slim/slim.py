import sys
import logging
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from view import SlimWindow
import resources_rc


def run():
    time_str = time.strftime('%Y%m%d-%H%M%S')
    log_filename = 'slim_' + time_str + '.log'
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    logging.info('Starting SLIM reduction')

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(':/icons/scattering.png'))
    myapp = SlimWindow()

    myapp.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
