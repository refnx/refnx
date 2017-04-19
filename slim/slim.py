import sys
import logging
import time
import os.path

from PyQt5 import QtCore, QtGui, QtWidgets
from view import SlimWindow
import resources_rc


# find out where the .ui and data files are.
if getattr(sys, 'frozen', False):
    # running in a bundle
    ui_loc = os.path.join(sys._MEIPASS, 'ui')
else:
    # running live
    ui_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui')


def run():
    time_str = time.strftime('%Y%m%d-%H%M%S')
    log_filename = 'slim_' + time_str + '.log'
    log_filename = os.path.join(os.path.expanduser('~'), log_filename)

    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    logging.info(ui_loc)
    logging.info('Starting SLIM reduction')

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(':/icons/scattering.png'))
    myapp = SlimWindow(ui_loc)

    myapp.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
