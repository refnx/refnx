import sys
from PySide import QtCore, QtGui

from View import MotofitMainWindow
import resources_rc


def run():
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(':/icons/scattering.png'))
    myapp = MotofitMainWindow()

    myapp.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
