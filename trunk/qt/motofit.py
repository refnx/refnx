import sys
from PySide import QtCore, QtGui

from View import MyMainWindow

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = MyMainWindow()

    myapp.show()
    sys.exit(app.exec_())
