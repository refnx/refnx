import sys


def gui():
    from PyQt5 import QtGui, QtWidgets
    from refnx.reflect._app.view import MotofitMainWindow
    from refnx.reflect._app import resources_rc

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(':icons/scattering.png'))
    myapp = MotofitMainWindow()

    myapp.show()
    myapp.raise_()

    sys.exit(app.exec_())


__all__ = [gui]
