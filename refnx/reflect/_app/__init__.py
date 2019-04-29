import sys


def gui():
    from PyQt5 import QtGui, QtWidgets, QtCore
    from refnx.reflect._app.view import MotofitMainWindow
    from refnx.reflect._app import resources_rc

    app = QtWidgets.QApplication(sys.argv)

    # used to make sure that only decimal point is used for
    # entering floats (3.1), rejecting commas (3,1).
    # This reduces confusion for international users
    lo = QtCore.QLocale(QtCore.QLocale.C)
    lo.setNumberOptions(QtCore.QLocale.RejectGroupSeparator)
    QtCore.QLocale.setDefault(lo)

    app.setWindowIcon(QtGui.QIcon(':icons/scattering.png'))
    myapp = MotofitMainWindow()

    fnt = QtGui.QFont('Arial')
    fnt.setPixelSize(12)
    app.setFont(fnt)

    myapp.show()
    myapp.raise_()

    v = app.exec_()
    return v


def main():
    sys.exit(gui())


__all__ = [gui, main]
