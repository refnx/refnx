import sys


def gui():
    from PyQt5 import QtGui, QtWidgets
    from refnx.reflect._app.view import MotofitMainWindow
    from refnx.reflect._app import resources_rc

    app = QtWidgets.QApplication(sys.argv)
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
