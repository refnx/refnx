import sys


def gui(expt_file=None):
    from PyQt5 import QtGui, QtWidgets, QtCore

    # should enable high resolution on 4k desktops??
    # "https://coad.ca/2017/05/15/
    # one-way-to-deal-with-high-dpi-4k-screens-in-python/"
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.AA_EnableHighDpiScaling, True
        )
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.AA_UseHighDpiPixmaps, True
        )

    from refnx.reflect._app.view import MotofitMainWindow
    from refnx.reflect._app import resources_rc

    app = QtWidgets.QApplication(sys.argv)

    # used to make sure that only decimal point is used for
    # entering floats (3.1), rejecting commas (3,1).
    # This reduces confusion for international users
    lo = QtCore.QLocale(QtCore.QLocale.C)
    lo.setNumberOptions(QtCore.QLocale.RejectGroupSeparator)
    QtCore.QLocale.setDefault(lo)

    app.setWindowIcon(QtGui.QIcon(":icons/scattering.png"))
    myapp = MotofitMainWindow()

    fnt = QtGui.QFont("Arial")
    fnt.setPointSize(12)
    app.setFont(fnt)

    if expt_file is not None:
        myapp._load_files_different_types(expt_file)

    myapp.show()
    myapp.raise_()

    v = app.exec_()
    return v


def main(args=None):
    if args is None:
        expt_file = sys.argv[1:]
    else:
        expt_file = args

    sys.exit(gui(expt_file=expt_file))


__all__ = [gui, main]
