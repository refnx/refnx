import sys
import logging
import time
import os.path


def gui():
    from PyQt5 import QtWidgets
    from refnx.reduce._app.view import SlimWindow

    time_str = time.strftime("%Y%m%d-%H%M%S")
    log_filename = "slim_" + time_str + ".log"
    log_filename = os.path.join(os.path.expanduser("~"), log_filename)

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logging.info("Starting SLIM reduction")

    app = QtWidgets.QApplication(sys.argv)
    myapp = SlimWindow()

    myapp.show()
    v = app.exec_()
    return v


def main():
    sys.exit(gui())


__all__ = [gui, main]
