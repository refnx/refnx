import sys
import logging
import time
from pathlib import Path


def gui():
    from qtpy import QtWidgets
    from refnx.reduce._app.view import SlimWindow

    time_str = time.strftime("%Y%m%d-%H%M%S")
    log_filename = "slim_" + time_str + ".log"
    log_filename = Path.home() / log_filename

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logging.info("Starting SLIM reduction")

    app = QtWidgets.QApplication(sys.argv)
    myapp = SlimWindow()

    myapp.show()
    v = app.exec()
    return v


def main():
    sys.exit(gui())


__all__ = [gui, main]
