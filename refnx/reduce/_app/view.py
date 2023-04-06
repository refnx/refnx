import pickle
import os.path
import logging

from qtpy.compat import getopenfilename, getsavefilename
from qtpy import QtCore, QtWidgets, uic
from qtpy.QtCore import Qt
from refnx.reduce.manual_beam_finder import ManualBeamFinder
from .plot import SlimPlotWindow
from .model import ReductionTableModel, ReductionState


UI_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


class SlimWindow(QtWidgets.QMainWindow):
    """
    SLIM is an application for reducing neutron reflectometry data
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = uic.loadUi(os.path.join(UI_LOCATION, "slim.ui"), self)

        # event dialogue
        self.event_dialog = EventDialog(self)

        # options dialogue
        self.reduction_options_dialog = ReductionOptionsDialog(self)

        # reduction state contains all the file numbers to be reduced
        # and all the reduction options information. You could pickle this file
        # to save the state of the program
        self._reduction_state = ReductionState()

        self.ReductionTable = ReductionTableModel(
            self.reduction_state, parent=self
        )
        self.ui.file_list.setModel(self.ReductionTable)

        # a plot window for displaying datasets
        self._plot = SlimPlotWindow(self)
        self.output_directory.textChanged.connect(
            self._plot.data_directory_changed
        )

    @property
    def reduction_state(self):
        return self._reduction_state

    def set_state(self, state):
        """
        Recreates the GUI from a ReductionState instance

        Parameters
        ----------
        state : ReductionState
        """
        if not isinstance(state, ReductionState):
            return

        # restore the directories
        self.ui.streamed_directory.setText(state.streamed_directory)
        self.ui.output_directory.setText(state.output_directory)
        self.ui.data_directory.setText(state.data_directory)

        # restore reduction options dialog
        rod = self.reduction_options_dialog

        for attr, val in state.default_reduction_options.items():
            gui_element = getattr(rod, attr)
            state_element = getattr(state, attr)

            if val[1] is bool:
                if state_element:
                    gui_element.setCheckState(Qt.CheckState.Checked)
                else:
                    gui_element.setCheckState(Qt.CheckState.Unchecked)

            elif val[1] is float:
                gui_element.setValue(state_element)

        # mark the reducer table as changed
        self.ReductionTable.dataChanged.emit(
            QtCore.QModelIndex(), QtCore.QModelIndex()
        )

        self._reduction_state = state

        # just use default manual beam finder dialog
        self._reduction_state.manual_beam_finder = self.manual_beam_finder

    @QtCore.Slot()
    def on_reduce_clicked(self):
        """
        Performs a reduction in response to the reduce plot_button being
        clicked
        """
        if self.reduction_state.manual_beam_finder is None:
            mbf = ManualBeamFinder()
            self.manual_beam_finder = mbf
            self._reduction_state.manual_beam_finder = mbf

        redn_state = self._reduction_state

        # if you're doing event mode you need to know how long
        # each time slice is
        if redn_state.streamed_reduction:
            ok = self.event_dialog.exec()
            if not ok:
                return

            redn_state.stream_start = self.event_dialog.start.value()
            redn_state.stream_end = self.event_dialog.stop.value()
            redn_state.stream_duration = self.event_dialog.duration.value()

        # a progress dialog to show that reduction is occurring
        progress = QtWidgets.QProgressDialog(
            "Reducing files...", "Cancel", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.show()
        progress.raise_()
        progress.setValue(0)

        def callback(percent):
            progress.setValue(percent)
            if progress.wasCanceled():
                return False
            return True

        # the ReductionState object does the reduction
        redn_state.reducer(callback=callback)

        progress.setValue(100)

    @QtCore.Slot()
    def on_reducer_variables_clicked(self):
        """
        Present a dialogue to the user to change reduction options
        """
        ok = self.reduction_options_dialog.exec()
        if not ok:
            return

        # need to set state from reduction_options_dialog
        rod = self.reduction_options_dialog
        state = self._reduction_state

        for key, val in state.default_reduction_options.items():
            if val[1] is bool:
                value = getattr(rod, key).isChecked()
            elif val[1] is float:
                value = getattr(rod, key).value()
            setattr(self._reduction_state, key, value)

    @QtCore.Slot()
    def on_change_data_directory_clicked(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self)
        if directory:
            self.ui.data_directory.setText(directory)
            self._reduction_state.data_directory = directory

    @QtCore.Slot()
    def on_data_directory_editingFinished(self):
        directory = self.ui.data_directory.text()
        if os.path.isdir(directory):
            self._reduction_state.data_directory = directory
        else:
            self.ui.data_directory.setText(
                self._reduction_state.data_directory
            )

    @QtCore.Slot()
    def on_change_streamed_directory_clicked(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self)
        if directory:
            self.ui.streamed_directory.setText(directory)
            self._reduction_state.streamed_directory = directory

    @QtCore.Slot()
    def on_streamed_directory_editingFinished(self):
        directory = self.ui.streamed_directory.text()
        if os.path.isdir(directory):
            self._reduction_state.streamed_directory = directory
        else:
            self.ui.streamed_directory.setText(
                self._reduction_state.streamed_directory
            )

    @QtCore.Slot()
    def on_change_output_directory_clicked(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self)
        if directory:
            self.ui.output_directory.setText(directory)
            self._reduction_state.output_directory = directory

    @QtCore.Slot()
    def on_output_directory_editingFinished(self):
        directory = self.ui.output_directory.text()
        if os.path.isdir(directory):
            self._reduction_state.output_directory = directory
        else:
            self.output_directory.setText(
                self._reduction_state.output_directory
            )

    @QtCore.Slot()
    def on_actionSave_triggered(self):
        path = self._reduction_state.save_state_path
        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(self._reduction_state, f)
        else:
            self.on_actionSave_As_triggered()

    @QtCore.Slot()
    def on_actionSave_As_triggered(self):
        path = self._reduction_state.save_state_path
        initial_dir = "~/"
        if path is not None and os.path.isfile(path):
            initial_dir = os.path.dirname(path)

        fpath = getsavefilename(
            basedir=initial_dir, filters="slim files (*.slim)"
        )

        if fpath[0]:
            root, ext = os.path.splitext(fpath[0])
            fpath = root + ".slim"
            self._reduction_state.save_state_path = fpath
            self.on_actionSave_triggered()

    @QtCore.Slot()
    def on_actionLoad_triggered(self):
        path = self._reduction_state.save_state_path
        initial_dir = "~/"
        if path is not None and os.path.isfile(path):
            initial_dir = os.path.dirname(path)

        fpath = getopenfilename(
            self, basedir=initial_dir, filters="slim files (*.slim)"
        )

        if os.path.isfile(fpath[0]):
            with open(fpath[0], "rb") as f:
                try:
                    state = pickle.load(f)
                    if isinstance(state, ReductionState):
                        state.save_state_path = fpath[0]
                        self.set_state(state)
                # problem unpickling, or file had zero size.
                except (pickle.UnpicklingError, EOFError) as e:
                    logging.info(repr(e))

    @QtCore.Slot()
    def on_plot_clicked(self):
        self._plot.show()


class EventDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi(os.path.join(UI_LOCATION, "event.ui"), self)


class ReductionOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi(
            os.path.join(UI_LOCATION, "reduction_options.ui"), self
        )
