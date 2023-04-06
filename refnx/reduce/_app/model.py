import os.path
import os
import logging
from copy import copy

from qtpy import QtCore
from qtpy.QtCore import Qt
import numpy as np

from refnx._lib import preserve_cwd
from refnx.dataset import ReflectDataset
from refnx.reduce import (
    number_datafile,
    PlatypusReduce,
    basename_datafile,
    PlatypusNexus,
)


reducer_entry = [
    ("use", False),
    ("scale", 1.0),
    ("reflect-1", ""),
    ("reflect-2", ""),
    ("reflect-3", ""),
    ("direct-1", ""),
    ("direct-2", ""),
    ("direct-3", ""),
    ("flood", ""),
]
default_reducer_entry = {re[0]: re[1] for re in reducer_entry}


class ReductionState:
    """
    Reduces multiple reflectometry datafiles. Attributes control how the
    reduction proceeds.
    """

    def __init__(self, manual_beam_finder=None):
        super().__init__()

        # Each entry specifies the scale factor, whether the file is used for
        # reduction and the reflected and direct beam runs
        self.reduction_entries = {}

        self.default_reduction_options = {
            "low_wavelength": (2.5, float),
            "high_wavelength": (19, float),
            "rebin_percent": (2.0, float),
            "expected_centre": (500.0, float),
            "manual_beam_find": (False, bool),
            "background_subtraction": (True, bool),
            "monitor_normalisation": (True, bool),
            "save_offspecular": (True, bool),
            "save_spectrum": (True, bool),
            "streamed_reduction": (False, bool),
        }

        # the time slices for streamed reduction
        self.stream_start = 0
        self.stream_end = 3600
        self.stream_duration = 30

        # where the data, streamed data and output files are located
        self.data_directory = ""
        self.streamed_directory = ""
        self.output_directory = ""

        self.manual_beam_finder = manual_beam_finder

        for attr, val in self.default_reduction_options.items():
            setattr(self, attr, val[0])

        self.peak_pos = (90, 5)

        self.save_state_path = None

    def __getstate__(self):
        d = self.__dict__
        e = copy(d)
        e.pop("manual_beam_finder")
        return e

    @preserve_cwd
    def reducer(self, callback=None):
        """
        Reduce all the entries in reduction_entries

        Parameters
        ----------
        callback : callable
            Function, `f(percent_finished)` that is called with the current
            percentage progress of the reduction
        """

        # refnx.reduce.reduce needs you to be in the directory where you're
        # going to write files to
        if self.output_directory:
            os.chdir(self.output_directory)

        # if no data directory was specified then assume it's the cwd
        data_directory = self.data_directory
        if not data_directory:
            data_directory = "./"

        def full_path(fname):
            f = os.path.join(data_directory, fname)
            return f

        # if the streamed directory isn't mentioned then assume it's the same
        # as the data directory
        streamed_directory = self.streamed_directory
        if not os.path.isdir(streamed_directory):
            self.streamed_directory = data_directory

        logging.info(
            "-------------------------------------------------------"
            "\nStarting reduction run"
        )
        logging.info(
            "data_folder={data_directory}, trim_trailing=True, "
            "lo_wavelength={low_wavelength}, "
            "hi_wavelength={high_wavelength}, "
            "rebin_percent={rebin_percent}, "
            "normalise={monitor_normalisation}, "
            "background={background_subtraction} "
            "eventmode={streamed_reduction} "
            "event_folder={streamed_directory}".format(**self.__dict__)
        )

        # sets up time slices for event reduction
        if self.streamed_reduction:
            eventmode = np.arange(
                self.stream_start, self.stream_end, self.stream_duration
            )
            eventmode = np.r_[eventmode, self.stream_end]
        else:
            eventmode = None

        # are you manual beamfinding?
        peak_pos = None
        if self.manual_beam_find and self.manual_beam_finder is not None:
            peak_pos = -1

        idx = 0

        cached_direct_beams = {}

        for row, val in self.reduction_entries.items():
            if not val["use"]:
                continue

            flood = None
            if val["flood"]:
                flood = full_path(val["flood"])

            combined_dataset = None

            # process entries one by one
            for ref, db in zip(
                ["reflect-1", "reflect-2", "reflect-3"],
                ["direct-1", "direct-2", "direct-3"],
            ):
                reflect = val[ref]
                direct = val[db]

                # if the file doesn't exist there's no point continuing
                if (not os.path.isfile(full_path(reflect))) or (
                    not os.path.isfile(full_path(direct))
                ):
                    continue

                # which of the nspectra to reduce (or all)
                ref_pn = PlatypusNexus(full_path(reflect))

                if direct not in cached_direct_beams:
                    cached_direct_beams[direct] = PlatypusReduce(
                        direct, data_folder=data_directory
                    )

                reducer = cached_direct_beams[direct]

                try:
                    reduced = reducer(
                        ref_pn,
                        scale=val["scale"],
                        h5norm=flood,
                        lo_wavelength=self.low_wavelength,
                        hi_wavelength=self.high_wavelength,
                        rebin_percent=self.rebin_percent,
                        normalise=self.monitor_normalisation,
                        background=self.background_subtraction,
                        manual_beam_find=self.manual_beam_finder,
                        peak_pos=peak_pos,
                        eventmode=eventmode,
                        event_folder=streamed_directory,
                    )
                except Exception as e:
                    # typical Exception would be ValueError for non overlapping
                    # angles
                    logging.info(e)
                    continue

                logging.info(
                    "Reduced {} vs {}, scale={}, angle={}".format(
                        reflect,
                        direct,
                        val["scale"],
                        reduced[1]["omega"][0, 0],
                    )
                )

                if combined_dataset is None:
                    combined_dataset = ReflectDataset()

                    fname = basename_datafile(reflect)
                    fname_dat = os.path.join(
                        self.output_directory, "c_{0}.dat".format(fname)
                    )
                    fname_xml = os.path.join(
                        self.output_directory, "c_{0}.xml".format(fname)
                    )

                try:
                    combined_dataset.add_data(
                        reducer.data(),
                        requires_splice=True,
                        trim_trailing=True,
                    )
                except ValueError as e:
                    # datasets don't overlap
                    logging.info(e)
                    continue

            if combined_dataset is not None:
                # after you've finished reducing write a combined file.
                with open(fname_dat, "wb") as f:
                    combined_dataset.save(f)
                with open(fname_xml, "wb") as f:
                    combined_dataset.save_xml(f)
                logging.info(
                    "Written combined files: {} and {}".format(
                        fname_dat, fname_xml
                    )
                )

            # can be used to create a progress bar
            idx += 1
            if callback is not None:
                ok = callback(100 * idx / len(self.reduction_entries))
                if not ok:
                    break

        logging.info(
            "\nFinished reduction run"
            "-------------------------------------------------------"
        )


class ReductionTableModel(QtCore.QAbstractTableModel):
    """
    a model for displaying in a QtGui.QTableView
    """

    def __init__(self, reduction_state, parent=None):
        super().__init__(parent)
        self._reduction_state = reduction_state

    @property
    def reduction_state(self):
        return self._reduction_state

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 200

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(reducer_entry)

    def headerData(
        self, section, orientation, role=Qt.ItemDataRole.DisplayRole
    ):
        """Set the headers to be displayed."""
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Vertical:
            return None

        if orientation == Qt.Orientation.Horizontal:
            return reducer_entry[section][0]

        return None

    def flags(self, index):
        # row = index.row()
        col = index.column()
        if not col:
            return Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled

        return (
            Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
        )

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return False

        row = index.row()
        col = index.column()

        state = self.reduction_state
        attr_name = reducer_entry[col][0]

        if row in state.reduction_entries:
            entry = state.reduction_entries[row]
        else:
            entry = default_reducer_entry

        value = entry[attr_name]

        if role == Qt.ItemDataRole.CheckStateRole:
            if not col and value:
                return Qt.CheckState.Checked
            elif not col:
                return Qt.CheckState.Unchecked

        if role == Qt.ItemDataRole.DisplayRole and col:
            if attr_name == "scale":
                return str(float(value))
            else:
                return value

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        row = index.row()
        col = index.column()

        if not index.isValid():
            return False

        state = self.reduction_state
        attr_name = reducer_entry[col][0]

        if row in state.reduction_entries:
            entry = state.reduction_entries[row]
        else:
            entry = {re[0]: re[1] for re in reducer_entry}
            state.reduction_entries[row] = entry

        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            entry["use"] = value == Qt.CheckState.Checked

        if role == Qt.ItemDataRole.EditRole:
            if col == 0:
                save_value = False
                if value == Qt.CheckState.Checked:
                    save_value = True
            elif col == 1:
                try:
                    save_value = float(value)
                except ValueError:
                    save_value = 1
            else:
                # you're editing reflect/direct beam names
                try:
                    save_value = int(value)
                    save_value = number_datafile(save_value)
                except ValueError:
                    if value:
                        if value.endswith(".nx.hdf"):
                            save_value = value
                        else:
                            save_value = value + ".nx.hdf"
                    else:
                        return False

            entry[attr_name] = save_value

        self.dataChanged.emit(index, index)
        return True
