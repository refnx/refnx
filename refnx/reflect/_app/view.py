import os.path
from copy import deepcopy
import pickle
import os
import sys
import time

import numpy as np
import matplotlib

from PyQt5 import QtCore, QtGui, QtWidgets, uic

# matplotlib.rcParams['backend.qt5'] = 'PyQt5'
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib.artist as artist
import matplotlib.lines as lines


from .SLD_calculator_view import SLDcalculatorView
from .datastore import DataStore
from .treeview_gui_model import (TreeModel, Node, DatasetNode, DataObjectNode,
                                 ComponentNode, StructureNode,
                                 ReflectModelNode, ParNode, TreeFilter,
                                 find_data_object)


import refnx
from refnx.analysis import (CurveFitter, Objective,
                            Transform, GlobalObjective)
from refnx.reflect import SLD, ReflectModel
from refnx.reflect._code_fragment import code_fragment
from refnx._lib import unique, flatten


matplotlib.use('Qt5Agg')
UI_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ui')


class MotofitMainWindow(QtWidgets.QMainWindow):
    """
    Main View window for Motofit
    """

    def __init__(self, parent=None):
        super(MotofitMainWindow, self).__init__(parent)

        # load the GUI from the ui file
        self.ui = uic.loadUi(os.path.join(UI_LOCATION, 'motofit.ui'),
                             self)

        self.error_handler = QtWidgets.QErrorMessage()

        # redirect stdout to a console window
        console = EmittingStream()
        sys.stdout = console
        console.textWritten.connect(self.writeTextToConsole)

        #######################################################################
        # Everything ending in 'model' refers to a QtAbstract<x>Model.  These
        # are the basis for GUI elements.  They also contain data.
        data_container = DataStore()

        # a flag that is used by update_gui_model to decide whether to redraw
        # object graphs in response to the treeModel being changed
        self._hold_updating = False

        # set up tree view
        self.treeModel = TreeModel(data_container)
        # self.ui.treeView.setModel(self.treeModel)
        # the filter controls what rows are presented in the treeView
        self.treeFilter = TreeFilter(self.treeModel)
        self.treeFilter.setSourceModel(self.treeModel)
        self.ui.treeView.setModel(self.treeFilter)

        self.ui.treeView.setColumnWidth(0, 260)

        self.ui.treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        # context menu for the treeView
        self.context_menu = OpenMenu(self.ui.treeView)

        self.ui.treeView.customContextMenuRequested.connect(self.context_menu)
        self.context_menu.add_to_fit_action.triggered.connect(
            self.on_add_to_fit_button_clicked)
        self.context_menu.remove_from_fit_action.triggered.connect(
            self.on_remove_from_fit_action)
        self.context_menu.link_action.triggered.connect(
            self.link_action)
        self.context_menu.unlink_action.triggered.connect(
            self.unlink_action)
        self.context_menu.copy_from_action.triggered.connect(
            self.copy_from_action)

        self.treeModel.dataChanged.connect(self.tree_model_data_changed)
        self.treeModel.rowsRemoved.connect(self.tree_model_structure_changed)
        self.treeModel.rowsInserted.connect(self.tree_model_structure_changed)
        #######################################################################

        # attach the reflectivity graphs and the SLD profiles
        self.modify_gui()

        # holds miscellaneous information on program settings
        self.settings = ProgramSettings()
        self.settings.current_dataset_name = 'theoretical'

        theoretical = data_container['theoretical']
        model = theoretical.model
        dataset = theoretical.dataset
        resolution = self.settings.resolution / 100.
        transform = Transform(self.settings.transformdata)
        fit = model(dataset.x, x_err=dataset.x * resolution)
        fit, _ = transform(dataset.x, fit)
        sld = model.structure.sld_profile()

        graph_properties = theoretical.graph_properties
        graph_properties['ax_fit'] = self.reflectivitygraphs.axes[0].plot(
            dataset.x,
            fit, color='r',
            linestyle='-', lw=1,
            label='theoretical',
            picker=5)[0]

        graph_properties['ax_sld_profile'] = self.sldgraphs.axes[0].plot(
            sld[0],
            sld[1],
            linestyle='-', color='r')[0]

        self.restore_settings()

        print('Session started at:', time.asctime(time.localtime(time.time())))

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    @QtCore.pyqtSlot(QtGui.QDropEvent)
    def dropEvent(self, event):
        m = event.mimeData()
        urls = m.urls()

        # convert url's to files
        urls_as_files = [url.toLocalFile() for url in urls]

        # if you've only got one url then try and load it as an experiment
        if len(urls_as_files) == 1:
            try:
                self._restore_state(urls_as_files[0])
                return
            except Exception:
                pass

        # then try and load urls as data files
        try:
            loaded_names = self.load_data(urls_as_files)
        except Exception:
            loaded_names = []

        # then try and load the remainder as models
        remainder_urls = set(urls_as_files).difference(set(loaded_names))
        for url in remainder_urls:
            try:
                self.load_model(url)
                continue
            except Exception:
                pass

    @QtCore.pyqtSlot(QtGui.QDragEnterEvent)
    def dragEnterEvent(self, event):
        m = event.mimeData()
        if m.hasUrls():
            event.acceptProposedAction()

    def writeTextToConsole(self, text):
        self.ui.console_text_edit.moveCursor(QtGui.QTextCursor.End)
        self.ui.console_text_edit.insertPlainText(text)

    def _saveState(self, experiment_file_name):
        state = {}
        self.settings.experiment_file_name = experiment_file_name
        state['datastore'] = self.treeModel.datastore
        state['history'] = self.ui.console_text_edit.toPlainText()
        state['settings'] = self.settings

        widget = self.ui.currently_fitting
        state['currently_fitting'] = [widget.item(i).text() for i in
                                      range(widget.count())]

        with open(os.path.join(experiment_file_name), 'wb') as f:
            pickle.dump(state, f, 0)

        self.setWindowTitle('Motofit - ' + experiment_file_name)

    @QtCore.pyqtSlot()
    def on_actionSave_File_triggered(self):
        if os.path.isfile(self.settings.experiment_file_name):
            self._saveState(self.settings.experiment_file_name)
        else:
            self.on_actionSave_File_As_triggered()

    @QtCore.pyqtSlot()
    def on_actionSave_File_As_triggered(self):
        experiment_file_name, ok = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save experiment as:', 'experiment.mtft')

        if not ok:
            return

        path, ext = os.path.splitext(experiment_file_name)
        if ext != '.mtft':
            experiment_file_name = path + '.mtft'

        self._saveState(experiment_file_name)

    def _restore_state(self, experiment_file_name):
        with open(experiment_file_name, 'rb') as f:
            state = pickle.load(f)

        if not state:
            print("Couldn't load experiment")
            return

        try:
            self.treeModel._data = state[
                'datastore']
            self.treeModel.rebuild()
            self.ui.console_text_edit.setPlainText(state['history'])
            self.settings = state['settings']
            self.settings.experiment_file_name = experiment_file_name

            widget = self.ui.currently_fitting
            for i in reversed(range(widget.count())):
                widget.takeItem(i)
            for name in state['currently_fitting']:
                widget.addItem(name)

        except KeyError as e:
            print(type(e), e.message)
            return

        self.restore_settings()

        # remove and add datasetsToGraphs
        self.reflectivitygraphs.remove_traces()
        self.sldgraphs.remove_traces()
        ds = [d for d in self.treeModel.datastore]
        self.add_data_objects_to_graphs(ds)

        # when you load in the theoretical model you destroy the link to the
        # gui, reinstate it.
        self.treeModel.rebuild()

        self.reflectivitygraphs.draw()

    def restore_settings(self):
        """
        applies the program settings to the GUI
        """
        title = 'Motofit'
        if len(self.settings.experiment_file_name):
            title += ' - ' + self.settings.experiment_file_name
        self.setWindowTitle(title)

        self.ui.use_errors_checkbox.setChecked(self.settings.useerrors)
        self.ui.actionLevenberg_Marquardt.setChecked(False)
        self.ui.actionDifferential_Evolution.setChecked(False)
        if self.settings.fitting_algorithm == 'LM':
            self.ui.actionLevenberg_Marquardt.setChecked(True)
        elif self.settings.fitting_algorithm == 'DE':
            self.ui.actionDifferential_Evolution.setChecked(True)
        elif self.settings.fitting_algorithm == 'MCMC':
            self.ui.actionMCMC.setChecked(True)
        elif self.settings.fitting_algorithm == 'L-BFGS-B':
            self.ui.actionL_BFGS_B.setChecked(True)

        self.settransformoption(self.settings.transformdata)

    def apply_settings_to_params(self, params):
        for key in self.settings.__dict__:
            params[key] = self.settings[key]

    @QtCore.pyqtSlot()
    def on_actionLoad_File_triggered(self):
        experimentFileName, ok = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption='Select Experiment File',
            filter='Experiment Files (*.mtft)')
        if not ok:
            return

        self._restore_state(experimentFileName)

    def load_data(self, files):
        """
        Loads datasets into the app

        Parameters
        ----------
        files : sequence
            Sequence of `str` specifying filenames to load the datasets from

        Returns
        -------
        fnames : sequence
            Sequence of `str` containing the filenames of the successfully
            loaded datasets.
        """
        datastore = self.treeModel.datastore

        existing_data_objects = set(datastore.names)

        data_objects = []
        fnames = []
        for file in files:
            if os.path.isfile(file):
                try:
                    data_object = self.treeModel.load_data(file)
                    if data_object is not None:
                        data_objects.append(data_object)
                        fnames.append(file)
                except RuntimeError:
                    continue

        # new data_object_names
        loaded_data_objects = set([data_object.name for data_object
                                   in data_objects])

        # give a newly loaded data object a simple model. You don't want to do
        # this for an object that's already been loaded.
        for name in loaded_data_objects:
            fronting = SLD(0, name='fronting')
            sio2 = SLD(3.47, name='1')
            backing = SLD(2.07, name='backing')
            s = fronting() | sio2(15, 3) | backing(0, 3)
            data_object_node = self.treeModel.data_object_node(name)
            model = ReflectModel(s)
            model.name = name
            data_object_node.set_reflect_model(model)

        # for the intersection of loaded and old, refresh the plot.
        refresh_names = existing_data_objects.intersection(loaded_data_objects)
        refresh_data_objects = [datastore[name] for name
                                in refresh_names]
        self.redraw_data_object_graphs(refresh_data_objects)

        # for totally new, then add to graphs
        new_names = loaded_data_objects.difference(existing_data_objects)
        new_data_objects = [datastore[name] for name in new_names]
        self.add_data_objects_to_graphs(new_data_objects)

        self.calculate_chi2(data_objects)
        return fnames

    @QtCore.pyqtSlot()
    def on_actionLoad_Data_triggered(self):
        """
        you load data
        """
        files = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            caption='Select Reflectivity Files')

        if files:
            self.load_data(files[0])

    @QtCore.pyqtSlot()
    def on_actionRemove_Data_triggered(self):
        """
        you remove data
        """
        datastore = self.treeModel.datastore
        datasets = list(datastore.names)
        del(datasets[datasets.index('theoretical')])
        if not len(datasets):
            return

        which_dataset, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Which dataset did you want to remove?",
            "dataset",
            datasets,
            editable=False)

        if not ok:
            return

        # remove from list of datasets to be fitted, if present
        widget = self.ui.currently_fitting
        items_found = widget.findItems(which_dataset, QtCore.Qt.MatchExactly)
        for item in items_found:
            row = widget.row(item)
            widget.takeItem(row)

        self.reflectivitygraphs.remove_trace(
            datastore[which_dataset])
        self.treeModel.remove_data_object(which_dataset)

    @QtCore.pyqtSlot()
    def on_actionSave_Fit_triggered(self):
        datastore = self.treeModel.datastore
        fits = datastore.names
        fits.append('-all-')

        which_fit, ok = QtWidgets.QInputDialog.getItem(
            self, "Which fit did you want to save?", "", fits, editable=False)
        if not ok:
            return

        if which_fit == '-all-':
            dialog = QtWidgets.QFileDialog(self)
            dialog.setFileMode(QtWidgets.QFileDialog.Directory)
            if dialog.exec_():
                folder = dialog.selectedFiles()
                fits.pop()
                for fit in fits:
                    datastore[fit].save_fit(
                        os.path.join(folder[0], 'fit_' + fit + '.dat'))
        else:
            fitFileName, ok = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save fit as:', 'fit_' + which_fit + '.dat')
            if not ok:
                return
            datastore[which_fit].save_fit(fitFileName)

    def load_model(self, model_file_name):
        with open(model_file_name, 'rb') as f:
            model = pickle.load(f)

        data_object_node = self.treeModel.data_object_node(model.name)
        if data_object_node is not None:
            data_object_node.set_reflect_model(model)
        else:
            # there is not dataset with that name, put the model over the
            # theoretical one
            data_object_node = self.treeModel.data_object_node('theoretical')
            data_object_node.set_reflect_model(model)
        self.update_gui_model([data_object_node.data_object])

    @QtCore.pyqtSlot()
    def on_actionLoad_Model_triggered(self):
        # load a model
        model_file_name, ok = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Model File')
        if not ok:
            return
        self.load_model(model_file_name)

    @QtCore.pyqtSlot()
    def on_actionSave_Model_triggered(self):
        # save a model
        # which model are you saving?
        datastore = self.treeModel.datastore

        # all data_objects have a model, it's just that they're
        # not all developed.
        model_names = datastore.names
        model_names.append('-all-')

        which_model, ok = QtWidgets.QInputDialog.getItem(
            self, "Which model did you want to save?",
            "model",
            model_names,
            editable=False)

        if not ok:
            return

        if which_model == '-all-':
            dialog = QtWidgets.QFileDialog(self)
            dialog.setFileMode(QtWidgets.QFileDialog.Directory)
            if dialog.exec_():
                folder = dialog.selectedFiles()

                for model_name in datastore.names:
                    model = datastore[model_name]
                    fname = os.path.join(folder[0],
                                         'coef_' + model_name + '.pkl')
                    with open(fname, 'wb') as f:
                        pickle.dump(model, f)
        else:
            suggested_name = os.path.join(os.getcwd(),
                                          'coef_' + which_model + '.pkl')
            modelFileName, ok = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Save model as:',
                suggested_name)
            if not ok:
                return

            # retrieve the data_object
            data_object = datastore[which_model]
            with open(modelFileName, 'wb') as f:
                pickle.dump(data_object.model, f)

    @QtCore.pyqtSlot()
    def on_actionExport_Code_Fragment_triggered(self):
        widget = self.ui.currently_fitting
        rows = widget.count()
        names_to_fit = [widget.item(i).text() for i in range(rows)]

        if not names_to_fit:
            return

        # retrieve data_objects
        datastore = self.treeModel.datastore
        data_objects = [datastore[name] for name in names_to_fit]

        objective = self.create_objective(data_objects)
        code = code_fragment(objective)

        suggested_name = os.path.join(os.getcwd(), 'code.py')
        modelFileName, ok = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save code fragment as:',
            suggested_name)
        if not ok:
            return
        with open(suggested_name, 'w') as f:
            f.write(code)

    @QtCore.pyqtSlot()
    def on_actionDifferential_Evolution_triggered(self):
        if self.ui.actionDifferential_Evolution.isChecked():
            self.settings.fitting_algorithm = 'DE'
            self.ui.actionLevenberg_Marquardt.setChecked(False)
            self.ui.actionMCMC.setChecked(False)
            self.ui.actionL_BFGS_B.setChecked(False)

    @QtCore.pyqtSlot()
    def on_actionMCMC_triggered(self):
        if self.ui.actionMCMC.isChecked():
            self.settings.fitting_algorithm = 'MCMC'
            self.ui.actionLevenberg_Marquardt.setChecked(False)
            self.ui.actionDifferential_Evolution.setChecked(False)
            self.ui.actionL_BFGS_B.setChecked(False)

    @QtCore.pyqtSlot()
    def on_actionLevenberg_Marquardt_triggered(self):
        if self.ui.actionLevenberg_Marquardt.isChecked():
            self.settings.fitting_algorithm = 'LM'
            self.ui.actionDifferential_Evolution.setChecked(False)
            self.ui.actionMCMC.setChecked(False)
            self.ui.actionL_BFGS_B.setChecked(False)

    @QtCore.pyqtSlot()
    def on_actionL_BFGS_B_triggered(self):
        if self.ui.actionL_BFGS_B.isChecked():
            self.settings.fitting_algorithm = 'L-BFGS-B'
            self.ui.actionDifferential_Evolution.setChecked(False)
            self.ui.actionMCMC.setChecked(False)
            self.ui.actionLevenberg_Marquardt.setChecked(False)

    def change_Q_range(self, qmin, qmax, numpnts):
        data_object_node = self.treeModel.data_object_node('theoretical')

        theoretical = data_object_node._data
        dataset = theoretical.dataset

        new_x = np.linspace(qmin, qmax, numpnts)
        new_y = np.zeros_like(new_x) * np.nan
        dataset.data = (new_x, new_y)
        data_object_node.set_dataset(dataset)
        self.update_gui_model([theoretical])

    @QtCore.pyqtSlot()
    def on_actionChange_Q_range_triggered(self):
        datastore = self.treeModel.datastore
        theoretical = datastore['theoretical']
        qmin = min(theoretical.dataset.x)
        qmax = max(theoretical.dataset.x)
        numpnts = len(theoretical.dataset)

        dvalidator = QtGui.QDoubleValidator(-2.0e-308, 2.0e308, 6)

        qrangeGUI = uic.loadUi(os.path.join(UI_LOCATION, 'qrangedialog.ui'))
        qrangeGUI.numpnts.setValue(numpnts)
        qrangeGUI.qmin.setValidator(dvalidator)
        qrangeGUI.qmax.setValidator(dvalidator)
        qrangeGUI.qmin.setText(str(qmin))
        qrangeGUI.qmax.setText(str(qmax))

        ok = qrangeGUI.exec_()
        if ok:
            self.change_Q_range(float(qrangeGUI.qmin.text()),
                                float(qrangeGUI.qmax.text()),
                                qrangeGUI.numpnts.value())

    @QtCore.pyqtSlot()
    def on_actionTake_Snapshot_triggered(self):
        snapshotname, ok = QtWidgets.QInputDialog.getText(
            self,
            'Take a snapshot',
            'snapshot name')
        if not ok:
            return

        # is the snapshot already present?
        datastore = self.treeModel.datastore
        snapshot_exists = datastore[snapshotname]
        data_object = self.treeModel.snapshot(snapshotname)

        if snapshot_exists is not None:
            self.redraw_data_object_graphs([data_object])
        else:
            self.add_data_objects_to_graphs([data_object])

    @QtCore.pyqtSlot()
    def on_actionResolution_smearing_triggered(self):
        currentVal = self.settings.quad_order
        value, ok = QtWidgets.QInputDialog.getInt(
            self,
            'Resolution Smearing',
            'Number of points for Gaussian Quadrature',
            currentVal,
            17)
        if not ok:
            return
        self.settings.quad_order = value
        self.update_gui_model([])

    @QtCore.pyqtSlot()
    def on_actionBatch_Fit_triggered(self):
        datastore = self.treeModel.datastore
        if len(datastore) < 2:
            return msg("You have no loaded datasets")

        theoretical = datastore['theoretical']
        # iterate and fit over all the datasets, but first copy the model from
        # the theoretical model because it's unlikely you're going to setup all
        # the individual models first.
        for data_object in datastore:
            if data_object.name == 'theoretical':
                continue
            name = data_object.name
            new_model = deepcopy(theoretical.model)
            new_model.name = name
            data_object_node = self.treeModel.data_object_node(name)
            data_object_node.set_reflect_model(new_model)
            self.do_a_fit_and_add_to_gui([data_object])

    @QtCore.pyqtSlot()
    def on_actionRefresh_Data_triggered(self):
        """
            you are refreshing existing datasets
        """
        self.treeModel.refresh()
        self.redraw_data_object_graphs(None, all=True)

    @QtCore.pyqtSlot()
    def on_actionlogY_vs_X_triggered(self):
        self.settransformoption('logY')

    @QtCore.pyqtSlot()
    def on_actionY_vs_X_triggered(self):
        self.settransformoption('lin')

    @QtCore.pyqtSlot()
    def on_actionYX4_vs_X_triggered(self):
        self.settransformoption('YX4')

    @QtCore.pyqtSlot()
    def on_actionYX2_vs_X_triggered(self):
        self.settransformoption('YX2')

    def settransformoption(self, transform):
        self.ui.actionlogY_vs_X.setChecked(False)
        self.ui.actionY_vs_X.setChecked(False)
        self.ui.actionYX4_vs_X.setChecked(False)
        self.ui.actionYX2_vs_X.setChecked(False)
        if transform is None:
            self.ui.actionY_vs_X.setChecked(True)
            transform = 'lin'
        if transform == 'lin':
            self.ui.actionY_vs_X.setChecked(True)
        elif transform == 'logY':
            self.ui.actionlogY_vs_X.setChecked(True)
        elif transform == 'YX4':
            self.ui.actionYX4_vs_X.setChecked(True)
        elif transform == 'YX2':
            self.ui.actionYX2_vs_X.setChecked(True)
        self.settings.transformdata = transform

        self.redraw_data_object_graphs(None, all=True)

        # need to relimit graphs and display on a log scale if the transform
        # has changed
        self.reflectivitygraphs.axes[0].autoscale(axis='both',
                                                  tight=False,
                                                  enable=True)
        self.reflectivitygraphs.axes[0].relim()
        if transform in ['lin', 'YX2']:
            self.reflectivitygraphs.axes[0].set_yscale('log')
        else:
            self.reflectivitygraphs.axes[0].set_yscale('linear')
        self.reflectivitygraphs.draw()

    @QtCore.pyqtSlot()
    def on_actionAbout_triggered(self):
        aboutui = uic.loadUi(os.path.join(UI_LOCATION, 'about.ui'))

        licence_dir = os.path.join(UI_LOCATION, 'licences')
        licences = os.listdir(licence_dir)
        licences.remove('about')

        text = [refnx.version.version]
        with open(os.path.join(licence_dir, 'about'), 'r',
                  encoding='utf-8', errors='replace') as f:
            text.append(''.join(f.readlines()))

        for licence in licences:
            fname = os.path.join(licence_dir, licence)
            with open(fname, 'r', encoding='utf-8', errors='replace') as f:
                text.append(''.join(f.readlines()))

        display_text = '\n_______________________________________\n'.join(text)
        aboutui.textBrowser.setText(display_text)
        aboutui.exec_()

    @QtCore.pyqtSlot()
    def on_actiondocumentation_triggered(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(
            'https://refnx.readthedocs.io/en/latest/'))

    @QtCore.pyqtSlot()
    def on_actionAutoscale_graph_triggered(self):
        self.reflectivitygraphs.autoscale()

    @QtCore.pyqtSlot()
    def on_actionSLD_calculator_triggered(self):
        SLDcalculator = SLDcalculatorView(self)
        SLDcalculator.show()

    @QtCore.pyqtSlot()
    def on_add_layer_clicked(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        if not selected_indices:
            return msg('Select a single row within a Structure to insert a'
                       ' new slab.')

        index = selected_indices[0]

        # from filter to model
        index = self.mapToSource(index)

        if not index.isValid():
            return

        item = index.internalPointer()
        hierarchy = item.hierarchy()
        # the row you selected was within the component list
        _component = [i for i in hierarchy if isinstance(i, ComponentNode)]
        if not _component:
            return msg('Select a single row within a Structure to insert a'
                       ' new slab.')

        # work out which component you have.
        component = _component[0]
        structure = component.parent()
        idx = component.row()
        if idx == len(structure._data) - 1:
            return msg("You can't append a layer after the backing medium,"
                       " select a previous layer")

        # all checking done, append a layer
        material = SLD(3.47)
        slab = material(15, 3)
        slab.name = 'slab'
        slab.thick.name = 'thick'
        slab.rough.name = 'rough'
        slab.sld.real.name = 'sld'
        slab.sld.imag.name = 'isld'
        slab.vfsolv.name = 'vfsolv'
        structure.insert_component(idx + 1, slab)

    @QtCore.pyqtSlot()
    def on_remove_layer_clicked(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        if not selected_indices:
            return msg('Select a single row within a Structure to remove.')

        index = selected_indices[0]
        if not index.isValid():
            return

        # from filter to model
        index = self.mapToSource(index)

        item = index.internalPointer()
        hierarchy = item.hierarchy()
        # the row you selected was within the component list
        _component = [i for i in hierarchy if isinstance(i, ComponentNode)]
        if not _component:
            return msg('Select a single row within a Structure to insert a'
                       ' new slab.')

        # work out which component you have.
        component = _component[0]
        structure = component.parent()
        idx = component.row()
        if idx in [0, len(structure._data) - 1]:
            return msg("You can't remove the fronting or backing media")

        # all checking done, remove a layer
        structure.remove_component(idx)

    @QtCore.pyqtSlot()
    def on_auto_limits_button_clicked(self):
        widget = self.ui.currently_fitting
        rows = widget.count()
        names_to_fit = [widget.item(i).text() for i in range(rows)]

        datastore = self.treeModel.datastore
        data_objects = [datastore[name] for name in names_to_fit]
        # auto set the limits for the theoretical model, because it's used as
        # a springboard for the batch fit.
        data_objects.append(datastore['theoretical'])

        for data_object in data_objects:
            # retrive data_object node from the treeModel
            node = self.treeModel.data_object_node(data_object.name)
            descendants = node.descendants()
            # filter out all the parameter nodes
            par_nodes = [n for n in descendants if isinstance(n, ParNode)]
            var_par_nodes = [n for n in par_nodes if n.parameter.vary]

            for par in var_par_nodes:
                p = par.parameter
                val = p.value
                bounds = p.bounds
                if val < 0:
                    bounds.ub = 0
                    bounds.lb = 2 * val
                else:
                    bounds.ub = 0
                    bounds.lb = 2 * val

                parent, row = par.parent(), par.row()
                idx1 = self.treeModel.index(row, 3, parent.index)
                idx2 = self.treeModel.index(row, 4, parent.index)
                self.treeModel.dataChanged.emit(idx1, idx2)

    @QtCore.pyqtSlot()
    def on_add_to_fit_button_clicked(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        widget = self.ui.currently_fitting

        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)

            data_object_node = find_data_object(index)
            if not data_object_node:
                continue
            # see if it's already in the list
            name = data_object_node.data_object.name
            if name == 'theoretical':
                continue

            find_names = widget.findItems(name, QtCore.Qt.MatchExactly)
            if not find_names:
                widget.addItem(name)

    @QtCore.pyqtSlot()
    def on_remove_from_fit_button_clicked(self):
        # work out what datasets are selected in the listwidget
        # remove all those that are selected
        widget = self.ui.currently_fitting
        selected_items = widget.selectedItems()
        for item in selected_items:
            row = widget.row(item)
            widget.takeItem(row)

    def on_remove_from_fit_action(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        widget = self.ui.currently_fitting

        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)

            data_object_node = find_data_object(index)
            if not data_object_node:
                continue
            # see if it's already in the list
            name = data_object_node.data_object.name
            list_items = widget.findItems(name, QtCore.Qt.MatchExactly)
            if list_items:
                row = widget.row(list_items[0])
                widget.takeItem(row)

    @QtCore.pyqtSlot()
    def on_do_fit_button_clicked(self):
        """
        you should do a fit
        """
        widget = self.ui.currently_fitting
        rows = widget.count()

        names_to_fit = [widget.item(i).text() for i in range(rows)]

        if not names_to_fit:
            return msg("Please add datasets to fit.")

        # retrieve data_objects
        datastore = self.treeModel.datastore
        data_objects = [datastore[name] for name in names_to_fit]
        self.do_a_fit_and_add_to_gui(data_objects)

    def create_objective(self, data_objects):
        # performs a global fit to the list of data_objects
        t = Transform(self.settings.transformdata)
        useerrors = self.settings.useerrors

        # TODO fix resolution
        # make objectives
        objectives = []
        for data_object in data_objects:
            objective = Objective(data_object.model,
                                  data_object.dataset,
                                  name=data_object.name,
                                  transform=t,
                                  use_weights=useerrors)
            objectives.append(objective)

        if len(objectives) == 1:
            objective = objectives[0]
        else:
            objective = GlobalObjective(objectives)

        return objective

    def do_a_fit_and_add_to_gui(self, data_objects):
        objective = self.create_objective(data_objects)

        alg = self.settings.fitting_algorithm

        # figure out how many varying parameters
        vp = objective.varying_parameters()
        if not vp:
            return msg("No parameters are being varied.")

        fitter = CurveFitter(objective)

        progress = ProgressCallback(self, objective=objective)
        progress.show()

        methods = {'DE': 'differential_evolution',
                   'LM': 'least_squares',
                   'L-BFGS-B': 'L-BFGS-B',
                   'MCMC': 'MCMC'}

        # least squares doesnt have a callback
        callback = progress.callback
        if alg == 'LM':
            callback = None

        if methods[alg] != 'MCMC':
            try:
                fitter.fit(method=methods[alg],
                           callback=callback)
                print(str(objective))
            except RuntimeError as e:
                # user probably aborted the fit
                # but it's still worth creating a fit curve, so don't return
                # in this catch block
                msg(repr(e))
                print(repr(e))
                print(objective)
            except ValueError as e:
                # Typically shown when sensible limits weren't provided
                msg(repr(e))
                progress.close()
                return None
        else:
            # TODO implement MCMC
            pass

        progress.close()

        # mark models as having been updated
        # prevent the GUI from updating whilst we change all the values
        self._hold_updating = True

        for data_object in data_objects:
            # retrive data_object node from the treeModel
            node = self.treeModel.data_object_node(data_object.name)
            descendants = node.descendants()
            # filter out all the parameter nodes
            par_nodes = [n for n in descendants if isinstance(n, ParNode)]
            var_par_nodes = [n for n in par_nodes if n.parameter.vary]

            for par in var_par_nodes:
                # for some reason the view doesn't know how to display float64
                # coerce to a float
                if par.parameter.stderr is not None:
                    par.parameter.stderr = float(par.parameter.stderr)
                    parent, row = par.parent(), par.row()
                    idx1 = self.treeModel.index(row, 1, parent.index)
                    idx2 = self.treeModel.index(row, 2, parent.index)
                    self.treeModel.dataChanged.emit(idx1, idx2)

        # re-enable the GUI updating whilst we change all the values
        self._hold_updating = False

        # update all the chi2 values
        self.calculate_chi2(data_objects)
        # plot the fit and sld_profile
        # TODO refactor both of the following into a single method?
        self.add_data_objects_to_graphs(data_objects)
        self.redraw_data_object_graphs(data_objects)

    @QtCore.pyqtSlot(int)
    def on_use_errors_checkbox_stateChanged(self, arg_1):
        """
        want to weight by error bars, recalculate chi2
        """

        if arg_1:
            use = True
        else:
            use = False

        self.settings.useerrors = use
        self.update_gui_model([])

    def mapToSource(self, index):
        # converts a model index in the filter model to the original model
        return self.treeFilter.mapToSource(index)

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_treeView_clicked(self, index):
        index = self.mapToSource(index)
        if not index.isValid():
            return None

        item = index.internalPointer()
        if not isinstance(item, ParNode):
            return

        self.currentCell = {}

        par = item.parameter
        val = par.value

        if val < 0:
            lowlim = 2 * val
            hilim = 0
        else:
            lowlim = 0
            hilim = 2 * val

        self.currentCell['item'] = item
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True

    @QtCore.pyqtSlot(int)
    def on_paramsSlider_valueChanged(self, arg_1):
        # short circuit if the treeview hasn't been clicked yet
        if not hasattr(self, 'currentCell'):
            return

        c = self.currentCell
        item = c['item']

        if not c['readyToChange']:
            return

        val = (c['lowlim'] +
               (arg_1 / 1000.) * np.fabs(c['lowlim'] - c['hilim']))

        item.parameter.value = val

        # get the index for the change
        row = item.index.row()
        # who is the parent
        parent_index = item.parent().index
        index = self.treeModel.index(row, 1, parent_index)
        self.treeModel.dataChanged.emit(index, index,
                                        [QtCore.Qt.EditRole])

    @QtCore.pyqtSlot()
    def on_paramsSlider_sliderReleased(self):
        try:
            self.currentCell['readyToChange'] = False
            self.ui.paramsSlider.setValue(499)
            item = self.currentCell['item']

            val = item.parameter.value

            if val < 0:
                low_lim = 2 * val
                hi_lim = 0
            else:
                low_lim = 0
                hi_lim = 2 * val

            self.currentCell['val'] = val
            self.currentCell['lowlim'] = low_lim
            self.currentCell['hilim'] = hi_lim
            self.currentCell['readyToChange'] = True

        except (ValueError, AttributeError, KeyError):
            return

        # for some reason linked parameters don't update well when the slider
        # moves. Setting focus to the treeview and back to the slider seems
        # to make the update happen. One could issue more dataChanged signals
        # from sliderValue changed, but I'd have to figure out all the other
        # nodes
        self.ui.treeView.setFocus(QtCore.Qt.OtherFocusReason)
        self.ui.paramsSlider.setFocus(QtCore.Qt.OtherFocusReason)

    def link_action(self):
        selected_indices = self.ui.treeView.selectedIndexes()
        par_nodes_to_link = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)
            item = index.internalPointer()
            if isinstance(item, ParNode):
                par_nodes_to_link.append(item)

        # get unique nodes
        par_nodes_to_link = list(unique(par_nodes_to_link))

        if len(par_nodes_to_link) < 2:
            return

        master_parameter = par_nodes_to_link[0]
        mp = master_parameter.parameter
        for par in par_nodes_to_link[1:]:
            par.parameter.constraint = mp
            idx = self.treeModel.index(par.row(), 1, par.parent().index)
            idx1 = self.treeModel.index(par.row(), 5, par.parent().index)
            self.treeModel.dataChanged.emit(idx, idx1)

        self.ui.paramsSlider.setFocus(QtCore.Qt.OtherFocusReason)
        self.ui.treeView.setFocus(QtCore.Qt.OtherFocusReason)

    def unlink_action(self):
        selected_indices = self.ui.treeView.selectedIndexes()
        par_nodes_to_unlink = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)
            item = index.internalPointer()
            if isinstance(item, ParNode):
                par_nodes_to_unlink.append(item)

        # get unique nodes
        par_nodes_to_unlink = list(unique(par_nodes_to_unlink))

        for par in par_nodes_to_unlink:
            par.parameter.constraint = None
            idx = self.treeModel.index(par.row(), 2, par.parent().index)
            idx1 = self.treeModel.index(par.row(), 5, par.parent().index)
            self.treeModel.dataChanged.emit(idx, idx1)

        # a trick to make the treeView repaint
        self.ui.paramsSlider.setFocus(QtCore.Qt.OtherFocusReason)
        self.ui.treeView.setFocus(QtCore.Qt.OtherFocusReason)

    def copy_from_action(self):
        # whose model did you want to use?
        datastore = self.treeModel.datastore

        model_names = datastore.names
        which_model, ok = QtWidgets.QInputDialog.getItem(
            self, "Which model did you want to copy?",
            "model",
            model_names,
            editable=False)

        if not ok:
            return

        # a trick to make the treeView repaint
        selected_indices = self.ui.treeView.selectedIndexes()
        source_model = datastore[which_model].model

        nodes = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)
            item = index.internalPointer()
            nodes.append(item.hierarchy())

        # filter out the data_object_nodes
        data_object_nodes = [n for n in flatten(nodes) if
                             isinstance(n, DataObjectNode)]

        # get unique data object nodes, these are the data objects whose
        # model you want to overwrite
        data_object_nodes = list(unique(data_object_nodes))

        # now set the model for all those data object nodes
        for don in data_object_nodes:
            new_model = deepcopy(source_model)
            new_model.name = don.data_object.name
            don.set_reflect_model(new_model)

        do = [node.data_object for node in data_object_nodes]
        self.update_gui_model(do)

    def modify_gui(self):
        """
        Only called at program initialisation. Used to add the plots
        """
        self.sldgraphs = MySLDGraphs(self.ui.sld)
        self.ui.gridLayout_4.addWidget(self.sldgraphs)

        self.reflectivitygraphs = MyReflectivityGraphs(self.ui.reflectivity)
        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs)

        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs.mpl_toolbar)
        self.ui.gridLayout_4.addWidget(self.sldgraphs.mpl_toolbar)

    def redraw_data_object_graphs(self, data_objects,
                                  all=False,
                                  transform=True):
        """ Asks the graphs to be redrawn, delegating generative
        calculation to reflectivitygraphs.redraw_data_objects

        Parameters
        ----------
        data_objects: list
            all the datasets to be redrawn
        all: bool
            redraw all the datasets
        transform: bool or curvefitter.Transform
            Transform the data in the dataset for visualisation.
            False - don't transform
            True - transform the data using the program default
            refnx.objective.Transform - use a custom transform
        """
        datastore = self.treeModel.datastore
        if all:
            data_objects = [data_object for data_object in datastore]

        if callable(transform):
            t = transform
        elif transform:
            t = Transform(self.settings.transformdata)
        else:
            t = None

        self.reflectivitygraphs.redraw_data_objects(data_objects,
                                                    transform=t)
        self.sldgraphs.redraw_data_objects(data_objects)

    def tree_model_structure_changed(self, parent, first, last):
        # called when you've removed or added layers to the model
        if not parent.isValid():
            return

        node = parent.internalPointer()

        # if you're not adding/removing Components don't respond.
        if not isinstance(node, StructureNode):
            return

        # find out which data_object / model we're adjusting
        hierarchy = node.hierarchy()
        for n in hierarchy:
            if isinstance(n, DataObjectNode):
                self.clear_data_object_uncertainties([n.data_object])
                self.update_gui_model([n.data_object])

    def clear_data_object_uncertainties(self, data_objects):
        if self._hold_updating:
            return

        for data_object in data_objects:
            # retrive data_object node from the treeModel
            node = self.treeModel.data_object_node(data_object.name)
            descendants = node.descendants()
            # filter out all the parameter nodes
            par_nodes = [n for n in descendants if isinstance(n, ParNode)]
            for par in par_nodes:
                p = par.parameter
                p.stderr = None

                parent, row = par.parent(), par.row()
                idx1 = self.treeModel.index(row, 2, parent.index)
                idx2 = self.treeModel.index(row, 2, parent.index)
                self.treeModel.dataChanged.emit(idx1, idx2)

    def tree_model_data_changed(self, top_left, bottom_right,
                                role=QtCore.Qt.EditRole):
        # if you've just changed whether you want to hold or vary a parameter
        # there is no need to update the reflectivity plots
        if not top_left.isValid():
            return

        # find out which data_object / model we're adjusting
        node = top_left.internalPointer()
        if node is None:
            return

        if (len(role) and role[0] == QtCore.Qt.CheckStateRole and
                isinstance(node, DataObjectNode)):
            # set visibility of data_object
            graph_properties = node.data_object.graph_properties
            graph_properties.visible = (node.visible is True)
            self.redraw_data_object_graphs([node.data_object])
            return

        # only redraw if you're altering values
        # otherwise we'd be performing continual updates of the model
        if top_left.column() != 1:
            return

        hierarchy = node.hierarchy()
        for n in hierarchy:
            if isinstance(n, DataObjectNode):
                self.clear_data_object_uncertainties([n.data_object])
                self.update_gui_model([n.data_object])

    def calculate_chi2(self, data_objects):
        # calculate chi2 for all the data objects
        # TODO correct resolution here
        if not len(data_objects):
            return

        if self._hold_updating:
            return

        for data_object in data_objects:
            if data_object.name == 'theoretical':
                continue

            useerrors = self.settings.useerrors

            t = None
            if self.settings.transformdata is not None:
                transform = Transform(self.settings.transformdata)
                t = transform

            # TODO implement correct resolution smearing
            objective = Objective(data_object.model,
                                  data_object.dataset,
                                  transform=t,
                                  use_weights=useerrors)
            chisqr = objective.chisqr()
            node = self.treeModel.data_object_node(data_object.name)
            node.chi2 = chisqr
            index = self.treeModel.index(node.row(), 2, node.parent().index)
            self.treeModel.dataChanged.emit(index, index)

    def update_gui_model(self, data_objects):
        # TODO correct resolution here
        if not len(data_objects):
            return

        if self._hold_updating:
            return

        self.redraw_data_object_graphs(data_objects)
        self.calculate_chi2(data_objects)

    def add_data_objects_to_graphs(self, data_objects):
        transform = Transform(self.settings.transformdata)
        t = transform

        self.reflectivitygraphs.add_data_objects(data_objects, transform=t)
        self.sldgraphs.add_data_objects(data_objects)


class ProgressCallback(QtWidgets.QDialog):
    def __init__(self, parent=None, objective=None):
        self.start = time.time()
        self.last_time = time.time()
        self.abort_flag = False
        super(ProgressCallback, self).__init__(parent)
        self.parent = parent
        self.ui = uic.loadUi(os.path.join(UI_LOCATION, 'progress.ui'), self)
        self.elapsed = 0.
        self.chi2 = 1.e308
        self.ui.timer.display(float(self.elapsed))
        self.ui.buttonBox.rejected.connect(self.abort)
        self.objective = objective

    def abort(self):
        self.abort_flag = True

    def callback(self, xk, *args, **kwds):
        # a callback for scipy.optimize.minimize, which enters
        # every iteration.
        new_time = time.time()
        if new_time - self.last_time > 1:
            # gp = self.dataset.graph_properties
            # if gp.line2Dfit is not None:
            #     self.parent.redraw_data_object_graphs([self.dataset])
            # else:
            #     self.parent.add_datasets_to_graphs([self.dataset])

            self.elapsed = new_time - self.start
            self.ui.timer.display(float(self.elapsed))
            self.last_time = new_time

            text = 'Chi2 : %f' % self.objective.chisqr(xk)
            self.ui.values.setPlainText(text)
            QtWidgets.QApplication.processEvents()
            if self.abort_flag:
                raise RuntimeError("WARNING: FIT WAS TERMINATED EARLY")

        return self.abort_flag


class ProgramSettings(object):

    def __init__(self, **kwds):
        _members = {'fitting_algorithm': 'DE',
                    'transformdata': 'logY',
                    'quad_order': 17,
                    'current_dataset_name': None,
                    'experiment_file_name': '',
                    'current_model_name': None,
                    'usedq': True,
                    'resolution': 5,
                    'fit_plugin': None,
                    'useerrors': True}

        for key in _members:
            if key in kwds:
                setattr(self, key, kwds[key])
            else:
                setattr(self, key, _members[key])

    def __getitem__(self, key):
        return self.__dict__[key]

    def __getattr_(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        if key in self.__dict__:
            setattr(self, key, value)


class NavToolBar(NavigationToolbar):
    """
    This class overloads the NavigationToolbar of matplotlib.
    """
    def __init__(self, canvas, parent, coordinates=True):
        NavigationToolbar.__init__(self, canvas, parent, coordinates)
        self.setIconSize(QtCore.QSize(20, 20))


class MyReflectivityGraphs(FigureCanvas):

    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1, 1, 1), edgecolor=(0, 0, 0))

        # reflectivity graph
        self.axes = []
        ax = self.figure.add_axes([0.06, 0.15, 0.9, 0.8])
        ax.margins(0.0005)
        self.axes.append(ax)

        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.axes[0].set_xlabel('Q')
        self.axes[0].set_ylabel('R')
        # self.axes[0].set_yscale('log')

# residual plot
# , sharex=self.axes[0]
# ax2 = self.figure.add_axes([0.1,0.04,0.85,0.14], sharex=ax, frame_on = False)
#   self.axes.append(ax2)
#   self.axes[1].set_visible(True)
#   self.axes[1].set_ylabel('residual')

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.mpl_toolbar = NavToolBar(self, parent)
        self.figure.canvas.mpl_connect('pick_event', self._pick_event)
        # self.figure.canvas.mpl_connect('key_press_event', self._key_press)

        self.draw()

    def _key_press(self, event):
        # auto scale
        if event.key == 'super+a':
            self.autoscale()

    def _pick_event(self, event):
        # pick event was a double click on the graph
        if (event.mouseevent.dblclick and
                event.mouseevent.button == 1):
            if isinstance(event.artist, lines.Line2D):
                self.mpl_toolbar.edit_parameters()

    def autoscale(self):
        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()

    def add_data_objects(self, data_objects, transform=None):
        for data_object in data_objects:
            dataset = data_object.dataset

            graph_properties = data_object.graph_properties

            if (graph_properties.ax_data is None and
                    data_object.name != 'theoretical'):
                yt = dataset.y
                if transform is not None:
                    yt, edata = transform(dataset.x, dataset.y, dataset.y_err)

                # add the dataset
                line_instance = self.axes[0].plot(dataset.x,
                                                  yt,
                                                  markersize=3,
                                                  marker='o',
                                                  linestyle='',
                                                  markeredgecolor=None,
                                                  label=dataset.name,
                                                  picker=5)
                mfc = artist.getp(line_instance[0], 'markerfacecolor')
                artist.setp(line_instance[0], **{'markeredgecolor': mfc})

                graph_properties['ax_data'] = line_instance[0]
                if graph_properties['data_properties']:
                    artist.setp(graph_properties.ax_data,
                                **graph_properties['data_properties'])

            yfit_t = data_object.generative
            if graph_properties.ax_fit is None and yfit_t is not None:
                if transform is not None:
                    yfit_t, temp = transform(dataset.x, yfit_t)

                color = 'b'
                if graph_properties.ax_data is not None:
                    color = artist.getp(graph_properties.ax_data, 'color')
                # add the fit
                graph_properties['ax_fit'] = self.axes[0].plot(
                    dataset.x,
                    yfit_t,
                    linestyle='-',
                    color=color,
                    lw=1,
                    label='fit_' + data_object.name,
                    picker=5)[0]
                if graph_properties['fit_properties']:
                    artist.setp(graph_properties.ax_fit,
                                **graph_properties['fit_properties'])

            # if (dataObject.line2Dresiduals is None and
            #     dataObject.residuals is not None):
            #     dataObject.line2Dresiduals = self.axes[1].plot(
            #         dataObject.x,
            #         dataObject.residuals,
            #         linestyle='-',
            #         lw = 1,
            #         label = 'residuals_' + dataObject.name)[0]
            #
            #     if graph_properties['residuals_properties']:
            #         artist.setp(dataObject.ax_residuals,
            #                     **graph_properties['residuals_properties'])

            graph_properties.save_graph_properties()
        self.draw()

    def redraw_data_objects(self, data_objects, transform=None):
        if not len(data_objects):
            return

        for data_object in data_objects:
            if not data_object:
                continue

            dataset = data_object.dataset
            model = data_object.model

            if data_object.name != 'theoretical':
                y = dataset.y
                if transform is not None:
                    y, _ = transform(dataset.x, y)

            if model is not None:
                # TODO take care of correct resolution
                yfit = model(dataset.x, x_err=dataset.x_err)

                if transform is not None:
                    yfit, efit = transform(dataset.x, yfit)

            graph_properties = data_object.graph_properties
            visible = graph_properties.visible

            if graph_properties.ax_data is not None:
                graph_properties.ax_data.set_data(dataset.x, y)
                graph_properties.ax_data.set_visible(visible)
            if graph_properties.ax_fit is not None:
                graph_properties.ax_fit.set_data(dataset.x, yfit)
                graph_properties.ax_fit.set_visible(visible)

#          if dataObject.line2Dresiduals:
#             dataObject.line2Dresiduals.set_data(dataObject.x,
#                                          dataObject.residuals)
#             dataObject.line2Dresiduals.set_visible(visible)

        self.draw()

    def remove_trace(self, data_object):
        graph_properties = data_object.graph_properties
        if graph_properties.ax_data is not None:
            graph_properties.ax_data.remove()
            graph_properties.ax_data = None

        if graph_properties.ax_fit is not None:
            graph_properties.ax_fit.remove()
            graph_properties.ax_fit = None

        if graph_properties.ax_residuals is not None:
            graph_properties.ax_residuals.remove()
            graph_properties.ax_residuals = None
        self.draw()

    def remove_traces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]
        self.draw()


class MySLDGraphs(FigureCanvas):

    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.axes = []
        # SLD plot
        self.axes.append(self.figure.add_subplot(111))

        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.axes[0].set_xlabel('z')
        self.axes[0].set_ylabel('SLD')

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.98)
        self.mpl_toolbar = NavigationToolbar(self, parent)

    def redraw_data_objects(self, data_objects):
        for data_object in data_objects:
            if not data_object:
                continue
            graph_properties = data_object.graph_properties
            visible = graph_properties.visible

            if (graph_properties.ax_sld_profile and
                    data_object.model is not None):
                sld_profile = data_object.model.structure.sld_profile()

                graph_properties.ax_sld_profile.set_data(
                    sld_profile[0],
                    sld_profile[1])
                graph_properties.ax_sld_profile.set_visible(visible)
        self.axes[0].relim()
        self.axes[0].autoscale_view(None, True, True)
        self.draw()

    def add_data_objects(self, data_objects):
        for data_object in data_objects:
            graph_properties = data_object.graph_properties
            if (graph_properties.ax_sld_profile is None and
                    data_object.sld_profile is not None):

                color = 'r'
                lw = 2
                if graph_properties.ax_data:
                    color = artist.getp(graph_properties.ax_data, 'color')
                    lw = artist.getp(graph_properties.ax_data, 'lw')

                graph_properties['ax_sld_profile'] = self.axes[0].plot(
                    data_object.sld_profile[0],
                    data_object.sld_profile[1],
                    linestyle='-',
                    color=color,
                    lw=lw,
                    label='sld_' + data_object.name)[0]

                if graph_properties['sld_profile_properties']:
                    artist.setp(graph_properties.ax_sld_profile,
                                **graph_properties['sld_profile_properties'])

        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()

    def remove_trace(self, data_object):
        if data_object.ax_sld_profile:
            data_object.ax_sld_profile.remove()
        self.draw()

    def remove_traces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]

        self.draw()


class DataSelectionChanges(QtCore.QObject):
    change = QtCore.pyqtSignal(int)

    def __init__(self):
        super(DataSelectionChanges, self).__init__()

    @QtCore.pyqtSlot(int)
    def selectionChanged(self, arg_1):
        self.change.emit(arg_1)


class EmittingStream(QtCore.QObject):
    # a class for rewriting stdout to a console window
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class OpenMenu(QtWidgets.QMenu):
    """
    A context menu class for the model tree view
    """
    def __init__(self, parent=None):
        super(OpenMenu, self).__init__(parent)
        self._parent = parent
        self.copy_from_action = self.addAction("Copy a model to here")
        self.addSeparator()
        self.add_to_fit_action = self.addAction("Add to fit")
        self.remove_from_fit_action = self.addAction("Remove from fit")
        self.addSeparator()
        self.link_action = self.addAction("Link parameters")
        self.unlink_action = self.addAction("Unlink parameters")

    def __call__(self, position):
        action = self.exec_(self._parent.mapToGlobal(position))
        if action == self.link_action:
            pass
        if action == self.unlink_action:
            pass


def msg(text):
    msgBox = QtWidgets.QMessageBox()
    msgBox.setText(text)
    msgBox.exec_()
    return
