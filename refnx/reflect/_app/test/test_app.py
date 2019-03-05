import os.path
from os.path import join as pjoin

from PyQt5 import QtWidgets, QtCore, QtGui

from refnx.reflect._app.view import MotofitMainWindow
import refnx.dataset as refd
from refnx.reflect._app import resources_rc
from refnx.reflect import Spline


def mysetup(qtbot):
    # app = QtWidgets.QApplication([])
    # app.setWindowIcon(QtGui.QIcon(':icons/scattering.png'))
    myapp = MotofitMainWindow()
    model = myapp.treeModel
    qtbot.add_widget(myapp)

    return myapp, model


def test_myapp(qtbot, tmpdir):
    myapp, model = mysetup(qtbot)

    # test if we can load a known-good datafile, and save and load the
    # an experiment
    ###########################################
    # load a file
    pth = os.path.dirname(refd.__file__)
    f = pjoin(pth, 'test', 'c_PLP0000708.dat')
    myapp.load_data([f])
    assert(len(model.datastore) == 2)

    myapp2 = save_and_reload_experiment(myapp, tmpdir)
    model2 = myapp2.treeModel
    assert(len(model2.datastore) == 2)


def test_add_spline_save(qtbot, tmpdir):
    # test if we can add a spline to a model and save an experiment
    myapp, model = mysetup(qtbot)

    # get index of theoretical dataset --> structure --> slab1
    data_object_node = model.data_object_node('theoretical')
    model_node = data_object_node.child(1)
    structure_node = model_node.child(3)

    # selection_model = myapp.ui.treeView.selectionModel()
    # slab_node = structure_node.child(1)
    # selection_model.select(slab_node.index,
    #                        QtCore.QItemSelectionModel.Select)

    # add a Spline after the slab
    component = Spline(50, [-1., -1.], [0.33, 0.33], name='spline')
    structure_node.insert_component(1, component)
    save_and_reload_experiment(myapp, tmpdir)


def save_and_reload_experiment(app, tmpdir):
    # save and reopen experiment.
    sf = pjoin(str(tmpdir), 'experiment1.mtft')
    # this is just to make sure that the file exists
    with open(sf, 'wb') as f:
        f.write(b'sksij')

    app.settings.experiment_file_name = sf
    app.on_actionSave_File_triggered()

    myapp2 = MotofitMainWindow()
    myapp2._restore_state(sf)
    return myapp2
