from __future__ import division
import logging
import os.path


from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtCore import pyqtSlot
import numpy as np

import matplotlib
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg
                                                as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib import patches
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT
                                                as NavigationToolbar)

from refnx.reduce.peak_utils import peak_finder
from refnx.reduce.platypusnexus import EXTENT_MULT, PIXEL_OFFSET

matplotlib.use('Qt5Agg')


class ManualBeamFinder(QtWidgets.QDialog):
    """
    Manual specular beam finding dialogue
    """
    def __init__(self, ui_loc=''):
        """
        """
        super(ManualBeamFinder, self).__init__()
        self.dialog = uic.loadUi(os.path.join(ui_loc, 'manual_beam.ui'), self)

        # values for spinboxes
        self._true_centre = 121.
        self._true_sd = 5.

        self._pixels_to_include = 200
        self._integrate_width = 50
        self._integrate_position = 121
        self._lopx = 122
        self._hipx = 122
        self._low_bkg = 122
        self._high_bkg = 122

        # detector image
        self.detector_image_layout = QtWidgets.QGridLayout(
            self.dialog.detector_image)
        self.detector_image_layout.setObjectName("detector_image_layout")

        self.detector_image = DetectorImage(self.dialog.detector_image)
        self.toolbar = NavToolBar(self.detector_image, self)
        self.detector_image_layout.addWidget(self.detector_image)
        self.detector_image_layout.addWidget(self.toolbar)

        # cross section image
        self.cross_section_layout = QtWidgets.QGridLayout(
            self.dialog.cross_section)
        self.cross_section_layout.setObjectName("cross_section_layout")

        self.cross_section = Cross_Section(self.dialog.cross_section)
        self.cross_section_layout.addWidget(self.cross_section)

    def __call__(self, detector, detector_err):
        """
        Start the manual beam find

        Parameters
        ----------
        detector : np.ndarray
            detector image
        detector_err: np.ndarray
            uncertainties (sd) associated with detector image

        Returns
        -------
        beam_centre, beam_sd : float, float
            Beam centre and standard deviation
        """
        self.detector = detector
        self.detector_err = detector_err

        self.recalculate_graphs()

        self.dialog.exec_()
        return np.array([self._true_centre]), np.array([self._true_sd])

    def recalculate_graphs(self):
        """
        After the ROI for the beam find has been changed redraw the detector
        cross section and recalculate the beam centre and widths.
        """
        ret = calculate_centre(self.detector,
                               self.detector_err,
                               self._integrate_position,
                               self._pixels_to_include,
                               self._integrate_width)

        x, xs, beam_centre, beam_sd = ret

        self._true_centre = beam_centre
        self._true_sd = beam_sd

        regions = fore_back_region(beam_centre, beam_sd)
        self._lopx, self._hipx, self._low_bkg, self._high_bkg = regions

        self.detector_image.display_image(self.detector, beam_centre, *regions,
                                          self._pixels_to_include,
                                          self._integrate_width,
                                          self._integrate_position)
        self.cross_section.display_cross_section(x, xs, beam_centre, *regions)

        self.true_centre.setValue(self._true_centre)
        self.true_fwhm.setValue(self._true_sd * 2.3548)

    def redraw_cross_section_regions(self):
        """
        After the peak centre/width is recalculated redraw the foreground and
        background regions on the detector image and cross section.
        """
        # first do cross section
        figcanvas = self.cross_section

        # background regions
        figcanvas.l_lbkg.set_xdata(self._low_bkg)
        figcanvas.l_hbkg.set_xdata(self._high_bkg)

        # beam centre
        figcanvas.l_bc.set_xdata(self._true_centre)

        # foreground regions
        figcanvas.l_lfore.set_xdata(self._lopx)
        figcanvas.l_hfore.set_xdata(self._hipx)

        # redraw cross section
        figcanvas.draw()

        # then do detector image
        figcanvas = self.detector_image

        # background regions
        figcanvas.l_lbkg.set_ydata(self._low_bkg - 0.5)
        figcanvas.l_hbkg.set_ydata(self._high_bkg + 0.5)

        # beam centre
        figcanvas.l_bc.set_ydata(self._true_centre)

        # foreground regions
        figcanvas.l_lfore.set_ydata(self._lopx - 0.5)
        figcanvas.l_hfore.set_ydata(self._hipx + 0.5)

        figcanvas.draw()

    @pyqtSlot(float)
    def on_true_centre_valueChanged(self, val):
        self._true_centre = val

        regions = fore_back_region(self._true_centre, self._true_sd)
        self._lopx, self._hipx, self._low_bkg, self._high_bkg = regions
        self.redraw_cross_section_regions()

    @pyqtSlot(float)
    def on_true_fwhm_valueChanged(self, val):
        self._true_sd = val / 2.3548

        regions = fore_back_region(self._true_centre, self._true_sd)
        self._lopx, self._hipx, self._low_bkg, self._high_bkg = regions
        self.redraw_cross_section_regions()

    @pyqtSlot(int)
    def on_pixels_to_include_valueChanged(self, val):
        self._pixels_to_include = val
        self.recalculate_graphs()

    @pyqtSlot(int)
    def on_integrate_width_valueChanged(self, val):
        self._integrate_width = val
        self.recalculate_graphs()

    @pyqtSlot(int)
    def on_integrate_position_valueChanged(self, val):
        self._integrate_position = val
        self.recalculate_graphs()


def fore_back_region(beam_centre, beam_sd):
    """
    Calculates the fore and background regions based on the beam centre and
    width
    """
    lopx = np.floor(beam_centre - beam_sd * EXTENT_MULT).astype('int')
    hipx = np.ceil(beam_centre + beam_sd * EXTENT_MULT).astype('int')

    # limit of background regions
    # from refnx.reduce.platypusnexus
    y1 = np.round(lopx - PIXEL_OFFSET).astype('int')
    y2 = np.round(hipx + PIXEL_OFFSET).astype('int')

    low_bkg = np.round(y1 - (EXTENT_MULT * beam_sd)).astype('int')
    high_bkg = np.round(y2 + (EXTENT_MULT * beam_sd)).astype('int')

    return lopx, hipx, low_bkg, high_bkg


def calculate_centre(detector, detector_err, integrate_position,
                     pixels_to_include, integrate_width):
    """
    Calculates the beam centre from a detector cross section

    Parameters
    ----------
    detector : np.ndarray
        detector image
    detector_err : np.ndarray
        uncertainty in detector image (standard deviation)

    Returns
    -------
    x : np.ndarray
        x-coordinates for the cross section
    xs : np.ndarray
        The cross section
    xs_err : np.ndarray
        Uncertainty in cross section (standard deviation)
    peak_centre : float
        peak centre (Gaussian - fitted)
    peak_sd : float
        peak standard deviation (Gaussian - fitted)
    """
    # remove the regions of the detector image we don't want to look at
    det = np.squeeze(detector)
    det_err = np.squeeze(detector_err)

    pixels = det.shape[-1]
    low_bracket = max(0, int(np.floor(integrate_position - integrate_width/2)))
    high_bracket = min(int(np.ceil(integrate_position + integrate_width/2)),
                       pixels - 1)

    x = np.arange(det.shape[-1], dtype=float)
    det = det[-pixels_to_include:, low_bracket:high_bracket + 1]
    x = x[low_bracket: high_bracket + 1]
    det_err = det_err[:-pixels_to_include, low_bracket:high_bracket + 1]

    xs = np.sum(det, axis=0)
    # xs_err = np.sqrt(np.sum(det_err**2, axis=0))

    # peak finder returns (centroid, gaussian coefs)
    beam_centre, beam_sd = peak_finder(xs, x=x)[1]

    return x, xs, beam_centre, beam_sd


class DetectorImage(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure()
        super(DetectorImage, self).__init__(self.figure)
        self.setParent(parent)

        self.axes = self.figure.add_axes([0.08, 0.06, 0.9, 0.93])
        self.axes.margins(0.0005)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.mpl_connect('motion_notify_event', self.mouse_move)

    def display_image(self, detector, beam_centre, low_px, high_px, low_bkg,
                      high_bkg, pixels_to_include, integrate_width,
                      integrate_position):
        self.axes.clear()

        # want the first colour to be white
        disp = np.copy(detector[0])
        disp[disp == 0.0] = np.nan

        self.axes.imshow(disp.T,
                         aspect='auto',
                         origin='lower',
                         vmin=-1 / np.max(detector))
        # display a rectangle that shows where we're looking for beam
        patch_x = np.size(disp, 0) - pixels_to_include
        patch_y = max(0, integrate_position - integrate_width/2)
        width = min(integrate_width, np.size(disp, 1) - patch_y)
        rect = patches.Rectangle((patch_x, patch_y),
                                 pixels_to_include,
                                 width,
                                 fill=False,
                                 color='yellow')
        self.axes.add_patch(rect)

        # also display foreground/background regions
        # background regions
        self.l_lbkg = self.axes.axhline(color='black')  # the vert line
        self.l_lbkg.set_ydata(low_bkg - 0.5)

        self.l_hbkg = self.axes.axhline(color='black')  # the vert line
        self.l_hbkg.set_ydata(high_bkg + 0.5)

        # beam centre
        self.l_bc = self.axes.axhline(color='red')  # the vert line
        self.l_bc.set_ydata(beam_centre)

        # foreground regions
        self.l_lfore = self.axes.axhline(color='blue')  # the vert line
        self.l_lfore.set_ydata(low_px - 0.5)

        self.l_hfore = self.axes.axhline(color='blue')  # the vert line
        self.l_hfore.set_ydata(high_px + 0.5)

        self.draw()

    # def mouse_move(self, event):
    #     if not event.inaxes:
    #         return
    #
    #     x, y = event.xdata, event.ydata
    #     # update the line positions
    #     self.lx.set_ydata(y)
    #     self.ly.set_xdata(x)
    #
    #     # self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
    #     # self.axes.draw_artist(self.ly)
    #     # self.axes.draw_artist(self.lx)
    #     # self.figure.canvas.update()
    #     # self.figure.canvas.flush_events()
    #     self.draw()


class Cross_Section(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure()
        super(Cross_Section, self).__init__(self.figure)
        self.setParent(parent)

        self.axes = self.figure.add_axes([0.1, 0.07, 0.95, 0.94])
        self.axes.margins(0.0005)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.mpl_connect('motion_notify_event', self.mouse_move)

    def display_cross_section(self, x, xs, beam_centre, low_px,
                              high_px, low_bkg, high_bkg):

        self.axes.clear()
        self.axes.plot(x, xs)
        self.axes.set_xlim(np.min(x), np.max(x))

        # background regions
        self.l_lbkg = self.axes.axvline(color='black')  # the vert line
        self.l_lbkg.set_xdata(low_bkg)

        self.l_hbkg = self.axes.axvline(color='black')  # the vert line
        self.l_hbkg.set_xdata(high_bkg)

        # beam centre
        self.l_bc = self.axes.axvline(color='red')  # the vert line
        self.l_bc.set_xdata(beam_centre)

        # foreground regions
        self.l_lfore = self.axes.axvline(color='blue')  # the vert line
        self.l_lfore.set_xdata(low_px)

        self.l_hfore = self.axes.axvline(color='blue')  # the vert line
        self.l_hfore.set_xdata(high_px)

        # for a cursor
        # self.ly = self.axes.axvline(color='k')  # the vert line
        # self.lx = self.axes.axhline(color='k')  # the vert line

        # text location in axes coords
        self.txt = self.axes.text(0.6, 0.9, '', transform=self.axes.transAxes)
        self.draw()

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        # update the line positions
        # self.ly.set_xdata(x)
        # self.lx.set_xdata(y)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        # self.axes.draw_artist(self.ly)
        # self.axes.draw_artist(self.lx)
        # self.figure.canvas.update()
        # self.figure.canvas.flush_events()
        self.draw()


class NavToolBar(NavigationToolbar):
    """
    Toolbar for the detector image
    """
    toolitems = [
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom')]

    def __init__(self, canvas, parent, coordinates=True):
        NavigationToolbar.__init__(self, canvas, parent, coordinates)
        self.setIconSize(QtCore.QSize(15, 15))
