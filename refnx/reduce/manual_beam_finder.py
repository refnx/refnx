import os.path

from qtpy import QtCore, QtWidgets, uic
import numpy as np

import matplotlib
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib import patches
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)

from refnx.reduce.peak_utils import peak_finder, centroid
from refnx.reduce.platypusnexus import fore_back_region, PIXEL_OFFSET
import refnx.reduce._app as floc

matplotlib.use("QtAgg")


UI_LOCATION = os.path.join(os.path.dirname(floc.__file__), "ui")


class ManualBeamFinder(QtWidgets.QDialog):
    """
    Manual specular beam finding dialogue.

    If being created from a Jupyter notebook then the ipython magic
    ``%gui qt`` should be used before creating an instance of this class.
    Otherwise the ipython kernel will crash immediately.
    """

    def __init__(self, parent=None):
        """ """
        super().__init__()
        self.dialog = uic.loadUi(
            os.path.join(UI_LOCATION, "manual_beam.ui"), self
        )

        # values for spinboxes
        self._true_centre = 500.0
        self._true_sd = 20.0

        self._pixels_to_include = 200
        self._integrate_width = 200
        self._integrate_position = 500
        self._low_px = 500
        self._high_px = 501
        self._low_bkg = 503
        self._high_bkg = 499

        # detector image
        self.detector_image_layout = QtWidgets.QGridLayout(
            self.dialog.detector_image
        )
        self.detector_image_layout.setObjectName("detector_image_layout")

        self.detector_image = DetectorImage(self.dialog.detector_image)
        self.toolbar = NavToolBar(self.detector_image, self)
        self.detector_image_layout.addWidget(self.detector_image)
        self.detector_image_layout.addWidget(self.toolbar)

        # cross section image
        self.cross_section_layout = QtWidgets.QGridLayout(
            self.dialog.cross_section
        )
        self.cross_section_layout.setObjectName("cross_section_layout")

        self.cross_section = Cross_Section(self.dialog.cross_section)
        self.cross_section_layout.addWidget(self.cross_section)

        # register a listener to drag events on cross section
        self.cross_section.mpl_connect(
            "button_release_event", self.on_cross_drag_release
        )

    def __call__(self, detector, detector_err, name):
        """
        Start the manual beam find

        Parameters
        ----------
        detector : np.ndarray
            detector image. Shape `(N, T, Y)` or `(T, Y)`. If N > 1 then only
            the first image is processed
        detector_err: np.ndarray
            uncertainties (sd) associated with detector image
        name: str
            Name of the dataset

        Returns
        -------
        beam_centre, beam_sd, lopx, hipx, background_pixels :
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list of np.ndarray

            Beam centre, standard deviation, lowest pixel in foreground region,
            highest pixel in foreground region, each of the entries in
            `background_pixels` is an array specifying pixels that are in the
            background region.
        """
        # assume that the ndim is 2 or 3.
        # only process the first detector image (N = 0).
        self.detector = detector
        self.detector_err = detector_err
        n_images = 1

        if detector.ndim > 2:
            self.detector = detector[0]
            self.detector_err = detector_err[0]
            n_images = np.size(detector, 0)

        # set min/max values for the detector image GUI. Crashes result
        # otherwise
        self.integrate_position.setMaximum(np.size(self.detector, -1) - 1)
        self.integrate_width.setMaximum(np.size(self.detector, -1) - 1)
        self.pixels_to_include.setMaximum(np.size(self.detector, 0))
        self.true_centre.setMaximum(np.size(self.detector, -1))

        # guess peak centre from centroid.
        xs = np.sum(self.detector, axis=0)
        self._integrate_position, _ = centroid(xs)
        self.integrate_position.setValue(
            int(np.round(self._integrate_position))
        )
        self.integrate_width.setValue(self._integrate_width)

        self.recalculate_graphs()
        # set the title of the window to give context about what dataset is
        # being processed.
        self.setWindowTitle("Manual beam finder")
        if name is not None:
            self.setWindowTitle("Manual beam finder: " + name)

        self.dialog.exec()

        y1 = int(round(self._low_px - PIXEL_OFFSET))
        y2 = int(round(self._high_px + PIXEL_OFFSET))
        background_pixels = np.r_[
            np.arange(self._low_bkg, y1 + 1), np.arange(y2, self._high_bkg + 1)
        ]

        return (
            np.ones((n_images,)) * self._true_centre,
            np.ones((n_images,)) * self._true_sd,
            np.ones((n_images,)) * self._low_px,
            np.ones((n_images,)) * self._high_px,
            [background_pixels for i in range(n_images)],
        )

    def recalculate_graphs(self):
        """
        After the ROI for the beam find has been changed redraw the detector
        cross section and recalculate the beam centre and widths.
        """
        x, xs, xs_err = get_cross_section(
            self.detector,
            self.detector_err,
            self._pixels_to_include,
            self._integrate_position,
            self._integrate_width,
        )

        # peak finder returns (centroid, gaussian coefs)
        beam_centre, beam_sd = peak_finder(xs, x=x)[1]

        self._true_centre = beam_centre
        self._true_sd = beam_sd

        regions = fore_back_region(beam_centre, beam_sd)
        self._low_px, self._high_px, bp = regions

        # perhaps fore_back_region returned regions that weren't on the
        # detector
        self._low_px = np.clip(self._low_px, 0, np.size(self.detector, 1))
        self._high_px = np.clip(self._high_px, 0, np.size(self.detector, 1))

        # perhaps fore_back_region returned no background pixels
        if len(bp[0]) > 0:
            self._low_bkg = np.min(bp[0])
            self._high_bkg = np.max(bp[0])

        self.detector_image.display_image(
            self.detector,
            beam_centre,
            self._low_px,
            self._high_px,
            self._low_bkg,
            self._high_bkg,
            self._pixels_to_include,
            self._integrate_width,
            self._integrate_position,
        )
        self.cross_section.display_cross_section(
            x,
            xs,
            beam_centre,
            self._low_px,
            self._high_px,
            self._low_bkg,
            self._high_bkg,
        )

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
        figcanvas.l_lfore.set_xdata(self._low_px)
        figcanvas.l_hfore.set_xdata(self._high_px)

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
        figcanvas.l_lfore.set_ydata(self._low_px - 0.5)
        figcanvas.l_hfore.set_ydata(self._high_px + 0.5)

        figcanvas.draw()

    def on_cross_drag_release(self, event):
        """
        A listener for press->drag->release events on the cross section plot.
        This is where the user is graphically editing foreground/background
        regions
        """
        low_bkg = self.cross_section._low_bkg
        high_bkg = self.cross_section._high_bkg
        lopx = self.cross_section._low_px
        hipx = self.cross_section._high_px

        dragged_attr, dragged_line = self.cross_section._press[0]
        if (lopx >= hipx) or (low_bkg >= high_bkg):
            # set it back to what it was
            setattr(
                self.cross_section, dragged_attr, getattr(self, dragged_attr)
            )
            # and redraw the line
            dragged_line.set_xdata(getattr(self, dragged_attr))
            self.cross_section.draw()
            return
        else:
            # drag was legal, update the attribute in here
            setattr(
                self, dragged_attr, getattr(self.cross_section, dragged_attr)
            )

        # recalculate beam_centre, beam_sd based off cross section
        x, xs, xs_err = get_cross_section(
            self.detector,
            self.detector_err,
            self._pixels_to_include,
            self._integrate_position,
            self._integrate_width,
        )
        # trim to foreground region
        in_foreground = np.logical_and(x >= lopx, x <= hipx)
        x = x[in_foreground]
        xs = xs[in_foreground]

        # calculate peak centre
        self._true_centre, _ = peak_finder(xs, x=x)[1]
        self._true_sd = (hipx - lopx) / 4.0

        # redraw beam centre on detector image and cross section
        self.cross_section.l_bc.set_xdata(self._true_centre)
        self.cross_section.draw()

        self.detector_image.l_lbkg.set_ydata(self._low_bkg - 0.5)
        self.detector_image.l_hbkg.set_ydata(self._high_bkg + 0.5)
        self.detector_image.l_lfore.set_ydata(self._low_px - 0.5)
        self.detector_image.l_hfore.set_ydata(self._high_px + 0.5)
        self.detector_image.l_bc.set_ydata(self._true_centre)

        self.detector_image.draw()

        # update the spinboxes, but without triggering its slot
        self.true_centre.valueChanged.disconnect()
        self.true_centre.setValue(self._true_centre)
        self.true_centre.valueChanged.connect(self.on_true_centre_valueChanged)

        self.true_fwhm.valueChanged.disconnect()
        self.true_fwhm.setValue(self._true_sd * 2.3548)
        self.true_fwhm.valueChanged.connect(self.on_true_fwhm_valueChanged)

    @QtCore.Slot(float)
    def on_true_centre_valueChanged(self, val):
        self._true_centre = val

        regions = fore_back_region(self._true_centre, self._true_sd)
        self._low_px, self._high_px, bp = regions

        # perhaps fore_back_region returned regions that weren't on the
        # detector
        self._low_px = np.clip(self._low_px, 0, np.size(self.detector, 1))
        self._high_px = np.clip(self._high_px, 0, np.size(self.detector, 1))

        # perhaps fore_back_region returned no background pixels
        if len(bp[0]) > 0:
            self._low_bkg = np.min(bp[0])
            self._high_bkg = np.max(bp[0])

        self.redraw_cross_section_regions()

    @QtCore.Slot(float)
    def on_true_fwhm_valueChanged(self, val):
        self._true_sd = val / 2.3548

        regions = fore_back_region(self._true_centre, self._true_sd)
        self._low_px, self._high_px, bp = regions
        self._low_bkg = np.min(bp[0])
        self._high_bkg = np.max(bp[0])

        self.redraw_cross_section_regions()

    @QtCore.Slot(int)
    def on_pixels_to_include_valueChanged(self, val):
        self._pixels_to_include = val
        self.recalculate_graphs()

    @QtCore.Slot(int)
    def on_integrate_width_valueChanged(self, val):
        self._integrate_width = val
        self.recalculate_graphs()

    @QtCore.Slot(int)
    def on_integrate_position_valueChanged(self, val):
        self._integrate_position = val
        self.recalculate_graphs()


def get_cross_section(
    detector,
    detector_err,
    pixels_to_include,
    integrate_position,
    integrate_width,
):
    """
    Obtains the detector cross section

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
    """
    det = np.squeeze(detector)
    det_err = np.squeeze(detector_err)

    pixels = det.shape[-1]
    low_bracket = max(
        0, int(np.floor(integrate_position - integrate_width / 2))
    )
    high_bracket = min(
        int(np.ceil(integrate_position + integrate_width / 2)), pixels - 1
    )

    x = np.arange(det.shape[-1], dtype=float)
    det = det[-pixels_to_include:, low_bracket : high_bracket + 1]
    x = x[low_bracket : high_bracket + 1]
    det_err = det_err[:-pixels_to_include, low_bracket : high_bracket + 1]

    # sum over time bins
    xs = np.sum(det, axis=0)
    xs_err = np.sqrt(np.sum(det_err**2, axis=0))

    return x, xs, xs_err


class DetectorImage(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)

        self.axes = self.figure.add_axes([0.08, 0.06, 0.9, 0.93])
        self.axes.margins(0.0005)

        FigureCanvas.setSizePolicy(
            self,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        FigureCanvas.updateGeometry(self)
        # self.mpl_connect('motion_notify_event', self.mouse_move)

    def display_image(
        self,
        detector,
        beam_centre,
        low_px,
        high_px,
        low_bkg,
        high_bkg,
        pixels_to_include,
        integrate_width,
        integrate_position,
    ):
        self.axes.clear()

        # want the first colour to be white
        disp = np.copy(detector)
        disp[disp == 0.0] = np.nan

        self.axes.imshow(
            disp.T, aspect="auto", origin="lower", vmin=-1 / np.max(detector)
        )
        # display a rectangle that shows where we're looking for beam
        patch_x = np.size(disp, 0) - pixels_to_include
        patch_y = max(0, integrate_position - integrate_width / 2)
        width = min(integrate_width, np.size(disp, 1) - patch_y)
        rect = patches.Rectangle(
            (patch_x, patch_y),
            pixels_to_include,
            width,
            fill=False,
            color="green",
        )
        self.axes.add_patch(rect)

        # also display foreground/background regions
        # background regions
        self.l_lbkg = self.axes.axhline(color="black")  # the vert line
        self.l_lbkg.set_ydata(low_bkg - 0.5)

        self.l_hbkg = self.axes.axhline(color="black")  # the vert line
        self.l_hbkg.set_ydata(high_bkg + 0.5)

        # beam centre
        self.l_bc = self.axes.axhline(color="red")  # the vert line
        self.l_bc.set_ydata(beam_centre)

        # foreground regions
        self.l_lfore = self.axes.axhline(color="blue")  # the vert line
        self.l_lfore.set_ydata(low_px - 0.5)

        self.l_hfore = self.axes.axhline(color="blue")  # the vert line
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
        super().__init__(self.figure)
        self.setParent(parent)

        # information for dragging a line
        self._dragging = False

        # values for foreground and background regions
        self._low_px = 500
        self._high_px = 505
        self._low_bkg = 490
        self._high_bkg = 515
        self.line_attrs = {}

        self.axes = self.figure.add_axes([0.1, 0.07, 0.95, 0.94])
        self.axes.margins(0.0005)

        FigureCanvas.setSizePolicy(
            self,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        FigureCanvas.updateGeometry(self)
        self.connect()

    def display_cross_section(
        self, x, xs, beam_centre, low_px, high_px, low_bkg, high_bkg
    ):
        self._beam_centre = beam_centre
        self._low_px = low_px
        self._high_px = high_px
        self._low_bkg = low_bkg
        self._high_bkg = high_bkg

        self.axes.clear()
        self.axes.plot(x, xs)
        self.axes.set_xlim(np.min(x), np.max(x))

        # background regions
        self.l_lbkg = self.axes.axvline(color="black")  # the vert line
        self.l_lbkg.set_xdata(low_bkg)

        self.l_hbkg = self.axes.axvline(color="black")  # the vert line
        self.l_hbkg.set_xdata(high_bkg)

        # beam centre
        self.l_bc = self.axes.axvline(color="red")  # the vert line
        self.l_bc.set_xdata(beam_centre)

        # foreground regions
        self.l_lfore = self.axes.axvline(color="blue")  # the vert line
        self.l_lfore.set_xdata(low_px)

        self.l_hfore = self.axes.axvline(color="blue")  # the vert line
        self.l_hfore.set_xdata(high_px)

        # text location in axes coords
        self.txt = self.axes.text(0.6, 0.9, "", transform=self.axes.transAxes)

        # these are the lines that are draggable
        self.line_attrs = {
            "_low_bkg": self.l_lbkg,
            "_high_bkg": self.l_hbkg,
            "_low_px": self.l_lfore,
            "_high_px": self.l_hfore,
        }
        self.draw()

    def connect(self):
        """
        connect to all the events we need
        """
        self.cidpress = self.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = self.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def on_press(self, event):
        if not event.inaxes:
            return
        found = False

        # could also be in self.axes.get_lines()
        for attr in self.line_attrs:
            if self.line_attrs[attr].contains(event)[0]:
                found = attr, self.line_attrs[attr]
        if not found:
            return

        self._dragging = True
        self._press = [found, found[1].get_xdata(), event.xdata, event.ydata]

    def on_release(self, event):
        if not event.inaxes:
            return

        self._dragging = False

    def on_motion(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        self.txt.set_text("x=%1.2f, y=%1.2f" % (x, y))

        if self._dragging:
            found, loc, xpress, ypress = self._press
            attr, line = found
            dx = x - xpress
            # dy = y - ypress
            new_loc = int(np.round(loc + dx))

            # TODO make sure lopx and high px cant cross
            # TODO recalc backgrounds and beam centre after plot_button release
            # TODO make sure background can't cross foreground
            # TODO connect Manual_beam_finder object to listen to changes
            # (on release)
            line.set_xdata(new_loc)
            setattr(self, attr, new_loc)

        self.draw()


class NavToolBar(NavigationToolbar):
    """
    Toolbar for the detector image
    """

    toolitems = [
        ("Home", "Reset original view", "home", "home"),
        ("Back", "Back to previous view", "back", "back"),
        ("Forward", "Forward to next view", "forward", "forward"),
        ("Pan", "Pan axes with left mouse, zoom with right", "move", "pan"),
        ("Zoom", "Zoom to rectangle", "zoom_to_rect", "zoom"),
    ]

    def __init__(self, canvas, parent, coordinates=True):
        NavigationToolbar.__init__(self, canvas, parent, coordinates)
        self.setIconSize(QtCore.QSize(15, 15))
