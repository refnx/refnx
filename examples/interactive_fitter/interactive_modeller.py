import numpy as np
import ipywidgets as widgets
from IPython.display import (DisplayHandle, clear_output)
import time
import traitlets
from traitlets import HasTraits
import matplotlib.pyplot as plt

from refnx.reflect import Slab, ReflectModel
from refnx.dataset import ReflectDataset
from refnx.analysis import Objective, CurveFitter
from refnx._lib import flatten


class ReflectModelView(HasTraits):
    # traitlet to say when params were last altered
    view_changed = traitlets.Float(time.time())

    # traitlet to ask when a redraw of the GUI is requested.
    view_redraw = traitlets.Float(time.time())

    def __init__(self, reflect_model):
        super(ReflectModelView, self).__init__()

        self.model = reflect_model
        self.structure_view = StructureView(self.model.structure)
        self.last_selected_param = None
        self.param_widgets_link = {}

        slab_views = self.structure_view.slab_views
        slab_views[0].w_thick.disabled = True
        slab_views[0].c_thick.disabled = True
        slab_views[0].w_rough.disabled = True
        slab_views[0].c_rough.disabled = True
        slab_views[-1].w_thick.disabled = True
        slab_views[-1].c_thick.disabled = True

        # got to listen to all the slab views
        for slab_view in slab_views:
            slab_view.observe(self.slab_params_changed, names=['view_changed'])

        # if you'd like to change the number of layers
        self.w_layers = widgets.BoundedIntText(
            description='Number of layers',
            value=len(slab_views) - 2, min=0, max=1000,
            style={'description_width': '120px'},
            continuous_update=False)

        self.w_layers.observe(self.change_layers, names=['value'])

        # where you're going to add/remove layers
        self._varying_layers = False
        self._location = None
        self.ok_button = None
        self.cancel_button = None

        # associated with ReflectModel
        p = reflect_model.scale
        self.w_scale = widgets.FloatText(value=p.value,
                                         description='scale', step=0.01,
                                         style={'description_width': '120px'})
        self.c_scale = widgets.Checkbox(value=p.vary)
        self.scale_low_limit = widgets.FloatText(value=p.bounds.lb, step=0.01)
        self.scale_hi_limit = widgets.FloatText(value=p.bounds.ub, step=0.01)

        p = reflect_model.bkg
        self.w_bkg = widgets.FloatText(value=p.value,
                                       description='background', step=1e-7,
                                       style={'description_width': '120px'})
        self.c_bkg = widgets.Checkbox(value=reflect_model.bkg.vary)
        self.bkg_low_limit = widgets.FloatText(p.bounds.lb, step=1e-8)
        self.bkg_hi_limit = widgets.FloatText(value=p.bounds.ub, step=1e-7)

        p = reflect_model.dq
        self.w_dq = widgets.BoundedFloatText(value=p.value,
                                             description='dq/q', step=0.1,
                                             min=0, max=20.)
        self.c_dq = widgets.Checkbox(value=reflect_model.dq.vary)
        self.dq_low_limit = widgets.BoundedFloatText(value=p.bounds.lb, min=0, max=20,
                                                     step=0.1)
        self.dq_hi_limit = widgets.BoundedFloatText(value=p.bounds.ub, min=0, max=20,
                                                    step=0.1)

        self.c_scale.style.description_width = '0px'
        self.c_bkg.style.description_width = '0px'
        self.c_dq.style.description_width = '0px'
        self.do_fit_button = widgets.Button(description='Do Fit')

        self.widget_list = [self.w_scale, self.c_scale, self.w_bkg,
                            self.c_bkg, self.w_dq, self.c_dq]

        self.limits_widgets_list = [self.scale_low_limit, self.scale_hi_limit,
                                    self.bkg_low_limit, self.bkg_hi_limit,
                                    self.dq_low_limit, self.dq_hi_limit]

        for widget in self.widget_list:
            widget.observe(self.model_params_changed, names=['value'])

        for widget in self.limits_widgets_list:
            widget.observe(self.model_limits_changed, names=['value'])

        # button to create default limits
        self.default_limits_button = widgets.Button(description='Set default limits')
        self.default_limits_button.on_click(self.default_limits)

        # widgets for easy model change
        self.model_slider = widgets.FloatSlider()
        self.model_slider.layout = widgets.Layout(width='100%')
        self.model_slider_link = None
        self.model_slider_min = widgets.FloatText()
        self.model_slider_min.layout = widgets.Layout(width='10%')
        self.model_slider_max = widgets.FloatText()
        self.model_slider_max.layout = widgets.Layout(width='10%')
        self.model_slider_max.observe(self.change_slider_limits,
                                      names=['value'])
        self.model_slider_min.observe(self.change_slider_limits,
                                      names=['value'])
        self.last_selected_param = None

        self.link_param_widgets()

    def change_slider_limits(self, change):
        self.model_slider.max = self.model_slider_max.value
        self.model_slider.min = self.model_slider_min.value
        self.model_slider.step = (self.model_slider.max -
                                  self.model_slider.min) / 1000.

    def model_params_changed(self, change):
        d = self.param_widgets_link

        for par in [self.model.scale, self.model.bkg, self.model.dq]:
            idx = id(par)
            wids = d[idx]

            if change['owner'] in wids:
                loc = wids.index(change['owner'])
                if loc == 0:
                    par.value = wids[0].value

                    # this captures when the user starts modifying a different parameter
                    self.possibly_change_slider(change['owner'])

                    self.view_changed = time.time()
                    break
                elif loc == 1:
                    par.vary = wids[1].value
                    # need to rebuild the limit widgets, achieved by redrawing box
                    self.view_redraw = time.time()
                    break
                else:
                    return

        # this captures when the user starts modifying a different parameter
        self.possibly_change_slider(change['owner'])

    def model_limits_changed(self, change):
        d = self.param_widgets_link
        for par in [self.model.scale, self.model.bkg, self.model.dq]:
            idx = id(par)
            wids = d[idx]

            if change['owner'] in wids:
                loc = wids.index(change['owner'])
                if loc == 2:
                    par.bounds.lb = wids[2].value
                    break
                elif loc == 3:
                    par.bounds.ub = wids[3].value
                    break

    def default_limits(self, change):
        varying_parameters = self.model.parameters.varying_parameters()

        for par in varying_parameters:
            par.bounds.lb = min(0, 2 * par.value)
            par.bounds.ub = max(0, 2 * par.value)

        self.refresh()

    def slab_params_changed(self, change):
        # this captures when the user starts modifying a different parameter
        self.possibly_change_slider(change['owner'].param_being_varied)
        if isinstance(change['owner'].param_being_varied, widgets.Checkbox):
            # need to rebuild the limit widgets, achieved by redrawing box
            self.view_redraw = time.time()
        else:
            self.view_changed = time.time()

    def possibly_change_slider(self, change_owner):
        if (change_owner is not self.last_selected_param):
            self.last_selected_param = change_owner
            if self.model_slider_link is not None:
                self.model_slider_link.unlink()
            self.model_slider_link = widgets.link(
                (self.last_selected_param, 'value'),
                (self.model_slider, 'value'))
            self.model_slider_max.value = max(0, 2. * self.last_selected_param.value)
            self.model_slider_min.value = min(0, 2. * self.last_selected_param.value)

    def change_layers(self, change):
        self.ok_button = widgets.Button(description="OK")
        if change['new'] > change['old']:
            description = 'Insert before which layer?'
            min_loc = 1
            max_loc = len(self.model.structure) - 2 + 1
            self.ok_button.on_click(self.increase_layers)
        elif change['new'] < change['old']:
            min_loc = 1
            max_loc = (len(self.model.structure) - 2 -
                       (change['old'] - change['new']) + 1)
            description = 'Remove from which layer?'
            self.ok_button.on_click(self.decrease_layers)
        else:
            return
        self._varying_layers = True
        self.w_layers.disabled = True
        self.do_fit_button.disabled = True
        self._location = widgets.BoundedIntText(
            value=min_loc,
            description=description,
            min=min_loc, max=max_loc,
            style={'description_width': 'initial'})
        self.cancel_button = widgets.Button(description="Cancel")
        self.cancel_button.on_click(self.cancel_layers)
        self.view_redraw = time.time()

    def increase_layers(self, b):
        self.w_layers.disabled = False
        self.do_fit_button.disabled = False

        how_many = self.w_layers.value - (len(self.model.structure) - 2)
        loc = self._location.value

        for i in range(how_many):
            slab = Slab(0, 0, 3)
            slab.thick.bounds = (0, 2 * slab.thick.value)
            slab.sld.real.bounds = (0, 2 * slab.sld.real.value)
            slab.sld.imag.bounds = (0, 2 * slab.sld.imag.value)
            slab.rough.bounds = (0, 2 * slab.rough.value)

            slab_view = SlabView(slab)
            self.model.structure.insert(loc, slab)
            self.structure_view.slab_views.insert(loc, slab_view)
            slab_view.observe(self.slab_params_changed)

        rename_params(self.model.structure)
        self._varying_layers = False
        self.view_redraw = time.time()

    def decrease_layers(self, b):
        self.w_layers.disabled = False
        self.do_fit_button.disabled = False

        loc = self._location.value
        how_many = len(self.model.structure) - 2 - self.w_layers.value
        for i in range(how_many):
            self.model.structure.pop(loc)
            slab_view = self.structure_view.slab_views.pop(loc)
            slab_view.unobserve_all()

        rename_params(self.model.structure)
        self._varying_layers = False
        self.view_redraw = time.time()

    def link_param_widgets(self):
        # link parameters to widgets (value, checkbox,
        #                             upperlim, lowerlim)
        self.param_widgets_link = {}
        d = self.param_widgets_link
        model = self.model

        d[id(model.scale)] = (self.w_scale, self.c_scale,
                              self.scale_low_limit, self.scale_hi_limit)
        d[id(model.bkg)] = (self.w_bkg, self.c_bkg,
                            self.bkg_low_limit, self.bkg_hi_limit)
        d[id(model.dq)] = (self.w_dq, self.c_dq,
                           self.dq_low_limit, self.dq_hi_limit)

    def cancel_layers(self, b):
        # disable the change layers widget to prevent recursion
        self.w_layers.unobserve(self.change_layers, names='value')
        self.w_layers.value = len(self.model.structure) - 2
        self.w_layers.observe(self.change_layers, names='value')
        self.w_layers.disabled = False
        self.do_fit_button.disabled = False

        self._varying_layers = False
        self.view_redraw = time.time()

    def refresh(self):
        for par in [self.model.scale, self.model.bkg, self.model.dq]:
            wid = self.param_widgets_link[id(par)]
            wid[0].value = par.value
            wid[1].value = par.vary
            wid[2].value = par.bounds.lb
            wid[3].value = par.bounds.ub

        slab_views = self.structure_view.slab_views

        for slab_view in slab_views:
            slab_view.refresh()

    @property
    def model_box(self):
        output = [self.w_layers,
                  widgets.HBox([self.w_scale, self.c_scale,
                                self.w_dq, self.c_dq]),
                  widgets.HBox([self.w_bkg, self.c_bkg]),
                  self.structure_view.box,
                  widgets.HBox([self.model_slider_min,
                                self.model_slider,
                                self.model_slider_max])]

        if self._varying_layers:
            output.append(widgets.HBox([self._location,
                                        self.ok_button,
                                        self.cancel_button]))

        output.append(self.do_fit_button)

        return widgets.VBox(output)

    @property
    def limits_box(self):
        varying_pars = self.model.parameters.varying_parameters()
        hboxes = [self.default_limits_button]

        d = {}
        d.update(self.param_widgets_link)

        slab_views = self.structure_view.slab_views
        for slab_view in slab_views:
            d.update(slab_view.param_widgets_link)

        for par in varying_pars:
            name = widgets.Text(par.name)
            name.disabled = True

            val, check, ll, ul = d[id(par)]

            hbox = widgets.HBox([name, ll, val, ul])
            hboxes.append(hbox)

        return widgets.VBox(hboxes)


class StructureView(object):
    def __init__(self, structure):
        self.structure = structure
        self.slab_views = [SlabView(slab) for slab in structure]

    @property
    def box(self):
        layout = widgets.Layout(flex='1 1 auto', width='auto')
        label_row = widgets.HBox([widgets.HTML('thick', layout=layout),
                                  widgets.HTML('sld', layout=layout),
                                  widgets.HTML('isld', layout=layout),
                                  widgets.HTML('rough', layout=layout)])

        hboxes = [label_row]
        hboxes.extend([view.box for view in self.slab_views])
        # add in layer numbers
        self.slab_views[0].w_thick.description = 'fronting'
        self.slab_views[-1].w_thick.description = 'backing'
        for i in range(1, len(self.slab_views) - 1):
            self.slab_views[i].w_thick.description = str(i)

        return widgets.VBox(hboxes)


class SlabView(HasTraits):
    # traitlet to say when params were last altered
    view_changed = traitlets.Float(time.time())

    def __init__(self, slab):
        self.slab = slab
        self.param_widgets_link = {}

        self.param_being_varied = None

        p = slab.thick
        self.w_thick = widgets.FloatText(value=p.value, step=1)
        self.c_thick = widgets.Checkbox(value=p.vary)
        self.thick_low_limit = widgets.FloatText(value=p.bounds.lb, step=1)
        self.thick_hi_limit = widgets.FloatText(value=p.bounds.ub,
                                                step=1)

        p = slab.sld.real
        self.w_sld = widgets.FloatText(value=p.value, step=0.01)
        self.c_sld = widgets.Checkbox(value=p.vary)
        self.sld_low_limit = widgets.FloatText(value=p.bounds.lb,
                                               step=0.01)
        self.sld_hi_limit = widgets.FloatText(value=p.bounds.ub,
                                              step=0.01)
        p = slab.sld.imag
        self.w_isld = widgets.FloatText(value=p.value, step=0.01)
        self.c_isld = widgets.Checkbox(value=p.vary)
        self.isld_low_limit = widgets.FloatText(value=p.bounds.lb, step=0.01)
        self.isld_hi_limit = widgets.FloatText(value=p.bounds.ub,
                                               step=0.01)

        p = slab.rough
        self.w_rough = widgets.FloatText(value=p, step=1)
        self.c_rough = widgets.Checkbox(value=p.vary)
        self.rough_low_limit = widgets.FloatText(p.bounds.lb, step=0.01)
        self.rough_hi_limit = widgets.FloatText(value=p.bounds.ub,
                                                step=0.01)

        self.widget_list = [self.w_thick, self.c_thick, self.w_sld,
                            self.c_sld, self.w_isld, self.c_isld,
                            self.w_rough, self.c_rough]
        self.limits_list = [self.thick_low_limit, self.thick_hi_limit,
                            self.sld_low_limit, self.sld_hi_limit,
                            self.isld_low_limit, self.isld_hi_limit,
                            self.rough_low_limit, self.rough_hi_limit]

        for widget in self.widget_list:
            widget.style.description_width = '0px'
            widget.observe(self.handle_slab_params_change, names='value')
        self.w_thick.style.description_width = '50px'

        for widget in self.limits_list:
            widget.observe(self.handle_slab_limits_change, names='value')

        self.link_param_widgets()

    def handle_slab_params_change(self, change):
        d = self.param_widgets_link
        slab = self.slab

        for par in flatten(slab.parameters):
            if id(par) in d and change['owner'] in d[id(par)]:
                wids = d[id(par)]
                loc = wids.index(change['owner'])
                if loc == 0:
                    par.value = wids[0].value
                    break
                elif loc == 1:
                    par.vary = wids[1].value
                    break

        self.param_being_varied = change['owner']
        self.view_changed = time.time()

    def handle_slab_limits_change(self, change):
        slab = self.slab
        d = self.param_widgets_link

        for par in flatten(slab.parameters):
            if id(par) in d and change['owner'] in d[id(par)]:
                wids = d[id(par)]
                loc = wids.index(change['owner'])
                if loc == 2:
                    par.bounds.lb = wids[loc].value
                    break
                elif loc == 3:
                    par.bounds.ub = wids[loc].value
                    break
                else:
                    return

    def link_param_widgets(self):
        # link parameters to widgets (value, checkbox,
        #                             upperlim, lowerlim)
        d = self.param_widgets_link

        d[id(self.slab.thick)] = (self.w_thick,
                                  self.c_thick,
                                  self.thick_low_limit,
                                  self.thick_hi_limit)
        d[id(self.slab.sld.real)] = (self.w_sld,
                                     self.c_sld,
                                     self.sld_low_limit,
                                     self.sld_hi_limit)
        d[id(self.slab.sld.imag)] = (self.w_isld,
                                     self.c_isld,
                                     self.isld_low_limit,
                                     self.isld_hi_limit)

        d[id(self.slab.rough)] = (self.w_rough,
                                  self.c_rough,
                                  self.rough_low_limit,
                                  self.rough_hi_limit)

    def refresh(self):
        # if the underlying slab parameters have changed, then the widgets need to be updated.
        d = self.param_widgets_link

        ids = {id(p):p for p in flatten(self.slab.parameters) if id(p) in d}
        for idx, par in ids.items():
            widgets = d[idx]
            widgets[0].value = par.value
            widgets[1].value = par.vary
            widgets[2].value = par.bounds.lb
            widgets[3].value = par.bounds.ub

    @property
    def box(self):
        return widgets.HBox(self.widget_list)


class Motofit(HasTraits):
    # id for the output instance
    display_id = traitlets.Instance(klass=DisplayHandle, args=())

    def __init__(self):

        self.tab = widgets.Tab()

        # attributes for the graph
        # for the graph
        self.qmin = 0.001
        self.qmax = 0.5
        self.qpnt = 1000
        self.fig = None
        self.ax_data = None
        self.ax_sld = None

        # attributes for a user dataset
        self.dataset = None
        self.objective = None
        self.curvefitter = None
        self.data_plot = None
        self.data_plot_sld = None

        self.dataset_name = widgets.Text(description='dataset:')
        self.dataset_name.disabled = True
        self.chisqr = widgets.FloatText(description='chi-squared:')
        self.chisqr.disabled = True

        # fronting
        slab0 = Slab(0, 0, 0)
        slab1 = Slab(25, 3.47, 3)
        slab2 = Slab(0, 2.07, 3)

        structure = slab0 | slab1 | slab2
        rename_params(structure)
        self.model = ReflectModel(structure)
        structure = slab0 | slab1 | slab2
        self.model = ReflectModel(structure)

        # give some default parameter limits
        self.model.scale.bounds = (0.1, 2)
        self.model.bkg.bounds = (1e-8, 2e-5)
        self.model.dq.bounds = (0, 20)
        for slab in self.model.structure:
            slab.thick.bounds = (0, 2 * slab.thick.value)
            slab.sld.real.bounds = (0, 2 * slab.sld.real.value)
            slab.sld.imag.bounds = (0, 2 * slab.sld.imag.value)
            slab.rough.bounds = (0, 2 * slab.rough.value)

        self.model_view = ReflectModelView(self.model)
        self.model_view.observe(self.update_model, names=['view_changed'])
        self.model_view.observe(self.redraw, names=['view_redraw'])
        self.model_view.do_fit_button.on_click(self.do_fit)

        self.tab.set_title(0, 'Model')
        self.tab.set_title(1, 'Limits')
        self.tab.observe(self.tab_changed, names='selected_index')

        # an output area for messages.
        self.output = widgets.Textarea()
        self.output.layout = widgets.Layout(width='100%', height='200px')


        self.display_box = widgets.VBox()

    def set_model(self, reflect_model):
        if self.model_view is not None:
            self.model_view.unobserve_all()

        self.model = reflect_model
        self.model_view = ReflectModelView(self.model)
        self.model_view.observe(self.update_model, names=['view_changed'])
        self.model_view.observe(self.redraw, names=['view_redraw'])
        self.model_view.do_fit_button.on_click(self.do_fit)

        self.update_analysis_objects()

        self.redraw(None)

    def update_model(self, change):
        q = np.linspace(self.qmin, self.qmax, self.qpnt)
        theoretical = self.model.model(q)
        sld_profile = self.model.structure.sld_profile()
        z, sld = sld_profile
        if self.theoretical_plot is not None:
            self.theoretical_plot.set_xdata(q)
            self.theoretical_plot.set_ydata(theoretical)

            self.theoretical_plot_sld.set_xdata(z)
            self.theoretical_plot_sld.set_ydata(sld)
            self.ax_sld.relim()
            self.ax_sld.autoscale_view()
            self.fig.canvas.draw()

        if self.dataset is not None:
            self.chisqr.value = self.objective.chisqr()

    def update_analysis_objects(self):
        self.objective = Objective(self.model, self.dataset)

    def __call__(self, data=None):
        # the theoretical model
        # display the main graph
        self.fig = plt.figure(figsize=(9, 5))
        self.ax_data = self.fig.add_subplot(121)
        self.ax_sld = self.fig.add_subplot(122)
        self.fig.tight_layout()

        q = np.linspace(self.qmin, self.qmax, self.qpnt)
        self.theoretical_plot = self.ax_data.plot(q, self.model.model(q))[0]
        self.ax_data.set_yscale('log')

        z, sld = self.model.structure.sld_profile()
        self.theoretical_plot_sld = self.ax_sld.plot(z, sld)[0]

        if data is not None:
            self.load_data(data)

        self.update_display_box(self.display_box)
        return self.display_box

    def load_data(self, data):
        self.dataset = ReflectDataset(data)
        self.dataset_name.value = self.dataset.name

        self.update_analysis_objects()

        self.qmin = np.min(self.dataset.x)
        self.qmax = np.max(self.dataset.x)
        if self.fig is not None:
            if self.data_plot is None:
                self.data_plot, = self.ax_data.plot(self.dataset.x,
                                                    self.dataset.y,
                                                    label=self.dataset.name,
                                                    ms=4,
                                                    marker='o', ls='')
            else:
                self.data_plot.set_xdata(self.dataset.x)
                self.data_plot.set_ydata(self.dataset.y)

            # calculate theoretical model over same range as data
            # use redraw over update_model because it ensures chi2 widget gets
            # displayed
            self.redraw(None)
            self.ax_data.relim()
            self.ax_data.autoscale_view()
            self.fig.canvas.draw()

    def redraw(self, change):
        self.update_display_box(self.display_box)
        self.update_model(None)

    def do_fit(self, change):
        if self.dataset is None:
            return

        if not self.model.parameters.varying_parameters():
            self.output.value = "No parameters are being varied"
            return

        self.curvefitter = CurveFitter(self.objective)

        def callback(xk, convergence):
            self.chisqr.value = self.objective.chisqr(xk)

        res = self.curvefitter.fit('differential_evolution', callback=callback)

        # place before set_model, because a redraw is required to stop the
        # output from getting too long.
        self.output.value = repr(self.objective)

        # need to update the widgets as the model will be updated.
        # this also redraws GUI.
        # self.model_view.refresh()
        self.set_model(self.model)

    def tab_changed(self, change):
        pass

    def update_display_box(self, box):

        vbox_widgets = []

        if self.dataset is not None:
            vbox_widgets.append(widgets.HBox([self.dataset_name, self.chisqr]))

        self.tab.children = [self.model_view.model_box,
                             self.model_view.limits_box]
        vbox_widgets.append(self.tab)
        vbox_widgets.append(self.output)
        box.children = tuple(vbox_widgets)


def rename_params(structure):
    for i in range(1, len(structure) - 1):
        structure[i].thick.name = '%d - thick' % i
        structure[i].sld.real.name = '%d - sld' % i
        structure[i].sld.imag.name = '%d - isld' % i
        structure[i].rough.name = '%d - rough' % i

    structure[0].sld.real.name = 'sld - fronting'
    structure[0].sld.imag.name = 'isld - fronting'
    structure[-1].sld.real.name = 'sld - backing'
    structure[-1].sld.imag.name = 'isld - backing'
    structure[-1].rough.name = 'rough - backing'
