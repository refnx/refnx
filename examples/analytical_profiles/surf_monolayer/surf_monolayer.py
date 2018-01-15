"""
Analytic profile for studying surfactant/lipid monolayers at an interface
"""

from refnx.analysis import Parameter, possibly_create_parameter
from refnx.reflect import SLD, Component


class SurfMono(Component):
    def __init__(self, head_scat_lens, tail_scat_lens, sub_phase_sld, super_phase_sld, guess_head_tail_thickness, 
                 guess_apm, name='', tail_volume=0):
        """
        A surfactant/lipid monolayer class that constrains the number density of the heads and tails to be the same
        both within a given contrast and across a series of contrasts.

        Parameters
        ----------
        head_scat_lens: array-like
            the scattering lengths of the head groups of the different species contrasts
        tail_scat_lens: array-like
            the scattering lengths of the tail groups of the different species contrasts
        sub_phase_sld: array-like
            the SLD of the subphase for each of the contrasts
        super_phase_sld: array-like
            the SLD of the superphase for each of the contrasts
        guess_head_tail_thickness: float[2]
            the head[0] and the tail[1] estimated thicknesses
        guess_apm: float
            an estimate of the system area per molecule
        name : str
            Name of this lipid component
        tail_volume: float (optional)
            If the volume of the tail is known, e.g. from MD simulations this can be assigned
        """
        # TODO allow for mixed monolayers
        # TODO allow the inclusion of proteins
        super(SurfMono, self).__init__()
        n = len(head_scat_lens)
        # Check that the right number of heads and tail scattering lengths and sub and super phase SLDs
        # are all the same.
        if any(len(x) != n for x in [head_scat_lens, tail_scat_lens, sub_phase_sld, super_phase_sld]):
            raise ValueError("The number of different contrasts is inconsistent!")
        # Check that there are two guess values given, one for the heads and one for the tails.
        if len(guess_head_tail_thickness) != 2:
            raise ValueError("Both the head and tail layer thicknesses should be estimated.")
        self.numberofcontrasts = len(head_scat_lens)
        self.head_scat_lens = {}
        self.tail_scat_lens = {}
        self.sub_phase_sld = {}
        self.super_phase_sld = {}
        # Assigning the scattering lengths and SLDs to the appropriate dictionary.
        for i in range(0, self.numberofcontrasts):
            self.head_scat_lens['con%s' % i] = head_scat_lens[i]
            self.tail_scat_lens['con%s' % i] = tail_scat_lens[i]
            self.sub_phase_sld['con%s' % i] = sub_phase_sld[i]
            self.super_phase_sld['con%s' % i] = super_phase_sld[i]
        self.name = name
        self.head_thick = possibly_create_parameter(guess_head_tail_thickness[0], 
                                                    name='%s - head layer thickness' % name)
        self.tail_thick = possibly_create_parameter(guess_head_tail_thickness[1], 
                                                    name='%s - tail layer thickness' % name)
        self.guess_apm = guess_apm
        self.structures = {}
        self.tails = {}
        self.heads = {}
        self.subs = {}
        self.supers = {}
        self.head_rough = 0
        self.tail_rough = 0
        # Defined as that for water, but can be changed when called.
        self.water_rough = Parameter(3.1, 'subphase_layer_rough')
        self.head_layers = {}
        self.tail_layers = {}
        self.sub_layers = {}
        self.tail_volume = tail_volume

    # This determines an initial tail SLD value based on the guess APM, and tail thickness
    def guess_sld(self):
        a = self.tail_scat_lens['con0'] / (self.tail_thick.value * self.guess_apm) * 1E6
        return a

    # Generates parameters associated with each of the required contrasts and assigns them to
    # appropriate SLD objects.
    def set_sld(self):
        tail_slds = {}
        head_slds = {}
        sub_slds = {}
        super_slds = {}
        if self.tail_volume == 0:
            tail_slds['con0'] = Parameter(self.guess_sld(), 'tail_layer_contrast0',
                                          bounds=(self.guess_sld() - (0.5 * self.guess_sld()),
                                                  self.guess_sld() + (0.5 * self.guess_sld())), vary=True)
        else:
            a = self.tail_scat_lens['con0'] / self.tail_volume * 1E6
            tail_slds['con0'] = Parameter(a, 'tail_layer_contrast0',
                                          bounds=(a - (0.25 * a), a + (0.25 * a)), vary=True)
        head_slds['con0'] = Parameter(1., 'head_layer_contrast0')
        for i in range(1, self.numberofcontrasts):
            tail_slds['con%s' % i] = Parameter(1, 'tail_layer_contrast%s' % i)
            head_slds['con%s' % i] = Parameter(1, 'head_layer_contrast%s' % i)
            sub_slds['con%s' % i] = Parameter(self.sub_phase_sld['con%s' % i], 'sub%s' % i)
            super_slds['con%s' % i] = Parameter(self.super_phase_sld['con%s' % i], 'super%s' % i)
        for i in range(0, self.numberofcontrasts):
            self.tails['con%s' % i] = SLD(tail_slds['con%s' % i], name='tail_contrast%s' % i)
            self.heads['con%s' % i] = SLD(head_slds['con%s' % i], name='head_contrast%s' % i)
            self.subs['con%s' % i] = SLD(self.sub_phase_sld['con%s' % i], name='sub_contrast%s' % i)
            self.supers['con%s' % i] = SLD(self.super_phase_sld['con%s' % i], name='super_contrast%s' % i)

    # Sets the roughnesses of the head and tail layers.
    def set_thick_rough(self):
        self.head_thick.setp(bounds=(self.head_thick.value - (0.25 * self.head_thick.value),
                                     self.head_thick.value + (0.25 * self.head_thick.value)), vary=True)
        self.tail_thick.setp(bounds=(self.tail_thick.value - (0.25 * self.tail_thick.value),
                                     self.tail_thick.value + (0.25 * self.tail_thick.value)), vary=True)
        self.head_rough = Parameter(0.2 * self.head_thick.value, 'head_layer_rough',
                                    bounds=(0, 0.5 * self.head_thick.value), vary=True)
        self.tail_rough = Parameter(0.2 * self.tail_thick.value, 'tail_layer_rough',
                                    bounds=(0, 0.5 * self.tail_thick.value), vary=True)

    # Loops through each contrast and assigned the head, tail, and subphase layers.
    def set_layers(self):
        for i in range(0, self.numberofcontrasts):
            self.head_layers['con%s' % i] = self.heads['con%s' % i](self.head_thick, self.head_rough)
            self.tail_layers['con%s' % i] = self.tails['con%s' % i](self.tail_thick, self.tail_rough)
            self.sub_layers['con%s' % i] = self.subs['con%s' % i](0., self.water_rough)

    # Here the constraints are set, these result in the same number of heads and tails across
    # all contrasts in the analysis.
    def set_constraints(self):
        vguess = 1 - (self.head_layers['con0'].sld.real.value * self.head_thick.value) / \
                     (self.head_scat_lens['con0'] * self.guess_apm)
        self.head_layers['con0'].vfsolv.setp(vguess, bounds=(0., 0.999999), vary=True)
        for i in range(1, self.numberofcontrasts):
            self.head_layers['con%s' % i].vfsolv.constraint = self.head_layers['con0'].vfsolv
        if self.numberofcontrasts == 1:
            self.head_layers['con0'].sld.real.constraint = \
                (self.tail_layers['con0'].sld.real * self.tail_thick * self.head_scat_lens['con0']) / \
                (self.head_thick * self.tail_scat_lens['con0'] * (1 - self.head_layers['con0'].vfsolv))
        if self.numberofcontrasts == 2:
            self.head_layers['con0'].sld.real.constraint = \
                (self.tail_layers['con0'].sld.real * self.tail_thick * self.head_scat_lens['con0']) / \
                (self.head_thick * self.tail_scat_lens['con0'] * (1 - self.head_layers['con0'].vfsolv))
            self.head_layers['con1'].sld.real.constraint = \
                (self.head_layers['con0'].sld.real * self.head_scat_lens['con1']) / \
                (self.head_scat_lens['con0'])
            self.tail_layers['con1'].sld.real.constraint = \
                (self.head_layers['con1'].sld.real * self.head_thick * self.tail_scat_lens['con1'] *
                 (1 - self.head_layers['con1'].vfsolv)) / (self.tail_thick * self.head_scat_lens['con1'])
        if self.numberofcontrasts > 2:
            if self.numberofcontrasts % 2 != 0:
                for i in range(0, self.numberofcontrasts - 2, 2):
                    a = str(i)
                    b = str(i + 1)
                    c = str(i + 2)
                    self.head_layers['con%s' % a].sld.real.constraint = \
                        (self.tail_layers['con%s' % a].sld.real * self.tail_thick *
                         self.head_scat_lens['con%s' % a]) / (self.head_thick *
                                                              self.tail_scat_lens['con%s' % a] *
                                                              (1 - self.head_layers['con%s' % a].vfsolv))
                    self.head_layers['con%s' % b].sld.real.constraint = \
                        (self.head_layers['con%s' % a].sld.real * self.head_scat_lens['con%s' % b]) / \
                        (self.head_scat_lens['con%s' % a])
                    self.tail_layers['con%s' % b].sld.real.constraint = \
                        (self.head_layers['con%s' % b].sld.real * self.head_thick *
                         self.tail_scat_lens['con%s' % b] * (1 - self.head_layers['con%s' % b].vfsolv)) / \
                        (self.tail_thick * self.head_scat_lens['con%s' % b])
                    self.tail_layers['con%s' % c].sld.real.constraint = \
                        (self.tail_layers['con%s' % b].sld.real * self.tail_scat_lens['con%s' % c]) / \
                        (self.tail_scat_lens['con%s' % b])
                self.head_layers['con%s' % str(self.numberofcontrasts-1)].sld.real.constraint = \
                    (self.tail_layers['con%s' % str(self.numberofcontrasts-1)].sld.real * self.tail_thick *
                     self.head_scat_lens['con%s' % str(self.numberofcontrasts-1)]) / \
                    (self.head_thick * self.tail_scat_lens['con%s' % str(self.numberofcontrasts-1)] *
                     (1 - self.head_layers['con%s' % str(self.numberofcontrasts-1)].vfsolv))
            else:
                for i in range(0, self.numberofcontrasts - 2, 2):
                    a = str(i)
                    b = str(i + 1)
                    c = str(i + 2)
                    self.head_layers['con%s' % a].sld.real.constraint = \
                        (self.tail_layers['con%s' % a].sld.real * self.tail_thick *
                         self.head_scat_lens['con%s' % a]) / (self.head_thick *
                                                              self.tail_scat_lens['con%s' % a] *
                                                              (1 - self.head_layers['con%s' % a].vfsolv))
                    self.head_layers['con%s' % b].sld.real.constraint = \
                        (self.head_layers['con%s' % a].sld.real * self.head_scat_lens['con%s' % b]) / \
                        (self.head_scat_lens['con%s' % a])
                    self.tail_layers['con%s' % b].sld.real.constraint = \
                        (self.head_layers['con%s' % b].sld.real * self.head_thick *
                         self.tail_scat_lens['con%s' % b] * (1 - self.head_layers['con%s' % b].vfsolv)) / \
                        (self.tail_thick * self.head_scat_lens['con%s' % b])
                    self.tail_layers['con%s' % c].sld.real.constraint = \
                        (self.tail_layers['con%s' % b].sld.real * self.tail_scat_lens['con%s' % c]) / \
                        (self.tail_scat_lens['con%s' % b])
                self.head_layers['con%s' % str(self.numberofcontrasts-2)].sld.real.constraint = \
                    (self.tail_layers['con%s' % str(self.numberofcontrasts-2)].sld.real * self.tail_thick *
                     self.head_scat_lens['contrast%s' % str(self.numberofcontrasts-2)]) / \
                    (self.head_thick * self.tail_scat_lens['con%s' % str(self.numberofcontrasts-2)] *
                     (1 - self.head_layers['con%s' % str(self.numberofcontrasts-2)].vfsolv))
                self.head_layers['con%s' % str(self.numberofcontrasts-1)].sld.real.constraint = \
                    (self.head_layers['con%s' % str(self.numberofcontrasts-2)].sld.real *
                     self.head_scat_lens['con%s' % str(self.numberofcontrasts-1)]) / \
                    (self.head_scat_lens['con%s' % str(self.numberofcontrasts-2)])
                self.tail_layers['con%s' % str(self.numberofcontrasts-1)].sld.real.constraint = \
                    (self.head_layers['con%s' % str(self.numberofcontrasts-1)].sld.real *
                     self.head_thick * self.tail_scat_lens['con%s' % str(self.numberofcontrasts-1)] *
                     (1 - self.head_layers['con%s' % str(self.numberofcontrasts-2)].vfsolv)) / \
                    (self.tail_thick * self.head_scat_lens['con%s' % str(self.numberofcontrasts-1)])

    # This is the function called to get the structures, basically consolidating a series of
    # functions here.
    def get_structures(self):
        self.set_sld()
        self.set_thick_rough()
        self.set_layers()
        self.set_constraints()
        self.tail_layers['con0'].sld.real.setp(vary=True, bounds=(self.guess_sld() - (0.25 * self.guess_sld()),
                                                                  self.guess_sld() + (0.25 * self.guess_sld())))
        for i in range(0, self.numberofcontrasts):
            self.structures['con%s' % i] = (self.supers['con%s' % i] | self.tail_layers['con%s' % i] |
                                            self.head_layers['con%s' % i] | self.sub_layers['con%s' % i])

    # Quick method to calculate the APM, better to do MCMC and treat probabilistically
    @property
    def get_apm(self):
        apm = self.tail_scat_lens['con0'] / (self.tail_layers['con0'].sld.real.value * 1E-6 *
                                             self.tail_layers['con0'].thick.value)
        return apm

    # Quick method to get the molecular volumes, better to do MCMC and treat probabilistically
    @property
    def get_molecular_volumes(self):
        head = self.head_layers['con0'].thick.value * self.head_scat_lens['con0'] * \
               (1 - self.head_layers['con0'].vfsolv.value) ** 2 / \
               (self.tail_layers['con0'].sld.real.value * 1E-6 * self.tail_layers['con0'].thick.value)
        tail = self.tail_scat_lens['con0'] / (self.tail_layers['con0'].sld.real.value * 1E-6)
        total = head + tail
        return head, tail, total
