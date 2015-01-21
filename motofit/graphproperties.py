"""
GraphProperties: storing information about Matplotlib traces
e.g. colors, linesizes, etc.
"""
import matplotlib.artist as artist

_requiredgraphproperties = {'lw': float,
                             'label': str,
                             'linestyle': str,
                             'fillstyle': str,
                             'marker': str,
                             'markersize': float,
                             'markeredgecolor': str,
                             'markerfacecolor': str,
                             'zorder': int,
                             'color': str,
                             'visible': bool,
                             'picker': float}

_lines = {'line2D_properties': dict,
          'line2Dfit_properties': dict,
          'line2Dsld_profile_properties': dict,
          'line2Dresiduals_properties': dict}

class GraphProperties(dict):
    def __init__(self):
        self['visible'] = True

        #a Matplotlib trace
        self['line2D'] = None
        self['line2Dfit'] = None
        self['line2Dresiduals'] = None
        self['line2Dsld_profile'] = None
        self['line2D_properties'] = {}
        self['line2Dfit_properties'] = {}
        self['line2Dsld_profile_properties'] = {}
        self['line2Dresiduals_properties'] = {}

    def __getattr__(self, key):
        if key in _requiredgraphproperties:
            return self[key]

    def __getstate__(self):
        self.save_graph_properties()
        d = {}
        for line in _lines:
            d[line] = self[line]
        return d

    def save_graph_properties(self):
        if self.line2D:
            for key in _requiredgraphproperties:
                self['line2D_properties'][key] = artist.getp(self.line2D, key)

        if self.line2Dfit:
            for key in _requiredgraphproperties:
                self['line2Dfit_properties'][key] = artist.getp(self.line2Dfit,
                                                                key)

        if self.line2Dresiduals:
            for key in _requiredgraphproperties:
                self['line2Dresiduals_properties'][key] = artist.getp(
                                        self.line2Dresiduals,
                                        key)

        if self.line2Dsld_profile:
            for key in _requiredgraphproperties:
                self['line2Dsld_profile_properties'][key] = artist.getp(
                                        self.line2Dsld_profile,
                                        key)
