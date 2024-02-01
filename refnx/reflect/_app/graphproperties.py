"""
GraphProperties: storing information about Matplotlib traces
e.g. colors, linesizes, etc.
"""

import matplotlib.artist as artist


_requiredgraphproperties = {
    "lw": float,
    "label": str,
    "linestyle": str,
    "fillstyle": str,
    "marker": str,
    "markersize": float,
    "markeredgecolor": str,
    "markerfacecolor": str,
    "zorder": int,
    "color": str,
    "visible": bool,
}

_lines = {
    "data_properties": dict,
    "fit_properties": dict,
    "sld_profile_properties": dict,
    "residuals_properties": dict,
}


class GraphProperties(dict):
    def __init__(self):
        self["visible"] = True

        # a Matplotlib trace
        self["ax_data"] = None
        self["ax_fit"] = None
        self["ax_residuals"] = None
        self["ax_sld_profile"] = None
        self["data_properties"] = {}
        self["fit_properties"] = {}
        self["sld_profile_properties"] = {}
        self["residuals_properties"] = {}

    def __getattr__(self, key):
        if key in _requiredgraphproperties:
            return self[key]

        try:
            v = self[key]
        except KeyError:
            return None

        return v

    def __setstate__(self, state):
        self["visible"] = state["visible"]
        self["ax_data"] = None
        self["ax_fit"] = None
        self["ax_sld_profile"] = None
        for line in _lines:
            self[line] = state[line]

    def __getstate__(self):
        self.save_graph_properties()
        d = {}
        d["ax_data"] = None
        d["ax_fit"] = None
        d["ax_sld_profile"] = None
        d["visible"] = self["visible"]
        for line in _lines:
            d[line] = self[line]
        return d

    def save_graph_properties(self):
        # pass
        if self.ax_data is not None:
            for key in _requiredgraphproperties:
                # ax_data is an ErrorbarContainer
                self["data_properties"][key] = artist.getp(
                    self.ax_data[0], key
                )

        if self.ax_fit is not None:
            for key in _requiredgraphproperties:
                self["fit_properties"][key] = artist.getp(self.ax_fit, key)

        if self.ax_residuals is not None:
            for key in _requiredgraphproperties:
                self["residuals_properties"][key] = artist.getp(
                    self.ax_residuals, key
                )

        if self.ax_sld_profile is not None:
            for key in _requiredgraphproperties:
                self["sld_profile_properties"][key] = artist.getp(
                    self.ax_sld_profile, key
                )
