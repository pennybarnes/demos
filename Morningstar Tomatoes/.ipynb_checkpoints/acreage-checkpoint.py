import IPython
import ipyleaflet

import matplotlib.dates as mdates
import ipywidgets as widgets

from collections import defaultdict

import descarteslabs as dl
import descarteslabs.workflows as wf

import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd

import datetime
import arrow

class acreage(object):
    def __init__(self, map_widget):
        self.m = map_widget
        
        self.draw_control = ipyleaflet.DrawControl(
            edit=False,
            remove=False,
            circlemarker={},
            polyline={},
            polygon={},
            rectangle={
                "shapeOptions": {
                    "fillColor": "#d534eb",
                    "color": "#d534eb",
                    "fillOpacity": 0.2,
                }
            },
        )
        self.m.add_control(self.draw_control)
        
        
    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            raise RuntimeError("Must click on a point first")