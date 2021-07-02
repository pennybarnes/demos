import descarteslabs as dl
import descarteslabs.workflows as wf
#from descarteslabs.vectors import FeatureCollection

import ipyleaflet
#from ipyleaflet import DrawControl, GeoJSON, LayerGroup
import ipywidgets
from ipywidgets import Layout #, link, HBox
import IPython

from collections import defaultdict
from scipy.signal import savgol_filter
from scipy import interpolate
import pandas as pd
import numpy as np

import datetime
import arrow

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class CoverCrops(object):
    """
    Sets up the interactive map and widget functionality
    for the cover crop demo
    """
    def __init__(self, map_widget):
        self.m = map_widget

        # Add draw control
        self.draw_control = ipyleaflet.DrawControl(
            edit=False,
            remove=False,
            circlemarker={},
            polyline={},
            polygon={"shapeOptions": {
                "fillColor": "#060200",
                "color": "#060200",
                "fillOpacity": 0.2,
            }},
            rectangle={"shapeOptions": {
                "fillColor": "#060200", # "#d534eb",
                "color": "#060200",
                "fillOpacity": 0.2
            }}
        )

        self.m.add_control(self.draw_control)

        # Year selection  
        self.year_select = ipywidgets.BoundedIntText(
            value=2020,
            min=2016,
            max=2020,
            step=1,
            description='Year',
            disabled=False
        )

        date_box = ipywidgets.VBox(
            [self.year_select],
            layout=Layout(
                border='3px solid black',
                overflow='hidden'
            )
        )
        self.dt_control = ipyleaflet.WidgetControl(
            widget=date_box,
            position='bottomleft'
        ) 
        self.m.add_control(self.dt_control)

        # Adding clear plot button
        clear_plot_button = ipywidgets.Button(
            description='Clear plot',
            disabled=False,
            button_style='warning',
            tooltip='Plot and all polygons will be cleared'
        )
        self.clear_plot_control = ipyleaflet.WidgetControl(
            widget=clear_plot_button,
            position='topright'
        )
        self.m.add_control(self.clear_plot_control)

        self.get_s2_collection()

        self.ax = None
        self.fig = None
        self.storage = defaultdict(dict)

        self.draw_control.on_draw(self.get_s2_collection)
        self.draw_control.on_draw(self.calculate)

        self.fig_output = ipywidgets.Output()

        self.draw_control.on_draw(self.plot_timeseries)
        self.clear_plot_control.widget.on_click(self.clear_plot)

    def get_s2_collection(self):
        year = self.year_select.value
        
        # Define Image Collection
        ic = wf.ImageCollection.from_id(
            "sentinel-2:L1C",
            wf.Datetime(year-1, month=8, day=15),
            wf.Datetime(year, month=5, day=15),
            processing_level="surface"
        )
        
        ic = ic.filter(lambda img: img.properties['cloud_fraction_0'] < 0.10)
        
        red, nir = ic.unpack_bands("red nir")
        ndvi = ((nir - red)/(nir + red)).rename_bands('ndvi')
        self.variable = ndvi
            
    def plot_timeseries(self, *args, **kwargs):
        """ ... """
        if self.ax is None or self.fig is None:
            fig, ax = plt.subplots(figsize=[5, 4])
            ax.cla()
            ax.set_visible(True)
            self.fig = fig
            self.ax = ax
            first_draw = True
        else:
            first_draw = False
            
        base_dt = np.datetime64('1970-01-01T00:00:00Z')

        if not self.df.empty:
            df = self.df.drop_duplicates(subset=['dates'])
            # Drop nan values
            df = df.dropna()
            
            # Drop masked values
            masked = []
            for val in df.iloc[:, 1]:
                masked.append(type(val) == np.ma.core.MaskedConstant)
            not_masked = [not m for m in masked]
            df = df[not_masked]
                
            dates = df['dates'].values.astype(np.datetime64)
            ndvi = df['ndvi'].values.astype(float)

            # Set up interpolation
            target_dates = np.arange(
                dates.min(),
                dates.max(),
                datetime.timedelta(days=1)
            )

            x1 = (dates - base_dt) / np.timedelta64(1, 's')
            x2 = (target_dates - base_dt) / np.timedelta64(1, 's')

            data = np.array([(x, y) for x, y in sorted(zip(x1, ndvi))])
        
            self.dates = dates
            self.data = data

            # Interpolation
            if not data.shape[0] < 2:
                y2 = interpolate.pchip_interpolate(data[:, 0], data[:, 1], x2)
            
                years = dates.astype('datetime64[Y]').astype(int) + 1970
                max_year = np.max(years)
        
            _ = self.ax.scatter(dates, ndvi)
            _ = self.ax.plot(target_dates, y2)

            _ = self.ax.axvline(np.datetime64(f'{max_year}-02-23'), c='r')
            _ = self.ax.axvline(np.datetime64(f'{max_year}-04-01'), c='r')

        else:
            _ = self.ax.text(
                0.5,
                0.5,
                "Sentinel-2 has no valid observations",
                horizontalalignment="center",
                verticalalignment="center",
            )
            
        # Format ticks
        self.ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%Y-%m-%d')
        )
        self.ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        _ = self.ax.set_xlabel("Date")
        _ = self.ax.set_ylabel("NDVI")
        
        _ = self.ax.set_ylim(0, 1.0)
        self.fig.autofmt_xdate()

        if not first_draw:
            with self.fig_output:
                IPython.display.clear_output()

        with self.fig_output:
            IPython.display.display(self.fig)

        return ""

    def calculate(self, *args, **kwargs):
        last_draw = self.draw_control.last_draw

        if last_draw['geometry']['type'] == 'Point':
            auger_context = wf.GeoContext.from_dltile_key(
                dl.raster.dltile_from_latlon(
                    self.draw_control.last_draw['geometry']['coordinates'][1],
                    self.draw_control.last_draw['geometry']['coordinates'][0],
                    156543.00/2**(max(self.m.zoom, 0)), 2, 0).properties.key
            )
        elif last_draw['geometry']['type'] == 'Polygon':
            auger_context = wf.GeoContext(
                geometry=last_draw['geometry'],
                resolution=156543.00/2**(max(self.m.zoom, 0)),
                crs='EPSG:3857',
                bounds_crs='EPSG:4326',
            )

        with self.m.output_log:
            timeseries = self.variable.map(
                lambda img: (img.properties['date'], img.median(axis='pixels'))
            ).compute(auger_context)

            self.timeseries = timeseries

        values = defaultdict(list)
        dates = []
        with self.m.output_log:
            for date, valdict in timeseries:
                for k, v in valdict.items():
                    values[k].append(v)
                dates.append(arrow.get(date).datetime)

        self.storage['dates'] = dates
        for k, v in values.items():
            self.storage[k] = v

        self._df = pd.DataFrame.from_dict(self.storage)

    def clear_plot(self, *args, **kwargs):
        # Clear draw control polygons
        self.draw_control.clear()

        # Clear plot
        with self.fig_output:
            IPython.display.clear_output()

        # Clear axes and fig
        self.ax = None
        self.fig = None

    _df = None

    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            raise RuntimeError('Must click on a point first')





