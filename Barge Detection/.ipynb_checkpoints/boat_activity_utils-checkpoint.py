import descarteslabs as dl
import descarteslabs.workflows as wf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import arrow

import IPython
from ipyleaflet import DrawControl, WidgetControl, FullScreenControl
from ipywidgets import Output, DatePicker, Button, Layout
from collections import defaultdict
from skimage.draw import polygon

class TimeSeries(object):
    def __init__(self, map_widget, out_widget=None, draw_control=None, date_picker=None, remove_plot_button=None, bin_month=False, filename=None):
        self.m  = map_widget
        if not draw_control:
            self.draw_control = DrawControl(
                edit=False,
                remove=False,
                circlemarker={},
                polyline={},
                polygon={"shapeOptions": {
                            "fillColor": "#d534eb",
                            "color": "#d534eb",
                            "fillOpacity": 0.2
                        }},
                rectangle={"shapeOptions": {
                            "fillColor": "#d534eb",
                            "color": "#d534eb",
                            "fillOpacity": 0.2
                        }}
                )
        else:
            self.draw_control = draw_control
        
        self.m.add_control(self.draw_control)
        self.m.add_control(FullScreenControl())
        
        if not out_widget:
            self.out_widget = Output()
        else:
            self.out_widget = out_widget
        
        widget_control = WidgetControl(widget=self.out_widget,
                                       max_width=400,
                                       max_height=150,
                                       position='bottomleft')
        
        self.m.add_control(widget_control)
        
        if not date_picker:
            self.date_picker = DatePicker(description='Start date',
                                          value=datetime.datetime.strptime('01/01/2019', '%m/%d/%Y'),
                                          layout=Layout(width='250px', margin='2px 5px 2px 0px'),
                                          disabled=False)
        else:
            self.date_picker = date_picker

        widget_control_date = WidgetControl(widget=self.date_picker,
                                            max_width=400,
                                            max_height=50,
                                            position='topright')
        
        self.m.add_control(widget_control_date)
        
        if not remove_plot_button:
            self.remove_plot_button = Button(layout=Layout(width='37px'))
            self.remove_plot_button.description = '\u274C'
            self.remove_plot_button.tooltip = 'Clear plot'
        else:
            self.remove_plot_button = remove_plot_button
        
        widget_control_remove_plot = WidgetControl(widget=self.remove_plot_button,
                                                   max_width=50,
                                                   max_height=50,
                                                   position='bottomright')
        
        self.m.add_control(widget_control_remove_plot)
        
        self.bin_month = bin_month
        self.filename = filename
        
        self.draw_control.on_draw(self.compute)
        self.remove_plot_button.on_click(self.remove_plot)

    @wf.map.output_log.capture(clear_output=True)
    def remove_plot(self, *args, **kwargs):
        with self.out_widget:
            IPython.display.clear_output()
        
        self.draw_control.clear()
    
    @wf.map.output_log.capture(clear_output=True)
    def compute(self, *args, **kwargs):
        last_draw = self.draw_control.last_draw
        self.out_widget.clear_output()
        
        tilesize = 1
        resolution = 10.0
        pad = 500
        
        imgs = self.get_sar()
        
        lat, lon = self.get_center_location(last_draw)
        dltile = dl.scenes.DLTile.from_latlon(
            lat,
            lon,
            tilesize=tilesize,
            resolution=resolution,
            pad=pad
        )
        context = wf.GeoContext.from_dltile_key(dltile.key)
        
        # Convert polygon coordinates into pixel coordinates
        coordinates = [self.convert_to_pixel(dltile, c[0], c[1], tilesize, pad) for c in last_draw['geometry']['coordinates'][0]]
        
        timeseries = imgs.mean(axis='bands').compute(context)
        status = [self.get_fraction(np.squeeze(a), coordinates) for a in timeseries.ndarray.data]
        dates = [arrow.get(d['date']).datetime for d in timeseries.properties]
        
        self.df = pd.DataFrame(columns=['dates', 'status'], data={'dates': dates, 'status': status})
        self.df = self.df.dropna()
        
        # bin by month
        if self.bin_month:
            self.df['year_month'] = self.df['dates'].apply(lambda row: datetime.datetime.strftime(row, '%Y%m'))
            self.df = self.df.groupby('year_month').mean().reset_index()
            self.df['dates'] = self.df['year_month'].apply(lambda row: pd.Timestamp(datetime.datetime.strptime(row, '%Y%m')))
        
        if self.filename:
            self.df.to_csv(self.filename)
        
        fig, ax = plt.subplots(1, 1, figsize=(4,2))
        self.df.plot('dates', 'status', ax=ax, rot=0, legend=None)
        ax.set_xlabel('')
        ax.set_ylabel('')
        #ax.set_ylim(0.0, np.ceil(self.df['status'].max() * 10.0)/10.0)
        ax.grid(True)
        
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        with self.out_widget:
            IPython.display.display(fig)
    
    def get_sar(self):
        return (
            wf.ImageCollection.from_id('sentinel-1:GRD',
                                       start_datetime=self.date_picker.value,
                                       end_datetime=datetime.datetime.now())
            .pick_bands(['vv', 'vh'])
        )
    
    def get_center_location(self, fc):
        # Calculates the coordinates of the polygon center
        coordinates = np.array(fc['geometry']['coordinates'][0])
        lon = np.mean(coordinates[:,0])
        lat = np.mean(coordinates[:,1])
        
        return lat, lon
    
    def get_fraction(self, arr, coordinates):
        # Calculates the fraction of the polygon with pixel values >threshold
        # given the input array and the pixel coordinates of the polygon
        a = np.array(arr)
        coord = np.array(coordinates)
        img_masked = np.zeros((a.shape[0], a.shape[1]), 'float32')
        
        rr, cc = polygon(coord[:,1], coord[:,0], img_masked.shape)
        img_masked[rr, cc] = a[rr, cc]
        
        polygon_area = np.sum((img_masked>0).astype(np.int32))
        threshold_area = np.sum((img_masked>0.3*0.75).astype(np.int32))
        
        return threshold_area / polygon_area
    
    def convert_to_pixel(self, dltile, lon, lat, tilesize, pad):
        # Simple method to convert lon and lat into pixel coordinates given
        # the dltile
        tile_coords = np.array(dltile['geometry']['coordinates'][0])
        a_x = np.max(tile_coords[:,0]) - np.min(tile_coords[:,0])
        a_y = np.max(tile_coords[:,1]) - np.min(tile_coords[:,1])
        x_min = np.min(tile_coords[:,0])
        y_min = np.min(tile_coords[:,1])
        
        img_size = tilesize + 2*pad
        
        return int(img_size / a_x * (lon - x_min)), int(img_size - img_size / a_y * (lat - y_min))
