import IPython
import ipyleaflet
import ipywidgets as widgets

import numpy as np
import pandas as pd

from typing import List, Optional
import descarteslabs as dl
import descarteslabs.workflows as wf

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import time
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta


class CarbonLost(object):
    def __init__(self, map_widget):

        # Set the variables to work with
        self.m = map_widget

        # Rectangle & Polygon draw control
        self.draw_control = ipyleaflet.DrawControl(
            edit=False,
            remove=False,
            circlemarker={},
            polyline={},
            polygon={},
            rectangle={
                "shapeOptions": {
                    "fillColor": "#6bc2e5",
                    "color": "#6bc2e5",
                    "fillOpacity": 0.2,
                }
            },
        )
        self.m.add_control(self.draw_control)

        # Adding clear plot button
        clear_plot_button = widgets.Button(
            description="Clear plot",
            disabled=False,
            button_style="warning",
            tooltip="Plot and all polygons will be cleared",
        )
        self.clear_plot_control = ipyleaflet.WidgetControl(
            widget=clear_plot_button, position="topright"
        )
        self.m.add_control(self.clear_plot_control)

        # Get the Workflows Images for deforestation and forest carbon
        self.get_deforestation()
        self.get_forest_carbon()

        # Initialize the plotting variables
        self.ax = None
        self.fig = None
        # self.draw_control.on_draw(self.get_deforestation)
        # self.draw_control.on_draw(self.get_forest_carbon)
        self.draw_control.on_draw(self.calculate)

        self.fig_output = widgets.Output()
        self.fig_widget = ipyleaflet.WidgetControl(
            widget=self.fig_output, position="bottomright"
        )
        self.m.add_control(self.fig_widget)

        self.draw_control.on_draw(self.plot_timeseries)
        self.clear_plot_control.widget.on_click(self.clear_plot)

    def get_deforestation(self):

        # Definitions
        deforestation_product = "descarteslabs:ul_deforestation_external_v3"
        deforestation_start = "2020-07-01"
        deforestation_end = "2020-10-30"

        # Load Descartes Labs' deforestation product
        defor_ic = wf.ImageCollection.from_id(
            deforestation_product,
            start_datetime=deforestation_start,
            end_datetime=deforestation_end,
            resampler="near",
        ).max(axis="images")
        detections = defor_ic.pick_bands("detection_date")
        dl_deforestation = detections.mask(detections == 0)

        self.deforestation = dl_deforestation

    def get_forest_carbon(self):

        # Get Descartes Labs' forest carbon density product
        dl_forest_carbon = wf.ImageCollection.from_id(
            "descarteslabs:GEDI:TCH:ForestCarbon:final:v2.1",
            start_datetime="2019-01-01",
            end_datetime="2020-12-31",
            resampler="near",
        ).mosaic()
        dl_acd = dl_forest_carbon.pick_bands(["acd_tons"])
        dl_acd = dl_acd.mask(dl_acd == 0)

        self.carbon_density = dl_acd

    def calculate(self, *args, **kwargs):
        """
        Calculate the time series to be displayed
        """

        # Get the geocontext from the polygon drawn
        last_draw = self.draw_control.last_draw
        if last_draw["geometry"]["type"] == "Polygon":

            # Get CRS
            resolution = 30
            geocontext = self.m.geocontext(resolution=resolution)
            crs = geocontext.crs

            ctx = dl.scenes.AOI(last_draw["geometry"], crs=crs, resolution=resolution)

        # Get the variales
        dl_deforestation = self.deforestation
        dl_acd = self.carbon_density

        # Load arrays of forest carbon and deforestation for the given geocontext
        forest_carbon_data = dl_acd.compute(ctx, progress_bar=True)
        forest_carbon_array = forest_carbon_data.ndarray
        deforestation_data = dl_deforestation.compute(ctx, progress_bar=True)
        deforestation_array = deforestation_data.ndarray

        # Range of deforestationd date values (number of days from 2015-01-01)
        defor_min = np.nanmin(deforestation_array)
        defor_max = np.nanmax(deforestation_array)
        defor_ndays = int(defor_max - defor_min)

        # Define relevant dates
        plot_start_date = datetime(2020, 1, 1)
        detector_start_date = datetime(2015, 1, 1)
        deforestation_start_date = detector_start_date + relativedelta(days=defor_min)
        deforestation_end_date = deforestation_start_date + relativedelta(
            days=defor_max
        )

        # Initialize carbon loss time series
        acd_time_series = np.zeros(defor_ndays)

        # Go through each date between the start and end of the deforestation within the region
        for i in range(0, defor_ndays):

            # Get forest carbon values of the region deforested in the current date
            carbon_lost = np.nansum(
                forest_carbon_array[deforestation_array == defor_min + i]
            )
            if not carbon_lost:
                carbon_lost = 0
            acd_time_series[i] = carbon_lost

        # Create cumulative sum of forest carbon lost
        cumulative_carbon_lost = np.cumsum(acd_time_series)

        # Create dataframe with daily and cumulative carbon lost
        data_dictionary = {
            "carbon_lost_daily": acd_time_series,
            "carbon_lost_cumulative": cumulative_carbon_lost,
        }
        date_array = pd.date_range(
            deforestation_start_date, periods=defor_ndays, freq="D"
        )
        df = pd.DataFrame(index=date_array, data=data_dictionary)
        self._df = df

    def plot_timeseries(self, *args, **kwargs):

        # Initialize the axis
        if self.ax is None or self.fig is None:
            fig, ax = plt.subplots(figsize=[5, 4])
            ax.cla()
            ax.set_visible(True)
            self.fig = fig
            self.ax = ax
            first_draw = True
        else:
            first_draw = False

        # Get the dataframe
        df = self._df

        # Make the plot
        _ = self.ax.plot(df.index, df["carbon_lost_cumulative"], "-", linewidth=3)

        # format the ticks
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.ax.format_xdata = mdates.DateFormatter("%Y-%m-%d")
        _ = self.ax.set_xlabel("Date")
        _ = self.ax.set_ylabel("Mg C/pixel area")
        _ = self.ax.grid(True)
        self.fig.autofmt_xdate()
        self.fig.patch.set_facecolor("white")

        if not first_draw:
            with self.fig_output:
                IPython.display.clear_output()

        with self.fig_output:
            IPython.display.display(self.fig)

        return

    def clear_plot(self, *args, **kwargs):

        # Clear draw control polygons
        self.draw_control.clear()

        # Clear plot
        with self.fig_output:
            IPython.display.clear_output()

        # Clear axes and fig
        self.ax = None
        self.fig = None

    def clear_control_button(self, *args, **kwargs):

        # Clear draw control
        self.m.remove_control(self.clear_plot_control)

    _df = None

    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            raise RuntimeError("Must click on a point first")


def get_masked_images(
    product: str,
    bands: List[str],
    start_datetime: str,
    end_datetime: str,
    processing_level: str,
    strategy: str = "median",
    cloudmask_product: Optional[str] = None,
    cloudmask_invalid: Optional[int] = None,
    cloudmask_bands: Optional[List[str]] = None,
) -> wf.Image:
    """
    This is light implementation of the compositing code, just relevant to
    generate composites from Sentinel-2 imagery
    """

    # Load imagery
    image_collection = wf.ImageCollection.from_id(
        product,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        resampler="lanczos",
        processing_level=processing_level,
    )

    # Select just the bands we want.
    image_collection = image_collection.pick_bands(bands)

    # Scale to the 0-10000 scale for S2
    rescaled = image_collection.map_bands(lambda bandname, band_ic: band_ic * 10000.0)
    arr = rescaled.ndarray
    arr = np.ma.clip(arr, 0, 65535).astype("uint16")
    image_collection = arr.to_imagery(
        properties=image_collection.properties, bandinfo=image_collection.bandinfo
    )

    # Apply cloud mask from the given product
    dlcloud = wf.ImageCollection.from_id(
        cloudmask_product,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        resampler="near",
    )
    clouds = dlcloud.pick_bands(cloudmask_bands)
    cloud_mask = clouds == cloudmask_invalid
    image_collection = apply_dlcloud(image_collection, cloud_mask)

    return image_collection


def apply_dlcloud(
    ic: wf.ImageCollection, cloud_mask_ic: wf.ImageCollection
) -> wf.ImageCollection:
    """
    Mask `ic` with images from `cloud_mask_ic` from corresponding dates.

    `cloud_mask_ic` should be a boolean single-band ImageCollection.

    Does an inner join---only returns masked images from dates that were in both ImageCollections.
    """

    src_groups = ic.groupby(dates=("year", "month", "day"))
    dlcloud_groups = cloud_mask_ic.groupby(dates=("year", "month", "day"))

    def map_dlcloud(group, cloud_imgs_for_date: wf.ImageCollection):
        src_mosaic = src_groups[group].mosaic()
        cloud_mosaic = cloud_imgs_for_date.mosaic()
        return src_mosaic.mask(cloud_mosaic)

    return dlcloud_groups.map(map_dlcloud)
