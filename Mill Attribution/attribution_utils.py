import ipyleaflet
import IPython
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point

from descarteslabs.client.services.storage import Storage


def regional_maps(aoi):
    """
    This function calculates the forest carbon lost due to deforestation within the AOI.

    Parameters
    ----------
    aoi : dl.scenes.AOI object
        The buffered area around one mill

    Returns
    -------
    acd : ndarray (2D)
        The map of above-ground carbon density over the AOI
    defor : ndarray (2D)
        The map of deforested area over the AOI
    """

    import numpy as np
    import descarteslabs as dl

    # Load forest carbon
    scenes, ctx = dl.scenes.search(
        aoi,
        products="descarteslabs:GEDI:TCH:ForestCarbon:final:v2.1",
        start_datetime="2015-01-01",
        end_datetime="2021-07-01",
    )
    acd = scenes.mosaic("acd_tons", ctx).squeeze()
    acd = acd.filled(0)

    # Load deforestation
    scenes, _ = dl.scenes.search(
        aoi,
        products="descarteslabs:ul_deforestation_external_v3",
        start_datetime="2020-07-01",
        end_datetime="2020-12-31",
        limit=500,
    )
    defor = scenes.stack("deforested", ctx)
    defor = defor.filled(0)
    defor = np.nanmax(defor, axis=0)
    defor[defor > 0] = 1
    defor = defor.squeeze()

    return acd, defor


def get_spatial_weights(ny, nx):
    """
    This function calculates the inverse-distance weights for each pixel
    using the center of a 2D array as the origin.

    Parameters
    ----------
    ny, nx: int
        Dimensions in the y and x directions respectively

    Returns
    -------
    weights : ndarray (2D)
        Map of weights used to calculate a weighted average over the AOI
    """

    import numpy as np

    # Initialize arrays with the x and y distances
    x = np.arange(nx) - round(nx / 2.0)
    y = np.arange(ny) - round(ny / 2.0)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing="xy")
    xv = xv + 0.1
    yv = yv + 0.1

    # Calculate deca
    dist = np.sqrt(xv ** 2 + yv ** 2)
    inv_dist = -dist + np.abs(np.min(-dist))
    weights = inv_dist / np.max(inv_dist)

    return weights


def calculate_scores_per_mill(args):
    """
    This function will calculate statistics of deforested area and carbon lost
    for the given mill. We calculate a "score" for deforestation and forest carbon
    which are the percentage of the total possible deforestation/forest carbon that
    have been lost in the AOI.

    Parameters
    ----------
    args : list
        List of arguments including, latitude and longitude of the mill, the ID of the mill,
        and the buffer to be drawn around the mill in degrees (0.1 ~ 10km)

    Returns
    -------
    results : dictionary
        Includes the total area deforested, total carbon lost due to deforestation,
        the deforestation score, and the carbon score.
    """

    import json
    import numpy as np
    from shapely.geometry import Point, mapping

    import descarteslabs as dl
    from descarteslabs.client.services.storage import Storage

    lat, lon, mill_id, buffer, version = args

    # Create the buffer around the mill
    geometry = Point(lon, lat).buffer(buffer)

    # AOI definitions
    resolution = 30  # m
    aoi = dl.scenes.AOI(geometry=mapping(geometry), resolution=resolution)
    pixel_area = resolution * resolution
    m2_to_ha = 0.0001

    # Get the rasters of deforestation and forest carbon over the AOI
    acd, defor = regional_maps(aoi)

    # Get the weights based on distance from the origin (center of the region)
    weights = get_spatial_weights(acd.shape[0], acd.shape[1])

    # Count the total area deforested (ha)
    total_defor_area = np.sum(defor) * pixel_area * m2_to_ha

    # Count the total forest carbon lost (Mg C)
    total_defor_carbon = np.sum(acd[defor > 0] * pixel_area)

    # Maximum weighted average of carbon lost if all area was deforested (Mg C)
    max_carbon = np.average(acd * pixel_area, weights=weights)

    # Maximum weighted average of deforested area if all area was deforested (ha)
    forest_mask = np.array(acd[:])
    forest_mask[forest_mask > 0] = 1
    max_area = np.average(forest_mask * pixel_area * m2_to_ha, weights=weights)

    # Weighted average of forest carbon lost (Mg C)
    defor_carbon = np.array(acd[:]) * pixel_area
    defor_carbon[defor == 0] = 0
    defor_carbon_mean = np.average(defor_carbon, weights=weights)

    # Weighted average of deforested area (ha)
    defor_area_mean = np.average(defor * pixel_area * m2_to_ha, weights=weights)

    # Percentage of maximum forest carbon lost and deforested area
    carbon_score = 100 * defor_carbon_mean / max_carbon
    area_score = 100 * defor_area_mean / max_area

    # Dictionary of results
    results = {
        "lat": str(lat),
        "lon": str(lon),
        "mill_id": str(mill_id),
        "total_deforested_area": str(total_defor_area),
        "total_carbon_lost": str(total_defor_carbon),
        "area_score": str(area_score),
        "carbon_score": str(carbon_score),
    }

    # Save to DL Storage
    key = "mill_stats_mvp_v{}_{}_{}".format(version, lat, lon)
    Storage().set(key, json.dumps(results), storage_type="data")
    print(key + " was uploaded to DL Storage!")

    return results


def get_wf_raster_layer(lons, lats, values, buffer):
    """
    This function creates a Workflows image by rasterizing give values to a
    disk around a mill.

    Parameters
    ----------
    lons, lats : 1D ndarray
        Longitudinal and latitudinal coordinates of each mill
    values : 1D ndarray
        Values corresponding to each mill that will be rasterized
    buffer : float
        Radius of buffer around the mill that will be used to rasterize the values

    Returns
    -------
    wf_raster : descarteslabs.workflows.types.geospatial.image.Image
        Workflows Image with the values rasterized for each mill
    """

    import descarteslabs.workflows as wf

    # Create Workflows geometries for each mill location
    geometries = [
        wf.Geometry(type="Point", coordinates=[lon, lat])
        for lon, lat in zip(lons, lats)
    ]
    geometries_buffered = [geometry.buffer(buffer) for geometry in geometries]

    # Rasterize the value of the carbon score to the respective mill
    features = [
        wf.Feature(geometry=geometry_buffered, properties={"value": value})
        for geometry_buffered, value in zip(geometries_buffered, values)
    ]
    vectors = wf.FeatureCollection(features)
    wf_raster = vectors.rasterize(value="value", merge_algorithm="replace")

    return wf_raster


def get_df_from_results(version):
    """
    This function aggregates the results saved to DL Storage and creates a geopandas dataframe.
    """

    import concurrent.futures
    import json
    import geopandas as gpd
    import pandas as pd

    # Get results stored in DL Storage
    storage_keys = list(
        Storage().iter_list(prefix="mill_stats_mvp_v{}_".format(version))
    )
    nmills = len(storage_keys)
    print("Number of mills processed: {}".format(nmills))

    # Aggregate all the results into a single data frame
    df = pd.DataFrame(
        columns=[
            "lat",
            "lon",
            "mill_id",
            "total_deforested_area",
            "total_carbon_lost",
            "area_score",
            "carbon_score",
        ]
    )
    with concurrent.futures.ThreadPoolExecutor() as exc:
        jobs = exc.map(
            lambda key: (json.loads(Storage().get(key))),
            storage_keys,
        )
        for result in jobs:
            try:
                df = df.append(result, ignore_index=True)
            except Exception as exc:
                print(f"{exc}")

    numeric_columns = [
        "lat",
        "lon",
        "total_deforested_area",
        "total_carbon_lost",
        "area_score",
        "carbon_score",
    ]
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Create geopandas from the mill stats
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

    return gdf


class MillDistance(object):
    def __init__(self, map_widget):
        self.m = map_widget

        self.ax = None
        self.fig = None

        # Setting up draw control (polygons)
        self.draw_control = ipyleaflet.DrawControl()
        self.draw_control.polygon = {
            "shapeOptions": {
                "fillColor": "#d534eb",
                "color": "#d534eb",
                "fillOpacity": 0.2,
            }
        }
        self.draw_control.edit = False
        self.m.add_control(self.draw_control)

        # Setting up clear plot button control
        self.clear_plot_control = ipyleaflet.WidgetControl(
            widget=widgets.Button(
                description="Clear plot",
                disabled=False,
                button_style="warning",
                tooltip="Plot and all polygons will be cleared",
            ),
            position="topright",
        )
        self.m.add_control(self.clear_plot_control)

        # Setting up output for plots
        self.fig_output = widgets.Output()
        self.fig_widget = ipyleaflet.WidgetControl(
            widget=self.fig_output, position="bottomright"
        )
        self.m.add_control(self.fig_widget)

        # Watching for events
        # self.draw_control.on_draw(self.get_df_from_results)
        # self.draw_control.on_draw(self.get_arguments)
        self.draw_control.on_draw(self.find_mills)
        self.clear_plot_control.widget.on_click(self.clear_plot)

    def get_arguments(self, geodataframe):
        self.geodataframe = geodataframe

    def find_mills(self, *args, **kwargs):

        # Find nearest mills
        last_draw = self.draw_control.last_draw
        if last_draw["geometry"]["type"] == "Point":

            palm_geom = Point(last_draw["geometry"]["coordinates"])

        elif last_draw["geometry"]["type"] == "Polygon":

            palm_geom = Polygon(last_draw["geometry"]["coordinates"][0])

        # Get the geopandas data frame
        gdf = self.geodataframe  # get_df_from_results(1)

        # Calculate the distances to each mill in the Universal Mill List and add it to a duplicated geopandas dataframe
        dists = list(
            map(lambda i: gdf["geometry"][i].distance(palm_geom), range(gdf.shape[0]))
        )
        gdf_palm = gdf.copy()
        gdf_palm["dist_to_plantation"] = dists

        # Sort to find the closest mills
        gdf_palm = gdf_palm.sort_values(by=["dist_to_plantation"], ignore_index=True)
        self._df = gdf_palm

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
            raise RuntimeError("Must click on a point first")
