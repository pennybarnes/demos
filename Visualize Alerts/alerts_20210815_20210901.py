document = {
    # Date-range for this set of alerts
    "start_datetime": "2021-08-15",
    "end_datetime": "2021-09-01",
    # Output Files
    "gs_dir": "gs://dl-appsci/deforestation/reporting/unilever/metadata_delivery_{start_datetime}_{end_datetime}/",
    "customer_gs_dir": "gs://unilever-data/deforestation-alerts/{file_format}/",
    "alerts_fname": "alerts_with_metadata_{start_datetime}_{end_datetime}.geojson",
    "clusters_fname": "clusters_with_metadata_{start_datetime}_{end_datetime}.geojson",
    # Final deliverable files to Unilever
    # Will produce .geojson and .zip (shp) files
    "alerts_output_prefix": "alerts_delivery_{start_datetime}_{end_datetime}",
    "clusters_output_prefix": "clusters_delivery_{start_datetime}_{end_datetime}",
    # Prefix for intermediate DL Storage products
    "metadata_prefix": "deforestation_{start_datetime}_{end_datetime}_metadata",
    # Processing parameters
    "deforestation_product": "descarteslabs:ul_deforestation_external_v3",
    "deforestation_band": "detection_date",
    "resolution": 20,  # deforestation alerts are processed in pixels of 20m
    "tilesize": 2048,
    "pad": 0,
    "min_pixels": 1,
    "task_kwargs": {
        "image": "us.gcr.io/dl-ci-cd/deforestation/base/master@sha256:97c8f059a23d2e4c055ba223cfc370bd96fbd1ab784d52ad2437814b93a108b0",  # Jul 26 master # NOQA
        "retry_count": 5,
        "maximum_concurrency": 80,
        "memory": "3Gi",
        "task_timeout": 28800,  # 8h in seconds
    },
    # Deploy over this aoi
    "aoi_file": "unilever/region_shapes/indonesia_malaysia_hull.geojson",
    # Intersect with these masks
    "raster_intersection": [
        {
            "name": "peat",
            "product": "descarteslabs:cifor_peatlands_v0",
            "band": "peat_mask",
        },
        {
            "name": "wdpa",
            "product": "descarteslabs:wdpa_indonesia_malaysia_v2",
            "band": "wdpa_mask",
        },
        {
            "name": "gfw_protected",
            "product": "descarteslabs:conservation_areas_gfw",
            "band": "protected",
        },
        {
            "name": "mangrove",
            "product": "descarteslabs:mangrove_forests_indonesia_malaysia_v1",
            "band": "mangrove_mask",
        },
    ],
    # Calculate the distance to the nearest non-zero pixel in these masks
    "raster_distance": [
        {
            "name": "palm",
            "product": "descarteslabs:unilever-palm-classification-v3",
            "band": "class",
            "max_distance_m": 10000,
        }
    ],
    # Use these catalog products to define mature forest
    "mature_forest": {
        "forest_start": {
            "product": "descarteslabs:Landsat5:TreeCover:1990:final:v3",
            "band": "forest_thresholded75",
        },
        "forest_end": {
            "product": "descarteslabs:Sentinel2:ForestMask:2020:vtest:deeplabv3plus_20200730-201849_ckpt-17_OS16",
            "band": "forest_thresholded50",
        },
    },
}
