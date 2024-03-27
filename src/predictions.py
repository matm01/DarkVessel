import matplotlib.pyplot as plt
import numpy as np
import pyproj
import os
import subprocess
import pandas as pd
import rasterio as rio
import utils.geo_utils as geos
import utils.image_processing as ipc
import utils.utils as utils

import sys

from google.cloud import storage
from ultralytics import YOLO
from rasterio.plot import show

MASK_THRESHOLD = 40
CLIP_MIN_MAX = (-30, 0)
FROM_GC_BUCKET = True
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 64
weights_path = 'runs/detect/train18/weights/best.torchscript'


def get_image(filename: str, preprocess: bool = True, plot: bool = False):
    if FROM_GC_BUCKET:
        client = storage.Client()
        bucket = client.get_bucket(os.environ['DV_BUCKET'])
        bucket_id = os.environ['DV_BUCKET']

        subprocess.run(f'gcsfuse --implicit-dirs {bucket_id} data/bucket', shell=True, executable="/bin/bash")
        # ! gcsfuse --implicit-dirs {bucket_id} {local_path}
        with rio.open('data/bucket/' + filename, 'r') as f:
            img = f.read(1)
            metadata = f.meta
        print('download complete')

    else:
        with rio.open('data/download/' + filename, 'r') as f:
            img = f.read(1)
            metadata = f.meta
    if preprocess:

        # img = ipc.histogram_stretch(ipc.to_linear_magnitude(img, *CLIP_MIN_MAX), scale_factor=32)
        # img = ipc.quarter_power_stretch(ipc.to_linear_magnitude(img, *CLIP_MIN_MAX), scale_factor=16)
        img = ipc.stretch_image(img, *CLIP_MIN_MAX)
        # iemg = ipc.arctangent_stretch(ipc.to_linear_magnitude(img, *CLIP_MIN_MAX), scale_factor=4000)

    if plot:

        ipc.plot_img_and_hist(img)

    return img, metadata


def get_tiles(img: np.ndarray) -> tuple:
    tiles = ipc.split_image(img)

    return utils.remove_land_tiles(tiles, MASK_THRESHOLD)


def do_prediction(tiles_list: list, batch_size: int = BATCH_SIZE):
    if batch_size > len(tiles_list):
        batch_size = len(tiles_list)
    model = YOLO(weights_path)
    results = model.predict(source=tiles_list, conf=CONFIDENCE_THRESHOLD, batch=batch_size, verbose=True)

    # batching the tiles_list and do the training
    results = []
    for i in range(len(tiles_list) // batch_size):
        results.extend(
            model(tiles_list[i * batch_size : (i + 1) * batch_size], conf=CONFIDENCE_THRESHOLD, verbose=False)
        )

    # predicting the last batch
    if batch_size > 1:
        results.extend(model(tiles_list[(i + 1) * batch_size :], conf=CONFIDENCE_THRESHOLD, verbose=False))

    return results


def get_transformer(metadata: dict) -> pyproj.Transformer:
    source_crs = pyproj.CRS(metadata['crs'])
    target_crs = pyproj.CRS("EPSG:4326")

    # Create a transformer object
    return pyproj.Transformer.from_crs(source_crs, target_crs)


def predict(filename: str):

    image, metadata = get_image(filename, plot=True)

    list_of_idx, tiles_list = get_tiles(image)

    results = do_prediction(tiles_list)
    transformer = get_transformer(metadata)
    ships_and_coords = geos.list_of_ships_and_coords_masked(results, metadata['transform'], transformer, list_of_idx)

    csv_name = filename.split('/')[-1].split('.')[0]
    pred_df = pd.DataFrame(ships_and_coords)
    pred_df.to_csv(f'data/mask_test.csv', index=False)
    return pred_df


if __name__ == '__main__':
    filename = sys.argv[1]
    predict(filename)
