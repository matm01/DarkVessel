import matplotlib.pyplot as plt
import numpy as np
import pyproj
import os
import pandas as pd
from pathlib import Path
import rasterio as rio

import utils.geo_utils as geos
import utils.image_processing as ipc
import utils.utils as utils
import utils.land_mask as lmsk


from google.cloud import storage
from ultralytics import YOLO
from rasterio.plot import show

MASK_THRESHOLD = 3
CLIP_MIN_MAX = (-30, 0)
FROM_GC_BUCKET = True
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 16

yolo_weights = 'runs/detect/best.torchscript'
# yolo_weights = 'runs/detect/train18_with_sts/weights/best.torchscript'

ocean_mask = 'data/mask_laconian_bay.geojson'

bucket_id = os.environ.get('SAR_BUCKET')
local_path = f'data/{bucket_id}/VH'
results_path = 'data/results'
# local_path = 'data/bucket/VH'


def process_image(img: np.ndarray) -> np.ndarray:
    return ipc.stretch_image(img, *CLIP_MIN_MAX)


def get_tiles(img: np.ndarray) -> tuple:
    tiles = ipc.split_image(img)
    return utils.remove_land_tiles(tiles, MASK_THRESHOLD)


def do_prediction(tiles_list: list, batch_size: int = BATCH_SIZE):
    model = YOLO(yolo_weights)
    tiles_list_size = len(tiles_list)
    results = []

    if tiles_list_size < batch_size:
        results = model(tiles_list, conf=CONFIDENCE_THRESHOLD, verbose=False)
    else:
        for i in range(len(tiles_list) // batch_size):
            batch_results = model(
                tiles_list[i * batch_size : (i + 1) * batch_size], conf=CONFIDENCE_THRESHOLD, verbose=False
            )
            results.extend(batch_results)
        # Predicting the last batch
        if len(tiles_list) % BATCH_SIZE != 0:
            results.extend(model(tiles_list[(i + 1) * batch_size :], conf=CONFIDENCE_THRESHOLD, verbose=False))

    return results


def get_transformer(metadata: dict) -> pyproj.Transformer:
    source_crs = pyproj.CRS(metadata['crs'])
    target_crs = pyproj.CRS("EPSG:4326")  # Latitude, Longitude coordinates
    return pyproj.Transformer.from_crs(source_crs, target_crs)


def save_image(list_ship_positions, image, filename, width=290, height=290):
    images = []
    image_id = filename.split('.')[0]
    # image_folder_path = f'{results_path}/{image_id}'
    # if not os.path.exists(image_folder_path):
    #     os.makedirs(image_folder_path)
    for idx, position in enumerate(list_ship_positions):
        img = image[
            position[1] - width // 2 : position[1] + width // 2, position[0] - height // 2 : position[0] + height // 2
        ]
        image_path = f'assets/{image_id}_ship_{idx}.png'
        plt.imsave(image_path, np.dstack((img, img, img)))
        images.append(image_path)
    return images


def predict(filename: str, plot=False):

    print("Applying land mask to image")
    image, metadata = lmsk.clip_image(f'{local_path}/{filename}', ocean_mask)

    print("Pre-processing SAR image")
    image = process_image(image)
    image = ipc.resize_image(image)

    if plot:
        ipc.plot_img_and_hist(image)

    print("Spliting image into tiles")
    list_of_idx, tiles_list = get_tiles(image)

    print("Predicting ships and STS-transfers in tiles")
    results = do_prediction(tiles_list)

    print("Getting detected ships coordinates in latitude and longitude")
    transformer = get_transformer(metadata)
    ships_and_coords, list_ship_positions = geos.list_of_ships_and_coords_masked(
        results, metadata['transform'], transformer, list_of_idx
    )

    print("Saving images of detected ships")
    images = save_image(list_ship_positions, image, filename)

    print("Saving results to csv")
    pred_df = pd.DataFrame(ships_and_coords)
    pred_df['image'] = images
    image_id = filename.split('.')[0]
    pred_df.to_csv(f'results/{image_id}.csv', index=False)
    return pred_df


if __name__ == '__main__':
    # filename = sys.argv[1]
    filename = 'S1A_IW_GRDH_1SDV_20220304T162332_20220304T162357_042174_05068B_5B73.tif'
    predict(filename)
