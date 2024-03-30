import requests
import shutil
import ee

import matplotlib.pyplot as plt
import numpy as np
from google.cloud import storage
import os


def get_image_via_thumbURL(index: int, coordinates: tuple, dates: tuple, multiplier: int = 1):
    """
    Function to get an image from a given point and dates, with optional multiplier for dimensions.

    Args:
        index (int): The index of the image.
        coordinates (tuple): The coordinates of the point.
        dates (tuple): The start and end dates for filtering the image collection.
        multiplier (int, optional): A multiplier for the dimensions of the image. Defaults to 1.
    """

    def mask_edge(image):
        edge = image.lt(-30.0)
        masked_image = image.mask().And(edge.Not())
        return image.updateMask(masked_image)

    if multiplier == 1:
        dimensions = '640x640'
    else:
        dimensions = f'{640 * multiplier}x{640 * multiplier}'

    region_size = 2000 * multiplier

    # Generate the desired image from the given point.
    point = ee.Geometry.Point(coordinates)
    region = point.buffer(region_size).bounds()
    image = (
        (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .select('VV')
            .filterDate(*dates)
            .map(mask_edge)
            .filter(ee.Filter.bounds(region))
        )
        .mosaic()
        .clip(region)
    )

    # Fetch the URL from which to download the image
    url = image.getThumbURL({'dimensions': dimensions, 'min': -20, 'max': 0, 'format': 'jpg'})

    # Request the image
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise r.raise_for_status()

    filename = 'data/download/img_%05d.jpg' % index
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", index)


def split_image(image: np.ndarray, multiplier: int = 1) -> np.ndarray:
    """
    Split the input image into a tiled array based on the specified multiplier.

    Args:
        image (np.ndarray): The input image to be split.
        multiplier (int, optional): The multiplier to determine the tiling size. Defaults to 1.

    Returns:
        np.ndarray: The tiled array representing the split image.
    """
    tiled_array = image.reshape(multiplier, 640, multiplier, 640, 3)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array


def combine_predictions_from_list(tiles: list, repetitions: int) -> np.ndarray:
    """
    Combines a list of tiles into a single array by concatenating them row-wise.

    Args:
        tiles (list): List of tiles to combine.
        repetitions (int): Number of repetitions for concatenation.

    Returns:
        np.ndarray: Array containing the combined tiles.
    """

    for i in range(repetitions):
        row = np.concatenate([tiles[5 * i].plot(), tiles[5 * i + 1].plot()], axis=1)

        for j in range(2, repetitions):
            row = np.concatenate([row, tiles[5 * i + j].plot()], axis=1)

        if i == 0:
            combined_tiles = row
        else:
            combined_tiles = np.concatenate([combined_tiles, row], axis=0)

    return combined_tiles


def combine_predictions_from_array(tiles: list, n_rows: int, n_columns: int) -> np.ndarray:
    """
    Combines an array of tiles into a single array by concatenating them row-wise.

    Args:
        tiles (list): List of tiles to combine.
        repetitions (int): Number of repetitions for concatenation.

    Returns:
        np.ndarray: Array containing the combined tiles.
    """

    for i in range(n_rows + 1):
        row = np.concatenate([tiles[i][0], tiles[i][1]], axis=1)

        for j in range(2, n_columns + 1):
            row = np.concatenate([row, tiles[i][j]], axis=1)

        if i == 0:
            combined_tiles = row
        else:
            combined_tiles = np.concatenate([combined_tiles, row], axis=0)

    return combined_tiles


def download_all_tifs():
    """
    Downloads all the TIFF files from the specified Google Cloud Storage bucket that are not already present in the local directory.

    This function uses the Google Cloud Storage Python client library to connect to the specified bucket and download the TIFF files that are not already present in the local directory. The bucket name is obtained from the 'DV_BUCKET' environment variable. The downloaded files are saved in the '../data/download/' directory.

    Parameters:
    None

    Returns:
    None
    """
    client = storage.Client()
    bucket = client.get_bucket(os.environ['DV_BUCKET'])
    dir_list = [file for file in os.listdir('../data/download/') if file.endswith('.tif')]

    for blob in bucket.list_blobs():
        if blob.name.endswith('.tif') and blob.name not in dir_list:
            blob.download_to_filename('../data/download/' + blob.name)


def plot_tiles(tiles: list, n_rows, n_columns) -> None:
    """
    Plots a array of tiles.

    Args:
        tiles (list): List of tiles to plot.

    Returns:
        None
    """
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(n_columns // 3, n_rows // 3))

    for i in range(n_rows):
        for j in range(n_columns):
            axes[i, j].imshow(tiles[i][j], cmap='gray')
            axes[i, j].axis('off')

    plt.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)
    plt.show()


def remove_land_tiles(tiles, threshold):
    """
    Removes land tiles from the given tiles array based on a threshold value and converts the remaining tiles to 3-channel grayscale.

    Parameters:
        tiles (ndarray): The input array of tiles.
        threshold (float): The threshold value to determine if a tile is land or not.

    Returns:
        tuple: A tuple containing two lists. The first list contains the indices of the remaining tiles.
        The second list contains the tiles themselves.
    """
    list_of_tiles = []
    list_of_idx = []
    n_rows, n_columns = tiles.shape[:2]
    for i in range(n_rows):
        for j in range(n_columns):
            if tiles[i][j].mean() > threshold:
                list_of_tiles.append(np.dstack([tiles[i][j], tiles[i][j], tiles[i][j]]))
                list_of_idx.append((i, j))
    print(f'The list contains {len(list_of_idx)} tiles.')
    return list_of_idx, list_of_tiles
