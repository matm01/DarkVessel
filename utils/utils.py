import requests
import shutil
import ee
import numpy as np


def get_image(index: int, coordinates: tuple, dates: tuple, multiplier: int = 1):
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


def combine_predictions(tiles: list, repetitions: int) -> np.ndarray:
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
