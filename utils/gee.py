
import csv
import ee
import pandas as pd
from typing import Generator
import requests
import urllib3
import http.client
import time


SENTINEL1_GRD = 'COPERNICUS/S1_GRD'


def format_date(date: str) -> ee.Date:
    """Format the date string into an ee.Date object.

    Args:
        date: A string representing the date.

    Returns:
        An ee.Date object representing the formatted date.
    """
    return ee.Date(date)


def generate_date_range(
        start: str,
        end: str,
        timedelta: int = 1
        ) -> Generator[ee.DateRange, None, None]:
    """Generate a date range between the start and end dates.

    Args:
        start: A string representing the start date.
        end: A string representing the end date.
        timedelta: An integer representing the time difference between each date in the range.

    Yields:
        An ee.DateRange object representing each date range in the generated range.
    """
    start = format_date(start)
    end = format_date(end)
    n_date_range = end.difference(start, 'days').getInfo()
    for _ in range(n_date_range):
        next_day = start.advance(timedelta, 'day')
        date_range = ee.DateRange(start, next_day)
        start = start.advance(timedelta, 'day')
        yield date_range


def get_image_collection(
        aoi: list,
        date_range: tuple,
        band: list = ['VH', 'VV'],
        collection: str = SENTINEL1_GRD
        ) -> ee.ImageCollection:
    """Get a collection of SAR images.

    Args:
        aoi: A list representing the area of interest.
        date_range: A tuple representing the start and end dates.
        band: A list of strings representing the bands to select.
        collection: A string representing the image collection.

    Returns:
        An ee.ImageCollection object representing the collection of SAR images.
    """
    if type(aoi) == list:
        aoi = ee.Geometry.Rectangle(aoi)

    if type(date_range) == tuple:
        date_range = ee.DateRange(*date_range)

    image_collection = (
        ee.ImageCollection(collection)
        .filterBounds(aoi)
        .filterDate(date_range)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .select(band)
    )
    return image_collection


def get_image_list(image_collection: ee.ImageCollection) -> ee.List:
    """Get a list of images from the image collection.

    Args:
        image_collection: An ee.ImageCollection object representing the collection of images.

    Returns:
        An ee.List object representing the list of images.
    """
    return image_collection.toList(image_collection.size())


def len_image_list(image_list: ee.List) -> int:
    """Get the length of the image list.

    Args:
        image_list: An ee.List object representing the list of images.

    Returns:
        An integer representing the length of the image list.
    """
    return image_list.size().getInfo()


def get_image_from_list(image_list: ee.List, image_index: int = 0) -> ee.Image:
    """Get an image from the image list.

    Args:
        image_list: An ee.List object representing the list of images.
        image_index: An integer representing the index of the image to retrieve.

    Returns:
        An ee.Image object representing the retrieved image.
    """
    return ee.Image(image_list.get(image_index))


def get_list_of_images(image_list: ee.List) -> list:
    """Get a list of images from the image list.

    Args:
        image_list: An ee.List object representing the list of images.

    Returns:
        A list of ee.Image objects representing the list of images.
    """
    return [ee.Image(image_list.get(i)) for i in range(len_image_list(image_list))]


def get_image_id_with_retry(image: ee.Image, max_retries: int =5):
    for i in range(max_retries):
        try:
            return image.get('system:index').getInfo()
        except (
            requests.exceptions.ConnectionError, 
            urllib3.exceptions.ProtocolError,
            http.client.RemoteDisconnected,
            ) as e:
            print(f"Connection was dropped. Retry {i+1} of {max_retries}")
            time.sleep(5)  # Wait for 5 seconds before retrying
            continue
    print("Failed to get the image id after several retries")
    return None


def get_image_id(image: ee.Image) -> str:
    """Get the ID of the image.

    Args:
        image: An ee.Image object representing the image.

    Returns:
        A string representing the ID of the image.
    """
    return get_image_id_with_retry(image)


def get_crs(image: ee.Image) -> str:
    """Get the CRS of the image.

    Args:
        image: An ee.Image object representing the image.

    Returns:
        A string representing the CRS of the image.
    """
    projection = image.select(0).projection().getInfo()
    return projection['crs']


def get_crs_transform(image: ee.Image) -> list:
    """Get the CRS transform of the image.

    Args:
        image: An ee.Image object representing the image.

    Returns:
        A list representing the CRS transform of the image.
    """
    projection = image.select(0).projection().getInfo()
    return projection['transform']


def save_image_timestamps_to_csv(
    image_list: ee.List, 
    filename: str = 'timestamps_sar_images.csv'
    ) -> None:
    """
    Save the timestamps and image IDs of a list of images to a CSV file.

    Args:
        image_list (ee.List): A list of images.
        filename (str, optional): The name of the CSV file to save the timestamps and image IDs to. 
            Defaults to 'timestamps_sar_images.csv'.
    """
    with open(filename, 'a') as f:
        col_headers = ['TIMESTAMP', 'IMAGE_ID']
        writer = csv.writer(f)
        writer.writerow(col_headers)
        new_row = []
        # Iterate over the list to extract the IDs and Timestamps of each image
        for i in range(image_list.size().getInfo()):
            image = ee.Image(image_list.get(i))
            image_id = image.get('system:index').getInfo()
            ee_date = ee.Date(image.get('system:time_start')).format().getInfo()
            new_row = [ee_date, image_id]
            writer.writerow(new_row)
            new_row = []
            

def load_image_timestamps_from_csv(
    filename: str = 'timestamps_sar_images.csv'
    ) -> pd.DataFrame:
    """
    Load image timestamps from a CSV file.

    Args:
        filename (str): The path to the CSV file containing the timestamps. Default is 'timestamps_sar_images.csv'.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded timestamps.

    """
    df = pd.read_csv(filename)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    return df