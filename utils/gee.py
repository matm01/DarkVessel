import ee


SENTINEL1_GRD = 'COPERNICUS/S1_GRD'


def format_date(date: str) -> ee.Date:
    """Format the date string into an ee.Date object.

    Args:
        date: A string representing the date.

    Returns:
        An ee.Date object representing the formatted date.
    """
    return ee.Date(date)


def generate_date_range(start: str, end: str, timedelta: int = 1) -> ee.DateRange:
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


def get_image_from_id(image_id: str, collection: str = SENTINEL1_GRD) -> ee.Image:
    """Get an image from the image ID.

    Args:
        image_id: A string representing the ID of the image.
        collection: A string representing the image collection.

    Returns:
        An ee.Image object representing the retrieved image.
    """
    image = ee.Image(f'{collection}/{image_id}')
    return image


def get_image_id(image: ee.Image) -> str:
    """Get the ID of the image.

    Args:
        image: An ee.Image object representing the image.

    Returns:
        A string representing the ID of the image.
    """
    return image.get('system:index').getInfo()


def get_image_date(image: ee.Image) -> ee.Date:
    """Get the date of the image.

    Args:
        image: An ee.Image object representing the image.

    Returns:
        An ee.Date object representing the date of the image.
    """
    return ee.Date(image.get('system:time_start'))


def get_image_timestamp(image: ee.Image) -> str:
    """Get the timestamp of the image.

    Args:
        image: An ee.Image object representing the image.

    Returns:
        A string representing the timestamp of the image.
    """
    return ee.Date(image.get('system:time_start')).format().getInfo()


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