import ee


SENTINEL1_GRD = 'COPERNICUS/S1_GRD'

def format_date(date: str):
    return ee.Date(date)


def generate_date_range(start: str, end: str, timedelta: int = 1):
    start = format_date(start)
    end = format_date(end)
    n_date_range = end.difference(start, 'days').getInfo()
    for _ in range(n_date_range):
        next_day = start.advance(timedelta, 'day')
        date_range = ee.DateRange(start, next_day)
        start = start.advance(timedelta, 'day')
        yield date_range


def get_image_collection(
        aoi: list or ee.Geometry,
        date_range: tuple or ee.Date,
        band: str = ['VH', 'VV'],
        collection=SENTINEL1_GRD
        ):
    """Get collection of SAR images """
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
    # desc = image_collection.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    # asc = image_collection.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    return image_collection


def get_image_list(image_collection):
    return image_collection.toList(image_collection.size())


def len_image_list(image_list):
    return image_list.size().getInfo()


def get_image_from_list(image_list, image_index: int = 0):
    """Get first image by default"""
    return ee.Image(image_list.get(image_index))


def get_image_from_id(image_id, collection=SENTINEL1_GRD):
    image = ee.Image(f'{collection}/{image_id}')
    return image


def get_image_id(image):
    return image.get('system:index').getInfo()


def get_image_date(image):
    return ee.Date(image.get('system:time_start'))


def get_image_timestamp(image):
    return ee.Date(image.get('system:time_start')).format().getInfo()


def get_crs(image):
    projection = image.select(0).projection().getInfo()
    return projection['crs']


def get_crs_transform(image):
    projection = image.select(0).projection().getInfo()
    return projection['transform']