import requests
import shutil
import ee


def get_image(index: int, coordinates: tuple, dates: tuple, dimensions: str = '600x600', region_size: int = 3000):
    """Handle the HTTP requests to download an image."""

    def mask_edge(image):
        edge = image.lt(-30.0)
        masked_image = image.mask().And(edge.Not())
        return image.updateMask(masked_image)

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

    filename = 'img_%05d.jpg' % index
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", index)
