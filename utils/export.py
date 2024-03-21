import ee
from gee import get_crs, get_crs_transform


def export_image_to_gcs(
        image,
        bucket_name: str,
        filename: str = None,
        resolution: int = 10,
        region_of_interest: list = None,
        export_format: str = 'GeoTIFF'
        ):
    """"""
    image_crs = get_crs(image)
    image_transform = get_crs_transform(image)

    if not filename:
        filename = f'{image}'

    if not region_of_interest:
        region_of_interest = image.geometry()

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description='SAR Sentinel-1',
        bucket=bucket_name,
        fileNamePrefix=filename,
        scale=resolution,  # Resolution in m per pixel. Default: 1000
        crs=image_crs,
        crsTransform=image_transform,
        region=region_of_interest,
        fileFormat=export_format,
        maxPixels=1e13,
    )
    task.start()
    return task
