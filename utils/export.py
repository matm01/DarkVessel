import ee
from gee import get_crs, get_crs_transform
import json
from typing import List, Optional
import ee


def export_image_to_gcs(
    image: ee.Image,
    bucket_name: str,
    filename: Optional[str] = None,
    resolution: int = 10,
    region_of_interest: Optional[List[float]] = None,
    export_format: str = 'GeoTIFF'
    ) -> ee.batch.Task:
    """
    Exports an image to Google Cloud Storage (GCS).

    Args:
    image: The image to be exported.
    bucket_name: The name of the GCS bucket to export the image to.
    filename: The name of the exported file. If not provided, the name of the image will be used.
    resolution: The resolution in meters per pixel. Default is 10.
    region_of_interest: The region of interest to export. If not provided, the entire image will be exported.
    export_format: The format of the exported file. Default is 'GeoTIFF'.

    Returns:
    The task object representing the export process.

    """
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


# Export status logging
def get_task_status(task):
    return task.status()


def log_task_status(task, log_file='task_status.json'):
    # Check the status of the task
    status = get_task_status(task)

    # # Prepare the log entry
    log_entry = {
        'task_id': status['id'],
        'status': status['state'],
    }

    # Append the log entry to the JSON file
    with open(log_file, 'a') as file:
        file.write(json.dumps(status))


def print_dict(d):
    """Print formatted dictionary."""
    for k, v in d.items():
        print(k, ":", v)
    print()


def halt_until_completed(task, max_time=1000):
    """Wait until task has completed or exit if failed."""
    failed_state = [
        "FAILED",
        "CANCELLED",
        "CANCEL_REQUESTED",
        "UNSUBMITTED",
    ]
    while task.status()["state"] != "COMPLETED":
        if task.status()["state"] in failed_state:
            break


def check_task(task):
    print_dict(task.status())
    print("running task ...\n")
    halt_until_completed(task)
    print_dict(task.status())
