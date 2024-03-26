from collections import Counter
import ee
import http.client
import json
import requests
import time
from typing import List, Optional
import sys
import urllib3

# sys.append('utils')
from gee import *  # get_crs, get_crs_transform, get_image_id


MAX_RETRIES = 5


def export_image_to_gcs(
        image: ee.Image,
        # image_id: str,
        bucket_name: str,
        folder_name: str,
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
        image_id = get_image_id(image)
        filename = f'{folder_name}/{image_id}'

    if not region_of_interest:
        region_of_interest = image.geometry()

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description='Export SAR Sentinel-1',
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


def get_task_status_with_retry(task, max_retries=MAX_RETRIES):
    for i in range(max_retries):
        try:
            return task.status()
        except (
            requests.exceptions.ConnectionError, 
            urllib3.exceptions.ProtocolError,
            http.client.RemoteDisconnected,
            ) as e:
            print(f"Connection was dropped. Retry {i+1} of {max_retries}")
            time.sleep(5)  # Wait for 5 seconds before retrying
            continue
    print("Failed to get the task status after several retries")
    return None


def get_task_status(task):
    return get_task_status_with_retry(task)


def get_task_id(task_status):
    return task_status['id']


def get_task_state(task_status):
    return task_status['state']


def get_task_status_from_id(task_id: str):
    return ee.data.getTaskStatus(task_id)


# TODO: Update status only if the task status is different from the current status
def update_task_statuses(tasks, task_statuses):
    for task in tasks:
        task_status = get_task_status(task)
        task_id = get_task_id(task_status)
        task_state = get_task_state(task_status)
        task_statuses[task_id] = task_state
        # time.sleep(1)  # Wait for 1 second before checking the next task
    return task_statuses


def update_task_states_counts(task_statuses):
        statuses = list(task_statuses.values())
        counts = Counter(statuses)
        counts_completed = counts['COMPLETED']
        counts_failed = counts['FAILED']
        counts_running = counts['RUNNING']
        counts_ready = counts['READY']
        return counts_completed, counts_failed, counts_running, counts_ready
