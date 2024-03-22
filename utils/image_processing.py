import numpy as np


def normalize_image(db_img):
    """
    Equalizes the input image by applying a histogram equalization technique.
    Parameters:
        img (numpy.ndarray): The input image to be equalized.
    Returns:
        numpy.ndarray: The equalized image.
    Note:
        - The input image is assumed to be in grayscale.
        - The equalized image is returned as a 'uint8' array.
        - The histogram equalization technique is applied by calculating the histogram of the image,
          cumulatively summing the histogram values, and mapping the cumulative sum values to the
          range [0, 255].
        - The input image is shifted by 100 before calculating the histogram to avoid negative values.
        - The equalized image is obtained by mapping the values of the input image to the equalized
          cumulative sum values.
    """
    hist = np.histogram((db_img + 100).flatten(), 256, [0, 256])[0]

    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0)
    return cdf[(db_img + 100).astype('uint8')].astype('uint8')


def stretch_image(db_img: np.ndarray, min: int = -30, max: int = 0) -> np.ndarray:
    """
    Stretches the given image by clipping its values between the specified minimum and maximum values
    and MaxMin scaling it to [0, 255].

    Parameters:
        db_img (np.ndarray): The input image to be stretched.
        min (int): The minimum value for clipping.
        max (int): The maximum value for clipping.

    Returns:
        np.ndarray: The stretched image, scaled between 0 and 255, with any NaN values converted to 0.
    """
    cliped_image = np.clip(db_img, min, max)
    scaled_image = (cliped_image - min) / (max - min)
    scaled_image = np.nan_to_num(scaled_image)
    return (scaled_image * 255).astype('uint8')


def split_image(image: np.ndarray) -> np.ndarray:
    """
    Splits an image into a tiled array based on its dimensions.

    Parameters:
        image (numpy.ndarray): The input image to be split.

    Returns:
        numpy.ndarray: The tiled array representing the split image.
    """
    width, height = image.shape
    num_x_tiles, num_y_tiles = int(np.floor(width / 640)), int(np.floor(height / 640))

    x_remainder, y_remainder = width % 640, height % 640

    tiled_array = image[:-x_remainder, :-y_remainder].reshape(num_x_tiles, 640, num_y_tiles, 640)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array
