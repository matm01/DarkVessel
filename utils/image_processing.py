import numpy as np
import matplotlib.pyplot as plt


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
    scaled_image = (cliped_image - min) * 255 / (max - min)
    scaled_image = np.nan_to_num(scaled_image, nan=0.0)
    return (scaled_image).astype('uint8')


def resize_image(image: np.ndarray) -> np.ndarray:
    """
    Resizes an image to a multiple of 640 pixels in both dimensions.

    Parameters:
        image (np.ndarray): The input image to be resized.

    Returns:
        np.ndarray: The resized image with dimensions that are a multiple of 640 pixels.
    """
    height, width = image.shape
    x_fill, y_fill = 640 - (width % 640), 640 - (height % 640)

    rows = np.ndarray([y_fill, image.shape[1]], dtype=np.uint8)
    rows.fill(0)
    image = np.vstack([image, rows])

    cols = np.ndarray([image.shape[0], x_fill], dtype=np.uint8)
    cols.fill(0)
    image = np.hstack([image, cols])

    return image


def split_image(image: np.ndarray) -> np.ndarray:
    """
    Splits an image into a tiled array based on its dimensions.

    Parameters:
        image (numpy.ndarray): The input image to be split.

    Returns:
        numpy.ndarray: The tiled array representing the split image.
    """
    num_x_tiles, num_y_tiles = int(image.shape[0] / 640), int(image.shape[1] / 640)

    tiled_array = image.reshape(num_x_tiles, 640, num_y_tiles, 640)
    tiled_array = tiled_array.swapaxes(1, 2)

    return tiled_array


def plot_img_and_hist(img: np.ndarray, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    Args:
        img: A NumPy array of shape (N, M).
        bins: Number of histogram bins.

    """

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    ax2.hist(img.flatten(), bins=bins)
    ax2.set_title('Histogram')
    ax2.set_xlim(0, 256)

    ax2.margins(0)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def histogram_stretch(img_linear: np.ndarray, scale_factor=8) -> np.ndarray:
    """
    Applies histogram stretching to an image.

    Args:
        img_linear (ndarray): The linear image to be stretched.
        scale_factor (int, optional): The scale factor used for stretching. Defaults to 8.

    Returns:
        ndarray: The stretched image as a numpy array of type 'uint8'

    Raises:
        None

    Examples:
        >>> img = np.array([0.001, 0.01, 0.1, 1])
        >>> histogram_stretch(img)
        array([0,   5,  57, 255])
    """
    mu = 1 / (scale_factor * np.nanmedian(img_linear))
    hs_img = 255 * mu * img_linear
    hs_img = np.clip(hs_img, 0, 255)
    hs_img = np.nan_to_num(hs_img, nan=255.0)

    return hs_img.astype('uint8')


def arctangent_stretch(img_linear, scale_factor=4000):
    """
    Calculate the arctangent stretch of the input image.

    Parameters:
    img_linear (numpy.ndarray): The input image as a numpy array.
    scale_factor (int, optional): The scale factor for stretching the arctangent function. Defaults to 4000.

    Returns:
    numpy.ndarray: The arctangent stretched image as a numpy array of type 'uint8'.
    """
    nu = scale_factor
    at_img = 255 * (2 / np.pi) * np.arctan((img_linear * nu) / (2**16)) * 255
    at_img = np.clip(at_img, 0, 255)
    at_img = np.nan_to_num(at_img, nan=255.0)

    return at_img.astype('uint8')


def quarter_power_stretch(img_linear, scale_factor=4):
    """
    Generate a quarter power stretch image from a given linear image.

    Parameters:
    img_linear (numpy.ndarray): The input linear image.
    scale_factor (int, optional): The scaling factor for the stretch. Defaults to 4.

    Returns:
    numpy.ndarray: The quarter power stretched image as a numpy array of type uint8.
    """
    beta = 1 / (scale_factor * np.nanmedian(np.sqrt(img_linear)))
    qp_img = 255 * beta * np.sqrt(img_linear)
    qp_img = np.clip(qp_img, 0, 255)
    qp_img = np.nan_to_num(qp_img, nan=255.0)

    return qp_img.astype('uint8')


def to_linear_magnitude(db_image, min=-30, max=0):
    """
    Calculate the linear magnitude of the input db_image using the given minimum and maximum values.

    Parameters:
    db_image (numpy.ndarray): The input db_image in decibels.
    min (int, optional): The minimum value to clip the db_image to. Default is -30.
    max (int, optional): The maximum value to clip the db_image to. Default is 0.

    Returns:
    numpy.ndarray: The linear magnitude image.
    """
    img = np.clip(db_image, min, max)
    img = 10 ** (img / 10)
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))

    return img
