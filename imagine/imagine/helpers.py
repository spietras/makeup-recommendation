import numpy as np


def normalize_images(imgs):
    """
    Normalize float images to range [0-255] and converts all images to uint8\

    Args:
        imgs: numpy array of shape (N, width, height, C) or (width, height, C) where C is 1 or 3

    Returns:
        numpy array with the same shape as imgs and uint8 type
    """

    np_array = np.asarray(imgs)
    if np_array.ndim not in [3, 4]:
        raise ValueError("Invalid number of dimensions: {}. Should be 3 or 4".format(np_array.ndim))
    if np_array.shape[-1] not in [3, 1]:
        raise ValueError("Invalid number of channels: {}. Should be 3 or 1".format(np_array.shape[-1]))
    if np.issubdtype(np_array.dtype, np.floating):
        return denormalize_range(np_array)
    if np.issubdtype(np_array.dtype, np.integer):
        return np_array.astype(np.uint8)
    raise ValueError("Invalid data type: {}".format(np_array.dtype))


def normalize_range(img):
    return img.astype(np.float) / 255


def denormalize_range(img):
    return np.round(img * 255).astype(np.uint8)
