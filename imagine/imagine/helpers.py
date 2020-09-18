import cv2
import numpy as np


def normalize_photo(photo):
    np_array = np.asarray(photo)
    if np_array.ndim not in [3, 4]:
        raise ValueError("Invalid number of dimensions: {}. Should be 3 or 4".format(np_array.ndim))
    if np_array.shape[-1] != 3:
        raise ValueError("Invalid number of channels: {}. Should be 3".format(np_array.shape[-1]))
    if np.issubdtype(np_array.dtype, np.floating):
        return denormalize_range(np_array)
    if np.issubdtype(np_array.dtype, np.integer):
        return np_array.astype(np.uint8)
    raise ValueError("Invalid data type: {}".format(np_array.dtype))


def to_lab(img):
    return cv2.cvtColor(np.array(img, ndmin=3), cv2.COLOR_RGB2Lab).reshape(img.shape)


def to_rgb(img):
    return cv2.cvtColor(np.array(img, ndmin=3), cv2.COLOR_Lab2RGB).reshape(img.shape)


def normalize_range(img):
    return img.astype(np.float) / 255


def denormalize_range(img):
    return np.round(img * 255).astype(np.uint8)
