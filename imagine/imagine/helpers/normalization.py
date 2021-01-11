import numpy as np

from imagine.functional.functional import ImageOperation, Batchable


def normalize_images(imgs):
    """
    Normalize float images to range [0-255] and converts all images to uint8

    Args:
        imgs: numpy array of shape (N, height, width, C) or (height, width, C) where C is 1 or 3

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
    """
    Divides input by 255 and converts to float type

    Args:
        img: any numpy array
    """
    return img.astype(np.float) / 255


def denormalize_range(img):
    """
    Multiplies input by 255 and converts to uint8 type

    Args:
        img: any numpy array
    """
    return np.rint(img * 255).astype(np.uint8)


# Functional Interface

class ToUInt8(Batchable, ImageOperation):
    """Normalize float images from range [0-1] to range [0-255] and converts all images to uint8. Batchable."""

    def perform(self, imgs, **kwargs):
        return normalize_images(imgs)


class Round(Batchable, ImageOperation):
    """Rounds images to uint8. Batchable."""

    def perform(self, imgs, **kwargs):
        return np.rint(imgs).astype(np.uint8)


class Normalize(Batchable, ImageOperation):
    """Divides input by 255 and converts to float type. Batchable."""

    def perform(self, imgs, **kwargs):
        return normalize_range(imgs)


class Denormalize(Batchable, ImageOperation):
    """Multiplies input by 255 and converts to uint8 type. Batchable."""

    def perform(self, imgs, **kwargs):
        return denormalize_range(imgs)
