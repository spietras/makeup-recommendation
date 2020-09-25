import cv2
import numpy as np

from imagine.functional.functional import ImageOperation
from imagine.shape.figures import Rect


def biggest_contour(binary_mask):
    """
    Find biggest contour in binary image

    Args:
        binary_mask: numpy array of shape (height, width, 1) or (height, width) with values greater than zero as
                     non-background pixels

    Returns:
        numpy array of shape (N, 2) with points representing the contour or empty array if not found
    """
    binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.ndim == 2:
        binary_mask = np.expand_dims(binary_mask, 2)
    contours, _ = cv2.findContours(binary_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([[]], dtype=np.int)
    return np.atleast_2d(max(contours, key=cv2.contourArea).squeeze())


def mass_center(contour):
    """
    Calculate mass center of contour

    Args:
        contour: numpy array of shape (N, 2) with points representing the contour

    Returns:
        numpy array of shape (2,) with (x,y)-position of mass center or None if contour is empty
    """
    if contour is None or len(contour) == 0 or contour.size == 0:
        return None
    moments = cv2.moments(contour)
    return np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']], dtype=np.float)


def bounding_rect(contour):
    """
    Find Rect that bounds the contour

    Args:
        contour: numpy array of shape (N, 2) with points representing the contour
    """
    if contour is None or len(contour) == 0 or contour.size == 0:
        return None
    return Rect.from_cv(cv2.boundingRect(contour))


def crop(img, rect):
    """
    Crop image to Rect

    Args:
        img: numpy array of shape (height, width, ...)
        rect: Rect to crop to. Must be inside image bounds.

    Returns:
        numpy array with the same number of dimensions as img, but height and width cropped
    """
    return img[rect.top:rect.bottom, rect.left:rect.right]


def erode(img, size, bg=0):
    """
    Perform erosion - "shrinking" of object area in an image

    For three-channel image, erosion is performed separately for each channel

    Args:
        img: numpy array of shape (height, width, channels) or (height, width)
        size: erosion size
        bg: value beside the edges of image

    Returns:
        numpy array of the same shape as img with eroded image
    """
    org_shape = img.shape
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * size + 1, 2 * size + 1), (size, size))
    return cv2.erode(img, element, borderValue=bg).reshape(org_shape)


def squarisize(rect):
    """Make Rect a square along biggest dimension"""
    rect = rect.to_cv()
    bigger_dim = int(np.argmax([rect[2], rect[3]]))
    new_rect = np.empty(4, dtype=np.int)
    new_rect[bigger_dim] = rect[bigger_dim]
    new_rect[1 - bigger_dim] = rect[1 - bigger_dim] - int((rect[bigger_dim + 2] - rect[1 - bigger_dim + 2]) / 2)
    new_rect[2] = rect[bigger_dim + 2]
    new_rect[3] = rect[bigger_dim + 2]
    return Rect.from_cv(new_rect)


def safe_rect(rect, img_dim, allow_scaling=False):
    """
    Move Rect to image bounds

    Args:
        rect: Rect to consider
        img_dim: tuple with image shape (height, width, ...)
        allow_scaling: True if Rect is allowed to be scaled down in case of being bigger than the whole image.
                       Defaults to False.
    """

    _, _, w, h = rect.to_cv()

    if allow_scaling:
        factor = min(img_dim[1] / w, img_dim[0] / h)
        if factor < 1:
            rect = rect.scale(factor)
    else:
        if w > img_dim[1]:
            raise ValueError("Can't safe rect when rect width {} is bigger than image width {}".format(w, img_dim[1]))
        if h > img_dim[0]:
            raise ValueError("Can't safe rect when rect height {} is bigger than image height {}".format(h, img_dim[0]))

    x, y, w, h = rect.to_cv()
    x = max(x, 0)
    y = max(y, 0)
    x -= max(x + w - img_dim[1], 0)
    y -= max(y + h - img_dim[0], 0)

    return Rect.from_cv((x, y, w, h))


def circle_mask(shape, center, radius):
    """
    Make boolean mask with circle

    Args:
        shape: tuple with desired shape (height, width, ...)
        center: tuple with (x, y) center positon
        radius: circle radius

    Returns:
        numpy array of shape (height, width) with True in circle area
    """

    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask != 0


def resize(img, shape, interpolation=cv2.INTER_LINEAR):
    """
    Resize image

    Args:
        img: numpy array of shape (height1, width1, channels) with image data
        shape: tuple with desired shape (height2, width2, ...)
        interpolation: opencv interpolation method to use

    Returns:
        numpy array of shape (height2, width2, channels) with resized image data
    """

    return cv2.resize(img, (shape[1], shape[0]), interpolation=interpolation)


# Functional Interface

class Crop(ImageOperation):
    """Crop image to Rect"""

    def __init__(self, rect):
        """
        Args:
            rect: Rect to crop to. Must be inside image bounds.
        """

        super().__init__()
        self.rect = rect

    def perform(self, img, **kwargs):
        return crop(img, self.rect)


class Erode(ImageOperation):
    """Perform erosion - "shrinking" of object area in an image"""

    def __init__(self, size, bg=0):
        """
        Args:
            size: erosion size
            bg: value beside the edges of image
        """

        super().__init__()
        self.size = size
        self.bg = bg

    def perform(self, img, **kwargs):
        return erode(img, self.size, self.bg)


class Resize(ImageOperation):
    """Resize image"""

    def __init__(self, shape, interpolation=cv2.INTER_LINEAR):
        """
        Args:
            shape: tuple with desired shape (height2, width2, ...)
            interpolation: opencv interpolation method to use
        """

        super().__init__()
        self.shape = shape
        self.interpolation = interpolation

    def perform(self, img, **kwargs):
        return resize(img, self.shape, self.interpolation)
