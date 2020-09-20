import cv2
import numpy as np


def biggest_contour(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.ndim == 2:
        binary_mask = np.expand_dims(binary_mask, 2)
    contours, _ = cv2.findContours(binary_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([[]], dtype=np.int)
    return np.atleast_2d(max(contours, key=cv2.contourArea).squeeze())


def mass_center(contour):
    if contour is None or len(contour) == 0 or contour.size == 0:
        return None
    moments = cv2.moments(contour)
    return np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']], dtype=np.float)


def bounding_rect(contour):
    if contour is None or len(contour) == 0 or contour.size == 0:
        return None
    return cv2.boundingRect(contour)


def crop(img, rect):
    return img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


def erode(img, size, bg=0):
    org_shape = img.shape
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * size + 1, 2 * size + 1), (size, size))
    return cv2.erode(img, element, borderValue=bg).reshape(org_shape)


def squarisize(rect):
    bigger_dim = int(np.argmax([rect[2], rect[3]]))
    new_rect = np.empty(4, dtype=np.int)
    new_rect[bigger_dim] = rect[bigger_dim]
    new_rect[1 - bigger_dim] = rect[1 - bigger_dim] - int((rect[bigger_dim + 2] - rect[1 - bigger_dim + 2]) / 2)
    new_rect[2] = rect[bigger_dim + 2]
    new_rect[3] = rect[bigger_dim + 2]
    return tuple(new_rect)


def safe_rect(rect, im_w, im_h):
    x, y, w, h = rect
    if w > im_w:
        raise ValueError("Can't safe rect when rect width {} is bigger than image width {}".format(w, im_w))
    if h > im_h:
        raise ValueError("Can't safe rect when rect height {} is bigger than image height {}".format(h, im_h))

    x = max(x, 0)
    y = max(y, 0)
    x -= max(x + w - im_w, 0)
    y -= max(y + h - im_h, 0)

    return x, y, w, h


def circle_mask(shape, center, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask != 0
