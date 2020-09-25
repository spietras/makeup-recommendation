import dlib
import numpy as np


class Rect:
    """
    Rectangle representation

    In images top-left corner is considered to be the origin, hence all bottom > top and right > left

    Attributes:
        top - inclusive position of top edge
        bottom - exclusive position of bottom edge
        left - inclusive position of left edge
        right - exclusive position of right edge
    """

    def __init__(self, top, bottom, left, right):
        super().__init__()
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    @staticmethod
    def from_dlib(rect):
        """Construct Rect from dlib.rectangle"""
        return Rect(rect.top(), rect.bottom() + 1, rect.left(), rect.right() + 1)

    @staticmethod
    def from_cv(rect):
        """Construct Rect from opencv rectangle: (x, y, w, h)"""
        return Rect(rect[1], rect[1] + rect[3], rect[0], rect[0] + rect[2])

    def to_dlib(self):
        """Convert Rect to dlib.rectangle"""
        return dlib.rectangle(self.left, self.top, self.right - 1, self.bottom - 1)

    def to_cv(self):
        """Convert Rect to opencv rectangle: (x, y, w, h)"""
        return self.left, self.top, self.right - self.left, self.bottom - self.top

    def scale(self, factor, origin=None):
        """
        Scale Rect

        Args:
            factor: factor to scale by
            origin: tuple (x, y) with point to consider as scaling origin. if None use Rect center

        Returns:
            Scaled Rect
        """
        if origin is None:
            origin = (0.5 * (self.left + self.right), 0.5 * (self.top + self.bottom))

        new_bounds = (self.top - origin[1],
                      self.bottom - origin[1],
                      self.left - origin[0],
                      self.right - origin[0])
        new_bounds = (factor * np.asarray(new_bounds)).astype(np.int)
        t, b, l, r = (int(new_bounds[0] + origin[1]),
                      int(new_bounds[1] + origin[1]),
                      int(new_bounds[2] + origin[0]),
                      int(new_bounds[3] + origin[0]))
        return Rect(t, b, l, r)

    def __eq__(self, o):
        if isinstance(o, Rect):
            return self.top == o.top and self.bottom == o.bottom and self.left == o.left and self.right == o.right
        return False
