import numpy as np


def normalize_photo(photo):
    np_array = np.asarray(photo)
    if np_array.ndim != 3:
        raise ValueError("Invalid number of dimensions: {}. Should be 3".format(np_array.ndim))
    if np_array.shape[2] != 3:
        raise ValueError("Invalid number of channels: {}. Should be 3".format(np_array.shape[2]))
    if np.issubdtype(np_array.dtype, np.floating):
        return (np_array * 255).astype(np.uint8)
    if np.issubdtype(np_array.dtype, np.integer):
        return np_array.astype(np.uint8)
    raise ValueError("Invalid data type: {}".format(np_array.dtype))
