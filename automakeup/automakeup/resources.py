from importlib import resources

import automakeup


def read_resource(path):
    """
    Reads and returns whole content of the resource as bytes (file is closed)

    Args:
        path: path to the resource (relative to resource directory)

    Returns:
        resource content as bytes
    """
    return resources.read_binary("{}.{}".format(automakeup.__package_name__, automakeup.__resources__), path)


def open_resource(path):
    """
    Opens the resource file and returns its handle
    To use with 'with' statement

    Args:
        path: path to the resource (relative to resource directory)

    Returns:
        Resource file handle
    """
    return resources.open_binary("{}.{}".format(automakeup.__package_name__, automakeup.__resources__), path)


def get_resource_path(path):
    """
    Resolves path to the resource and returns an object representing the absolute path
    The returned object can actually be used as a context manager (in 'with' statement)
    This is useful when the package is zipped, so the object will automatically extract the resource and clean it after

    For example:

    with get_resource_path("file.txt") as path:
        with open(str(path)) as f:
            f.read()

    Args:
        path: path to the resource (relative to resource directory)

    Returns:
        Object representing the absolute path to the resource (as a context manager)
    """
    return resources.path("{}.{}".format(automakeup.__package_name__, automakeup.__resources__), path)
