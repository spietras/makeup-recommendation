import importlib.resources as pkg_resources


def _get_resource(name):
    return pkg_resources.path("automakeup.resources", name)


def dlib_predictor_path():
    return _get_resource("dlib_face_predictor_68.dat")
