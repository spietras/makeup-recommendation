import importlib.resources as pkg_resources


def _get_resource(name):
    return pkg_resources.path("automakeup.resources", name)


def dlib_predictor_path():
    return _get_resource("dlib_face_predictor_68.dat")


def ganette_model_path():
    return _get_resource("ganette.pkl")


def ganette_x_scaler_path():
    return _get_resource("ganette_x_scaler.pkl")


def ganette_y_scaler_path():
    return _get_resource("ganette_y_scaler.pkl")
