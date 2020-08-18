import inspect
from abc import ABC, abstractmethod
from typing import BinaryIO, TextIO

from flask import request, abort, jsonify


def call_with_dict_args(f, args):
    return f(**args)


def normalize_annotation(annotation):
    return annotation if annotation is not inspect.Parameter.empty else None


def get_function_params(f):
    return {p.name: normalize_annotation(p.annotation) for p in inspect.signature(f).parameters.values()}


class EndpointHandler:
    def __init__(self, function):
        self.function = function
        self.required_params = get_function_params(self.function)

    @staticmethod
    def _add_present(handler, dict, p_name):
        try:
            value = handler.get_converted_value(request, p_name)
        except ValueError:
            raise ValueError("Can't convert {} to {}".format(p_name, handler.get_type_str()))
        dict[p_name] = value

    @staticmethod
    def _add_missing(handler, dict, p_name):
        dict_str = handler.get_param_dict_str()
        if dict_str not in dict:
            dict[dict_str] = []
        dict[handler.get_param_dict_str()].append(p_name)

    def _get_params(self):
        converted_params, missing_params, missing_once = {}, {}, False
        for p_name, p_type in self.required_params.items():
            handler = ParameterHandlerFactory.get(p_type)
            if handler.is_present(request, p_name):
                if not missing_once:
                    self._add_present(handler, converted_params, p_name)
            else:
                self._add_missing(handler, missing_params, p_name)
                missing_once = True
        return converted_params, missing_params

    @staticmethod
    def _get_missing_str(missing):
        def format_list(list):
            return ", ".join("'{}'".format(s) for s in list)

        dict_missing_strs = ["{} from {}".format(format_list(params), dict) for dict, params in missing.items()]
        return "Missing parameters: {}".format(' and '.join(dict_missing_strs))

    def __call__(self, *args, **kwargs):
        converted_params, missing_params = None, None
        try:
            converted_params, missing_params = self._get_params()
        except Exception as e:
            abort(400, str(e))
        if missing_params:
            abort(400, self._get_missing_str(missing_params))
        try:
            return jsonify(call_with_dict_args(self.function, converted_params))
        except Exception as e:
            abort(400, str(e))


class ParameterHandler(ABC):
    def __init__(self, p_type):
        self.type = p_type

    def get_type_str(self):
        return self.type.__name__

    @staticmethod
    @abstractmethod
    def get_param_dict(req):
        return NotImplemented

    @staticmethod
    @abstractmethod
    def get_param_dict_str():
        return NotImplemented

    def is_present(self, req, p_name):
        return p_name in self.get_param_dict(req)

    def get_converted_value(self, req, p_name):
        return self.get_param_dict(req).get(p_name)


class ArgsParameterHandler(ParameterHandler):
    @staticmethod
    def get_param_dict(req):
        return req.args

    @staticmethod
    def get_param_dict_str():
        return "arguments"


class NumericParameterHandler(ArgsParameterHandler):
    def get_converted_value(self, req, p_name):
        return self.type(super().get_converted_value(req, p_name))


class FilesParameterHandler(ParameterHandler):
    @staticmethod
    def get_param_dict(req):
        return req.files

    @staticmethod
    def get_param_dict_str():
        return "files"


class ParameterHandlerFactory:
    HANDLERS_ASSOCIATION = {
        BinaryIO: FilesParameterHandler,
        TextIO: FilesParameterHandler,
        int: NumericParameterHandler,
        float: NumericParameterHandler,
    }

    @staticmethod
    def normalize_type(p_type):
        return p_type if p_type else str

    @staticmethod
    def get(p_type):
        p_type = ParameterHandlerFactory.normalize_type(p_type)
        return ParameterHandlerFactory.HANDLERS_ASSOCIATION.get(p_type, ArgsParameterHandler)(p_type)
