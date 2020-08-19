"""Utility functions to manipulate Bazel files"""
import os
import textwrap
import json
from typing import Iterable, List, Dict, Set

from extract_wheels.lib import namespace_pkgs, wheel, purelib


def generate_build_file_contents(name: str, dependencies: List[str], pip_data_exclude: List[str]) -> str:
    """Generate a BUILD file for an unzipped Wheel

    Args:
        name: the target name of the py_library
        dependencies: a list of Bazel labels pointing to dependencies of the library

    Returns:
        A complete BUILD file as a string

    We allow for empty Python sources as for Wheels containing only compiled C code
    there may be no Python sources whatsoever (e.g. packages written in Cython: like `pymssql`).
    """

    data_exclude = ["**/*.py", "**/* *", "BUILD", "WORKSPACE"] + pip_data_exclude

    return textwrap.dedent(
        """\
        package(default_visibility = ["//visibility:public"])

        load("@rules_python//python:defs.bzl", "py_library")

        py_library(
            name = "{name}",
            srcs = glob(["**/*.py"], allow_empty = True),
            data = glob(["**/*"], exclude={data_exclude}),
            # This makes this directory a top-level in the python import
            # search path for anything that depends on this.
            imports = ["."],
            deps = [{dependencies}],
        )
        """.format(
            name=name, dependencies=",".join(dependencies), data_exclude=json.dumps(data_exclude)
        )
    )


def generate_requirements_file_contents(repo_name: str, targets: Iterable[str]) -> str:
    """Generate a requirements.bzl file for a given pip repository

    The file allows converting the PyPI name to a bazel label. Additionally, it adds a function which can glob all the
    installed dependencies. This is provided for legacy reasons and can be considered deprecated.

    Args:
        repo_name: the name of the pip repository
        targets: a list of Bazel labels pointing to all the generated targets

    Returns:
        A complete requirements.bzl file as a string
    """

    return textwrap.dedent(
        """\
        # Deprecated. This will be removed in a future release
        all_requirements = [{requirement_labels}]

        def requirement(name):
           name_key = name.replace("-", "_").replace(".", "_").lower()
           return "{repo}//pypi__" + name_key
        """.format(
            repo=repo_name, requirement_labels=",".join(sorted(targets))
        )
    )


def sanitise_name(name: str) -> str:
    """Sanitises the name to be compatible with Bazel labels.

    There are certain requirements around Bazel labels that we need to consider. From the Bazel docs:

        Package names must be composed entirely of characters drawn from the set A-Z, a–z, 0–9, '/', '-', '.', and '_',
        and cannot start with a slash.

    Due to restrictions on Bazel labels we also cannot allow hyphens. See
    https://github.com/bazelbuild/bazel/issues/6841

    Further, rules-python automatically adds the repository root to the PYTHONPATH, meaning a package that has the same
    name as a module is picked up. We workaround this by prefixing with `pypi__`. Alternatively we could require
    `--noexperimental_python_import_all_repositories` be set, however this breaks rules_docker.
    See: https://github.com/bazelbuild/bazel/issues/2636
    """

    return "pypi__" + name.replace("-", "_").replace(".", "_").lower()


def setup_namespace_pkg_compatibility(wheel_dir: str) -> None:
    """Converts native namespace packages and pkg_resource-style packages to pkgutil-style packages

    Namespace packages can be created in one of three ways. They are detailed here:
    https://packaging.python.org/guides/packaging-namespace-packages/#creating-a-namespace-package

    'pkgutil-style namespace packages' (2) works in Bazel, but 'native namespace packages' (1) and
    'pkg_resources-style namespace packages' (3) do not.

    We ensure compatibility with Bazel of methods 1 and 3 by converting them into method 2.

    Args:
        wheel_dir: the directory of the wheel to convert
    """

    namespace_pkg_dirs = namespace_pkgs.pkg_resources_style_namespace_packages(
        wheel_dir
    )
    if not namespace_pkg_dirs and namespace_pkgs.native_namespace_packages_supported():
        namespace_pkg_dirs = namespace_pkgs.implicit_namespace_packages(
            wheel_dir, ignored_dirnames=["%s/bin" % wheel_dir,],
        )

    for ns_pkg_dir in namespace_pkg_dirs:
        namespace_pkgs.add_pkgutil_style_namespace_pkg_init(ns_pkg_dir)


def extract_wheel(wheel_file: str, extras: Dict[str, Set[str]], pip_data_exclude: List[str]) -> str:
    """Extracts wheel into given directory and creates a py_library target.

    Args:
        wheel_file: the filepath of the .whl
        extras: a list of extras to add as dependencies for the installed wheel

    Returns:
        The Bazel label for the extracted wheel, in the form '//path/to/wheel'.
    """

    whl = wheel.Wheel(wheel_file)
    directory = sanitise_name(whl.name)

    os.mkdir(directory)
    whl.unzip(directory)

    # Note: Order of operations matters here
    purelib.spread_purelib_into_root(directory)
    setup_namespace_pkg_compatibility(directory)

    extras_requested = extras[whl.name] if whl.name in extras else set()

    sanitised_dependencies = [
        '"//%s"' % sanitise_name(d) for d in sorted(whl.dependencies(extras_requested))
    ]

    with open(os.path.join(directory, "BUILD"), "w") as build_file:
        contents = generate_build_file_contents(
            sanitise_name(whl.name), sanitised_dependencies, pip_data_exclude,
        )
        build_file.write(contents)

    os.remove(whl.path)

    return "//%s" % directory
