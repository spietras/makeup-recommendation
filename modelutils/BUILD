load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "modelutils",
    srcs = glob(["modelutils/**/*.py"]),
    data = glob(["modelutils/resources/**/*"]),
    imports = ["."],
    visibility = ["//visibility:public"],
)
