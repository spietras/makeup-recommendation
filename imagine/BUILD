load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "imagine",
    visibility = ["//visibility:public"],
    imports = ["."],
    srcs = glob(["imagine/**/*.py"]),
    data = glob(["imagine/resources/**/*"])
)