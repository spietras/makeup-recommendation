load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "faceparsing",
    visibility = ["//visibility:public"],
    imports = ["."],
    srcs = glob(["faceparsing/**/*.py"]),
    data = glob(["faceparsing/resources/**/*"])
)