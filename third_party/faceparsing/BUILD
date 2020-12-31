load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "faceparsing",
    srcs = glob(["faceparsing/**/*.py"]),
    data = glob(["faceparsing/resources/**/*"]),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//modelutils",
    ],
)
