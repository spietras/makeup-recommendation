load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "learn",
    srcs = glob(["learn.py"]),
    data = glob(["resources/**/*"]),
    deps = [
        "//ganette",
    ],
)

py_library(
    name = "ganette",
    srcs = glob(["ganette/**/*.py"]),
    data = glob(["ganette/resources/**/*"]),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//modelutils",
    ],
)
