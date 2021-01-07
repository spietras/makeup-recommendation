load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "search",
    srcs = glob(["search.py"]),
    data = glob(["resources/**/*"]),
    deps = [
        "//ganette",
        "//imagine",
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
