load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "facenet",
    srcs = glob(["facenet/**/*.py"]),
    data = glob(["facenet/resources/**/*"]),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//modelutils",
    ],
)
