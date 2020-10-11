load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "mtcnn",
    srcs = glob(["mtcnn/**/*.py"]),
    data = glob(["mtcnn/resources/**/*"]),
    imports = ["."],
    visibility = ["//visibility:public"],
)
