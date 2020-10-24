load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "main",
    srcs = glob(["main.py"]),
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup",
        "//preprocessing",
        "//third_party/facenet",
        "//third_party/faceparsing",
        "//third_party/mtcnn",
    ],
)

py_library(
    name = "preprocessing",
    srcs = glob(["preprocessing/**/*.py"]),
    data = glob(["preprocessing/resources/**/*"]),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//imagine",
    ],
)
