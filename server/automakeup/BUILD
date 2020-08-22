load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "automakeup",
    visibility = ["//visibility:public"],
    imports = ["."],
    srcs = glob(["automakeup/**/*.py"]),
    data = glob(["automakeup/resources/**/*"]),
    deps = [
        requirement("dlib"),
        requirement("opencv-python"),
        requirement("openface")
    ]
)