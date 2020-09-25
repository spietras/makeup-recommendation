load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "automakeup",
    srcs = glob(["automakeup/**/*.py"]),
    data = glob(["automakeup/resources/**/*"]),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//imagine",
    ],
)
