load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "automakeup",
    visibility = ["//visibility:public"],
    imports = ["."],
    srcs = glob(["automakeup/**/*.py"]),
    data = glob(["automakeup/resources/**/*"])
)