load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_normalization",
    size = "small",
    srcs = ["test_normalization.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
    ],
)
