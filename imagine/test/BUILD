load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_helpers",
    size = "small",
    srcs = ["test_helpers.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
    ],
)
