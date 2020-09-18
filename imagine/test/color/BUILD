load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_extract",
    size = "small",
    srcs = ["test_extract.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
    ],
)
