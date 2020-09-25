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

py_test(
    name = "test_utils",
    size = "small",
    srcs = ["test_utils.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
    ],
)

py_test(
    name = "test_conversion",
    size = "small",
    srcs = ["test_conversion.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
    ],
)
