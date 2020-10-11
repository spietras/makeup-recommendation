load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_segment",
    size = "medium",
    srcs = ["test_segment.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
        "//third_party/faceparsing",
    ],
)

py_test(
    name = "test_operations",
    size = "small",
    srcs = ["test_operations.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
    ],
)
