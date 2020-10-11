load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_bounding",
    size = "medium",
    srcs = ["test_bounding.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup",
        "//third_party/mtcnn",
    ],
)

py_test(
    name = "test_extract",
    size = "medium",
    srcs = ["test_extract.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup",
    ],
)
