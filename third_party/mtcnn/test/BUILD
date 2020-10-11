load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_mtcnn",
    size = "medium",
    srcs = ["test_mtcnn.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
        "//third_party/mtcnn",
    ],
)
