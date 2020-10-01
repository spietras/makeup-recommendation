load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_facenet",
    size = "medium",
    srcs = ["test_facenet.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//third_party/facenet",
    ],
)
