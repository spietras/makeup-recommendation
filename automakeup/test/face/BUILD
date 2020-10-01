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
    name = "test_face_extraction",
    size = "medium",
    srcs = ["test_face_extraction.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup",
    ],
)
