load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_feature_extraction",
    size = "medium",
    srcs = ["test_feature_extraction.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup",
        "//third_party/facenet",
        "//third_party/faceparsing",
    ],
)

py_test(
    name = "test_iris",
    size = "medium",
    srcs = ["test_iris.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup",
        "//third_party/faceparsing",
    ],
)
