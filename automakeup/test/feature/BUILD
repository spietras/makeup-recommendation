load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_extract",
    size = "medium",
    srcs = ["test_extract.py"],
    data = glob(["resources/**/*"]),
    tags = ["exclusive"],
    deps = [
        "//automakeup",
        "//third_party/facenet",
        "//third_party/faceparsing",
    ],
)

py_test(
    name = "test_face",
    size = "medium",
    srcs = ["test_face.py"],
    data = glob(["resources/**/*"]),
    tags = ["exclusive"],
    deps = [
        "//automakeup",
        "//third_party/faceparsing",
    ],
)

py_test(
    name = "test_makeup",
    size = "medium",
    srcs = ["test_makeup.py"],
    data = glob(["resources/**/*"]),
    tags = ["exclusive"],
    deps = [
        "//automakeup",
        "//third_party/faceparsing",
    ],
)
