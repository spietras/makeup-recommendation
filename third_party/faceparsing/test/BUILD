load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_parser",
    size = "medium",
    srcs = ["test_parser.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//third_party/faceparsing",
    ],
)
