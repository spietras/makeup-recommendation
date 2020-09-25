load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_functional",
    size = "small",
    srcs = ["test_functional.py"],
    data = glob(["resources/**/*"]),
    deps = [
        "//imagine",
    ],
)
