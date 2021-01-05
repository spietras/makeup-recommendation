load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_ganette",
    size = "medium",
    srcs = ["test_ganette.py"],
    data = glob(["resources/**/*"]),
    local = True,
    deps = [
        "//ganette",
    ],
)
