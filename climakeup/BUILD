load("@rules_python//python:defs.bzl", "py_binary")

alias(
    name = "climakeup",
    actual = "main"
)

py_binary(
    name = "main",
    srcs = glob(["**/*.py"]),
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup"
    ]
)