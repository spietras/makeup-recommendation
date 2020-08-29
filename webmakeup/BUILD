load("@rules_python//python:defs.bzl", "py_binary")

alias(
    name = "webmakeup",
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