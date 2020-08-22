load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

alias(
    name = "webmakeup",
    actual = "main"
)

py_binary(
    name = "main",
    srcs = glob(["**/*.py"]),
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup",
        requirement("flask")
    ]
)