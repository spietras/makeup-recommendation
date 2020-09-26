load("@rules_python//python:defs.bzl", "py_binary")

alias(
    name = "jupyter",
    actual = "main",
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    args = ["--notebook-dir=\"$(execpath notebooks)\""],
    data = ["notebooks"],
    deps = [
        "//automakeup",
        "//imagine",
        "//third_party/faceparsing",
        "//third_party/mtcnn",
    ],
)
