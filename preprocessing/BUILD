load("@rules_python//python:defs.bzl", "py_binary")

alias(
    name = "preprocessing",
    actual = "main",
)

py_binary(
    name = "main",
    srcs = glob(["**/*.py"]),
    data = glob(["resources/**/*"]),
    deps = [
        "//automakeup",
        "//imagine",
        "//third_party/facenet",
        "//third_party/faceparsing",
        "//third_party/mtcnn",
    ],
)
