# load py_binary rule from rules_python
load("@rules_python//python:defs.bzl", "py_binary")

# rule building python binary, that can be run
py_binary(

    # name of the binary should be the name of the entry point script
    name = "main",

    # sources are recoursively globbed from all python files below
    # sources from subpackages are automatically excluded
    srcs = glob(["**/*.py"]),

    # python doesn't have its own bundling resources mechanism, so we use generic data and glob everything under resources
    data = glob(["resources/**/*"]),

    # python library is a dependency (notice the full path to package)
    # target name is implicitly assumed to be the same as the last part of the label (pythonlib in this case)
    deps = [
        "//automakeup"
    ]
)