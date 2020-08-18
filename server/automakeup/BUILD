# load py_library rule from rules_python
load("@rules_python//python:defs.bzl", "py_library")

# load requirement function for stating pip dependencies
load("@pip//:requirements.bzl", "requirement")

# rule building python library, that can be used by other targets
py_library(

    # name of the library should be the name of the package
    name = "automakeup",

    # visibility is public, because the library is to be used externally
    visibility = ["//visibility:public"],

    # this one is kind of weird
    # py_libraries for some reason are actually accessible by path relative to the ROOT of the workspace
    # this means if we want to import like "from library import module" we need to first append the path to the library
    # notice that the package (where the __init__.py is) is in the subdirectory with the same as the library
    # that way we can just import as "from library import module"
    imports = ["."],

    # sources are recoursively globbed from all python files below
    # sources from subpackages are automatically excluded
    srcs = glob(["automakeup/**/*.py"]),

    # python doesn't have its own bundling resources mechanism, so we use generic data and glob everything under resources
    data = glob(["automakeup/resources/**/*"]),

    # depending on pip dependency, using requirement()
    deps = [
        requirement("dlib"),
        requirement("opencv-python"),
        requirement("openface")
    ]
)