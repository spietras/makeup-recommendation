##########################################  LOADING RULES  ###########################################

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

### RULES_PYTHON ###

RULES_PYTHON_NAME = "rules_python"
RULES_PYTHON_TAG = "0.0.2"
RULES_PYTHON_PREFIX = "%s-%s" % (RULES_PYTHON_NAME, RULES_PYTHON_TAG)
RULES_PYTHON_SHA = "a0480851566fc9c20a532d0dd6e21f03c95db5d1a167304d728aa52ebc820d26"
RULES_PYTHON_REPO = "bazelbuild"
RULES_PYTHON_ARCHIVE = "zip"
RULES_PYTHON_URL = "https://github.com/%s/%s/archive/%s.%s" % (RULES_PYTHON_REPO, RULES_PYTHON_NAME, RULES_PYTHON_TAG, RULES_PYTHON_ARCHIVE)

http_archive(
    name = RULES_PYTHON_NAME,
    strip_prefix = RULES_PYTHON_PREFIX,
    sha256 = RULES_PYTHON_SHA,
    url = RULES_PYTHON_URL
)

### RULES_PYTHON_EXTERNAL ###

RULES_PYTHON_EXTERNAL_NAME = "rules_python_external"
RULES_PYTHON_EXTERNAL_TAG = "8029ddb56227d97cd052ff034929b7790a63a133"
RULES_PYTHON_EXTERNAL_PREFIX = "%s-%s" % (RULES_PYTHON_EXTERNAL_NAME, RULES_PYTHON_EXTERNAL_TAG)
RULES_PYTHON_EXTERNAL_SHA = "c7d43551d44c7ca8bb1360c1076be228a0561c8abbdea7eb73e279b83773f51c"
RULES_PYTHON_EXTERNAL_REPO = "dillon-giacoppo"
RULES_PYTHON_EXTERNAL_ARCHIVE = "zip"
RULES_PYTHON_EXTERNAL_URL = "https://github.com/%s/%s/archive/%s.%s" % (RULES_PYTHON_EXTERNAL_REPO, RULES_PYTHON_EXTERNAL_NAME, RULES_PYTHON_EXTERNAL_TAG, RULES_PYTHON_EXTERNAL_ARCHIVE)

http_archive(
    name = RULES_PYTHON_EXTERNAL_NAME,
    strip_prefix = RULES_PYTHON_EXTERNAL_PREFIX,
    sha256 = RULES_PYTHON_EXTERNAL_SHA,
    url = RULES_PYTHON_EXTERNAL_URL
)

load("@rules_python_external//:repositories.bzl", "rules_python_external_dependencies")
rules_python_external_dependencies()

### RULES_PYENV ###

load("@//third_party/python:pyenv.bzl", "pyenv_install")

pyenv_install(
    py2 = "2.7.17",
    py3 = "3.7.5",
)

########################################## USING REPO RULES ##########################################

### PIP ###

load("@rules_python_external//:defs.bzl", "pip_install")

pip_install(
    name = "pip",
    requirements = "//third_party/python:requirements.txt",
    python_interpreter_target = "@pyenv//:py3/python",
)