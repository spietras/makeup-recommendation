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

### RULES_CONDA ###

RULES_CONDA_NAME = "rules_conda"
RULES_CONDA_TAG = "0.0.1"
RULES_CONDA_SHA = "945d040a3bcc91f9fea3069b4ab16a03ed0b699dcf00a7a97fcb8674ca780677"
RULES_CONDA_REPO = "spietras"
RULES_CONDA_ARCHIVE = "zip"
RULES_CONDA_URL = "https://github.com/{repo}/{name}/releases/download/{tag}/{name}-{tag}.{archive}".format(repo=RULES_CONDA_REPO, name=RULES_CONDA_NAME, tag=RULES_CONDA_TAG, archive=RULES_CONDA_ARCHIVE)

# use http_archive rule to load rules_conda repo
http_archive(
    name = RULES_CONDA_NAME,
    sha256 = RULES_CONDA_SHA,
    url = RULES_CONDA_URL
)

########################################## USING REPO RULES ##########################################

### CONDA ###

load("@rules_conda//:defs.bzl", "load_conda", "conda_create", "register_toolchain")

# download and install conda
load_conda(
    version="4.8.4" # optional, defaults to 4.8.4
)

# create environment
conda_create(
    name = "my_env",
    environment = "@//third_party/python:environment.yml" # label pointing to environment.yml file
)

# register pythons from environment as toolchain
register_toolchain(
    py3_env = "my_env"
)