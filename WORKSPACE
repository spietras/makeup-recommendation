##########################################  LOADING RULES  ###########################################

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

### RULES_PYTHON ###

RULES_PYTHON_NAME = "rules_python"
RULES_PYTHON_TAG = "0.1.0"
RULES_PYTHON_SHA = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0"
RULES_PYTHON_REPO = "bazelbuild"
RULES_PYTHON_ARCHIVE = "tar.gz"
RULES_PYTHON_URL = "https://github.com/{repo}/{name}/releases/download/{tag}/{name}-{tag}.{archive}".format(repo=RULES_PYTHON_REPO, name=RULES_PYTHON_NAME, tag=RULES_PYTHON_TAG, archive=RULES_PYTHON_ARCHIVE)

# use http_archive rule to load rules_python repo
http_archive(
    name = RULES_PYTHON_NAME,
    sha256 = RULES_PYTHON_SHA,
    url = RULES_PYTHON_URL
)

### RULES_CONDA ###

RULES_CONDA_NAME = "rules_conda"
RULES_CONDA_TAG = "0.0.4"
RULES_CONDA_SHA = "6c05d098ea82c172cd83d99c5fc892a488ffbf5f64ab3b2a32ab642c2a264e31"
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
    version = "4.9.2",
    quiet = False
)

# create environment
conda_create(
    name = "my_env",
    environment = "@//third_party/conda:environment.yml",  # label pointing to environment.yml file
    quiet = False,
    timeout = 7200
)

# register pythons from environment as toolchain
register_toolchain(
    py3_env = "my_env"
)
