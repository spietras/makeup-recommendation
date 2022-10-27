# makeup-recommendation

Makeup recommendation using machine learning and augmented reality 💄💅

Let the app find the most suitable makeup for you and apply it on your face in real-time.

## About

This repository provides the source code used in the paper
**Makeup Recommendation Using Machine Learning and Augmented Reality**
by [S. Pietras](https://github.com/spietras)
and [M. K. Kapuściński](https://github.com/Daraniel1000).

The paper was our Bachelor's thesis at the
[Faculty of Electronics and Information Technology](https://www.elka.pw.edu.pl/eng)
of the
[Warsaw University of Technology](https://www.pw.edu.pl/engpw).
It was completed in early 2021, but was written in Polish,
so we need to translate it to English first before we can publish it here.

## How it works

Now you don't have to worry about what lipstick to choose for a date ever again.
Trust in statistics and let the computer be your makeup artist, sweetie.

Run the Android app, point the camera at your face and click a button. 
Picture of your face will be uploaded to the server, 
where a trained neural network will generate a makeup that suits you best*.
After that, the results will be transferred back to the Android app
and the generated makeup will be applied to your face in real-time.

<sub><sup>* At least it will try, but beauty is subjective 😉</sup></sub>

## Requirements

Regular targets (e.g. ```webmakeup```, ```jupyter```) requirements:
- [Bazel](https://www.bazel.build/) dependencies
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/#system-requirements) dependencies
- [rules_python](https://github.com/bazelbuild/rules_python) dependencies
- (Optional) ```CUDA``` compatible GPU if you want to use GPU processing

```andromakeup``` runtime requirements:
- Android device with API level ```23``` at minimum

## Regular targets usage

Building only:

```sh
./bazelw build webmakeup
```

Running:

```sh
./bazelw run webmakeup -- --host 0.0.0.0
```

Testing:

```sh
./bazelw test webmakeup/...
```

If you are on Windows, use ```bazelw``` instead of ```./bazelw```.

## ```andromakeup``` usage

From ```andromakeup``` directory run:

```bash
./gradlew TASK
```

Server IP is kept as a string resource, so make sure to change that to your server IP.

See [here](https://developer.android.com/studio/build/building-cmdline) for more details or just use ```Android Studio```.

## Project structure

Main runnables:
- ```webmakeup``` - server for makeup recommendation
- ```andromakeup``` - main app for Android

Other runnables:
- ```climakeup``` - command line interface for makeup recommendation
- ```jupyter``` - jupyterlab with useful notebooks
- ```preprocessing``` - data preprocessing pipeline

Libraries:
- ```imagine``` - image processing library
- ```modelutils``` - machine learning model utilities
- ```ganette``` - main model used for recommendation
- ```automakeup``` - makeup recommendation library

Other:
- ```third_party``` - third-party code and environment definition
- ```config``` - project configuration
- ```tools``` - project tools
