# makeup-recommendation

Makeup recommendation system 💄💅

Let the app find the most suitable makeup for you and apply it on your face in real-time.

## How it works

Now you don't have to worry about what lipstick to choose for a date ever again.
Trust in statistics and let the computer be your makeup artist.

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
./bazelw build TARGET
```

Running:

```sh
./bazelw run TARGET [-- args...]
```

Testing:

```sh
./bazelw test TARGET/...
```

If you are on Windows, use ```bazelw``` instead of ```./bazelw```.

## ```andromakeup``` usage

From ```andromakeup``` directory run:

```bash
./gradlew TASK
```

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
