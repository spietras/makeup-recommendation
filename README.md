# makeup-recommendation
Makeup recommendation system 💄💅

## Requirements for everything except Android

- Bazel requirements (to run Bazel itself):

	Linux:

	- ```glibc``` (it should be available in most Linux distributions, but for smaller ones like ```alpine``` this can be a problem)

	Windows:

	- TOCHECK

- Tooling requirements (to configure all build tools):

	Linux:

	- any C compiler (like ```gcc```)

	- any ```python``` (note: ```python``` command must be on ```PATH```, if you don't have ```python``` but have ```python2``` or ```python3``` you should create a symlink)

	Windows:

	- TOCHECK

- Dependencies requirements (to setup all third party dependencies):

	- If you want to use ```dlib``` in GPU mode, you need ```CUDA``` compatible GPU and ```CUDA Toolkit``` installed. Otherwise ```dlib``` will run in CPU mode.

## Usage

Building only:

```sh
./bazelw build TARGET
```

Running:

```sh
./bazelw run TARGET [-- args...]
```

Runnable targets: ```webmakeup```, ```climakeup```

Testing:

```sh
./bazelw test TARGET/...
```

See the chosen target's README for further details.

If you are on Windows, use ```bazelw``` instead of ```./bazelw```.
