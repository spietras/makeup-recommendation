# makeup-recommendation
Makeup recommendation system 💄💅

## Requirements for everything except Android

- Bazel requirements (to run Bazel itself):

	Linux:

	- ```glibc``` (it should be available in most Linux distributions, but for smaller ones like ```alpine``` this can be a problem)

	Windows:

	- Nothing

- Tooling requirements (to configure all build tools):

	Linux:

	- development libraries for: ```readline```, ```zlib```, ```bzip2``` , ```sqlite3```, ```openssl```, ```libffi```

	- ```curl```

	- ```make```

	- any C and C++ compiler (like ```gcc``` and ```g++```)

	- any ```python``` (note: ```python``` command must be on ```PATH```, if you don't have ```python``` but have ```python2``` or ```python3``` you should create a symlink)

	Windows:

	- Windows 10

	- Developer Mode enabled

- Dependencies requirements (to setup all third party dependencies):

	Linux:

	- ```cmake```

	Windows:

	- ```cmake```

## Usage

Building only:

```
./bazelw build TARGET
```

Running:

```
./bazelw run TARGET [-- args...]
```

Runnable targets: ```webmakeup```, ```climakeup```

Testing:

```
./bazelw test TARGET/...
```

See the chosen target's README for further details.

If you are on Windows, use ```bazelw``` instead of ```./bazelw```.
