# Testing in MLE world
This repository hosts the codebase used as a resource in discussion about tests in MLE domain - used
while streaming on [8thPassengerOfTheNostromo channel](https://www.twitch.tv/8thpassengerofthenostromo).

## What is the project about?
Inside repository, there is a piece of code with set of simple tools built upon Object Detection model. The
library is rather simple, designed not to be used in production, but rather to be an example on how things can
be tested.

## Repository structure
Inside repository, you may find few directories:
* `detector_kit` - main package with our library
* `tests` - package with tests
* `assets` - images and videos to be used as reference
* `notebooks` - Jupyter notebooks to visualise results

## :rotating_light: Repository setup
To initialize conda environment use
```bash
conda create -n TestingInMLEWorld python=3.9
conda activate TestingInMLEWorld
```

To install dependencies use
```bash
(TestingInMLEWorld) repository_root$ pip install -r requirements[-gpu].txt
(TestingInMLEWorld) repository_root$ pip install -r requirements-dev.txt
```

To enable Jupyter kernel
```bash
(TestingInMLEWorld) repository_root$ python -m ipykernel install \
  --user \
  --name TestingInMLEWorld \
  --display-name "Python3.9 (TestingInMLEWorld)"
```

To enable `pre-commit` use
```bash
(TestingInMLEWorld) repository_root$ pre-commit install
```

To run `pre-commit` check
```bash
(TestingInMLEWorld) repository_root$ pre-commit
```

To run tests, linter and type-checker
```bash
(TestingInMLEWorld) repository_root$ pytest
(TestingInMLEWorld) repository_root$ black .
(TestingInMLEWorld) repository_root$ mypy .
```
