#!/usr/bin/env bash

# exit when any command fails
set -e

flake8 afed/
isort afed/afed.py
sphinx-build docs/ docs/_build
pytest -v --cov=afed --doctest-modules afed/tests/
