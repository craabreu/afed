#!/usr/bin/env bash
flake8 afed/
isort afed/afed.py
sphinx-build docs/ docs/_build
pytest -v --cov=afed --doctest-modules afed/tests/
