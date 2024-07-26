#!/usr/bin/env bash

#1. Installs or upgrades pip, the package installer for Python.
python -m pip install --upgrade pip
#2. Install virtualenv package using pip
python -m pip install virtualenv
#3. Creates a virtual environment named venv
python -m virtualenv venv
#4. Activate the environment for machine learning pipeline
. venv/bin/activate
#5. Install the required dependencies
pip install -r ./env/requirements.txt 