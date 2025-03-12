# Use bash and make each target run in a single shell
.ONESHELL:
SHELL := /bin/bash

# Define PYTHONPATH for the project
PYTHONPATH := $(shell pwd)/src
# Export all variables to subshells (including PYTHONPATH)
.EXPORT_ALL_VARIABLES:

init:
	echo "initializing..."
	python -m pip install --upgrade pip
	pip install virtualenv 
	virtualenv venv
	source venv/bin/activate
	pip install -r requirements.txt

base-model-inference:
	echo "running base model inference..."
	source venv/bin/activate
	python src/base-model/inference.py