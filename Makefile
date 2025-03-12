# Use bash and make each target run in a single shell
.ONESHELL:
SHELL := /bin/bash

# Define project root and PYTHONPATH
PYTHONPATH := $(shell pwd)/src

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
	PYTHONPATH=$(PYTHONPATH) python src/base-model/inference.py