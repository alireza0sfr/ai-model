init:
	echo "initializing..."
	python -m pip install --upgrade pip
	pip install virtualenv 
	virtualenv venv
	source venv/bin/activate
	pip install -r requirements.txt

demo:
	echo "running demo..."
	python src/demo/main.py