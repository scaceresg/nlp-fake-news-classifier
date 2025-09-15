venv:
	python -m venv venv
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
fmt:
	ruff format .