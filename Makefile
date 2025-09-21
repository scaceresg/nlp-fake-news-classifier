venv:
	python -m venv venv
install-all:
	python -m pip install --upgrade pip
	pip install -e .[all]
install-dev:
	python -m pip install --upgrade pip
	pip install -e .[dev]
install-api:
	python -m pip install --upgrade pip
	pip install -e .[api]
install-front:
	python -m pip install --upgrade pip
	pip install -e .[front]
fmt:
	ruff format .