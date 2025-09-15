venv:
	python -m venv venv
install:
	pip install -r requirements.txt
fmt:
	ruff format .