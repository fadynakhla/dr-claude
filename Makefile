all: help

######################
# MAIN
######################

app:
	poetry run uvicorn dr_claude.application:app

test:
	poetry run pytest

######################
# FORMATTING
######################

PYTHON_FILES=.
format:
	poetry run black $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'test                         - run unit tests'
	@echo 'app                          - start backend app'
