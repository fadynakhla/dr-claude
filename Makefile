all: help

######################
# HELP
######################

app:
	poetry run uvicorn dr_claude.application:app

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
	@echo 'app                    - start backend app'