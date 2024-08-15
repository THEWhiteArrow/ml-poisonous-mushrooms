install:
	@echo "Setting poetry config..."
	poetry config virtualenvs.in-project true
	@echo "Installing poetry..."
	poetry install
	@echo "Done!"


hyper_opt:
	@echo "Running hyperparameter optimization..."
	poetry run poe hyper_opt
	@echo "Done!"