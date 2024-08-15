install:
	@echo "Setting poetry config..."
	poetry config virtualenvs.in-project true
	@echo "Installing poetry..."
	poetry install
	@echo "Done!"