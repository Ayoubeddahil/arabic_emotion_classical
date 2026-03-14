.PHONY: install run train test clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make run        - Run FastAPI server"
	@echo "  make train      - Run training notebook"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean cache files"
	@echo "  make notebook   - Start Jupyter notebook"

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

train:
	jupyter nbconvert --to notebook --execute notebooks/2_train_model.ipynb --output notebooks/2_train_model_output.ipynb

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

notebook:
	jupyter notebook notebooks/