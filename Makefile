# Makefile for AlphaScrabble

.PHONY: help install install-dev test clean build-lexicon setup-colab

help: ## Show this help message
	@echo "AlphaScrabble Makefile"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install AlphaScrabble in production mode
	pip install -e .

install-dev: ## Install AlphaScrabble in development mode with dev dependencies
	pip install -e ".[dev]"

test: ## Run the test suite
	pytest tests/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=alphascrabble --cov-report=html --cov-report=term

lint: ## Run linting tools
	black alphascrabble/ tests/
	isort alphascrabble/ tests/
	flake8 alphascrabble/ tests/

type-check: ## Run type checking
	mypy alphascrabble/

format: ## Format code
	black alphascrabble/ tests/
	isort alphascrabble/ tests/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build-lexicon: ## Build lexicon files from ENABLE1
	@echo "Building lexicon files..."
	@mkdir -p lexica_cache
	@if [ ! -f lexica_cache/enable1.txt ]; then \
		echo "Downloading ENABLE1 word list..."; \
		wget -q https://norvig.com/ngrams/enable1.txt -O lexica_cache/enable1.txt; \
	fi
	@if [ ! -d third_party/quackle ]; then \
		echo "Cloning Quackle..."; \
		git clone --depth 1 https://github.com/quackle/quackle.git third_party/quackle; \
	fi
	@if [ ! -f third_party/quackle/build/src/makedawg ]; then \
		echo "Building Quackle..."; \
		cd third_party/quackle && mkdir -p build && cd build && \
		cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
		make -j4; \
	fi
	@echo "Compiling DAWG..."
	@third_party/quackle/build/src/makedawg lexica_cache/enable1.txt english.quackle_alphabet > lexica_cache/english_enable1.dawg
	@echo "Compiling GADDAG..."
	@third_party/quackle/build/src/makegaddag lexica_cache/english_enable1.dawg > lexica_cache/english_enable1.gaddag
	@echo "Lexicon build complete!"

setup-colab: ## Setup for Google Colab (download dependencies)
	@echo "Setting up for Google Colab..."
	@apt-get update
	@apt-get install -y qtbase5-dev libqt5core5a build-essential cmake ninja-build
	@pip install -U pip wheel cmake ninja pybind11 pytest tensorboard pandas pyarrow rich tqdm click
	@pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	@echo "Colab setup complete!"

demo: ## Run a quick demo
	@echo "Running AlphaScrabble demo..."
	python -c "from alphascrabble import Board, Tile; board = Board(); print('Board created successfully!'); print('Board size:', len(board.grid), 'x', len(board.grid[0]))"

selfplay-demo: ## Run a small self-play demo
	@echo "Running self-play demo..."
	alphascrabble selfplay --games 5 --simulations 50 --out data/demo

train-demo: ## Run a small training demo
	@echo "Running training demo..."
	alphascrabble train --data data/demo --epochs 2 --batch-size 64

eval-demo: ## Run evaluation demo
	@echo "Running evaluation demo..."
	alphascrabble eval --net-a checkpoints/demo_model.pt --opponent random --games 10

play-demo: ## Run interactive play demo
	@echo "Running interactive play demo..."
	alphascrabble play --net checkpoints/demo_model.pt --human-first

full-demo: selfplay-demo train-demo eval-demo ## Run complete demo pipeline

check-deps: ## Check if all dependencies are available
	@echo "Checking dependencies..."
	@python -c "import torch; print('PyTorch:', torch.__version__)"
	@python -c "import numpy; print('NumPy:', numpy.__version__)"
	@python -c "import pandas; print('Pandas:', pandas.__version__)"
	@python -c "import click; print('Click:', click.__version__)"
	@python -c "import rich; print('Rich:', rich.__version__)"
	@echo "All dependencies available!"

check-gpu: ## Check GPU availability
	@echo "Checking GPU availability..."
	@python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

create-dirs: ## Create necessary directories
	@mkdir -p data/selfplay
	@mkdir -p data/training
	@mkdir -p checkpoints
	@mkdir -p logs
	@mkdir -p lexica_cache
	@echo "Directories created!"

init: create-dirs install-dev ## Initialize development environment

all: clean install-dev test ## Clean, install, and test

# Development shortcuts
dev: install-dev ## Alias for install-dev
t: test ## Alias for test
c: clean ## Alias for clean
l: lint ## Alias for lint
