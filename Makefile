# Makefile for PINN Turbulent Channel Flow

# Variables
PYTHON := python3
PIP := pip3
PROJECT_DIR := $(shell pwd)
SRC_DIR := $(PROJECT_DIR)/src
OPENFOAM_DIR := $(PROJECT_DIR)/openfoam/channelKEpsilon
OUTPUT_DIR := $(PROJECT_DIR)/output
LOGS_DIR := $(PROJECT_DIR)/logs
DATA_DIR := $(PROJECT_DIR)/data

# Default target
.PHONY: help
help:
	@echo "PINN Turbulent Channel Flow - Available targets:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install         - Install Python dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo "  setup           - Create necessary directories"
	@echo ""
	@echo "OpenFOAM:"
	@echo "  openfoam-mesh   - Generate mesh for OpenFOAM case"
	@echo "  openfoam-run    - Run OpenFOAM simulation"
	@echo "  openfoam-clean  - Clean OpenFOAM case"
	@echo ""
	@echo "PINN Training:"
	@echo "  train           - Train PINN model"
	@echo "  train-resume    - Resume training from checkpoint"
	@echo "  plot            - Generate visualization plots"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-run      - Run container interactively"
	@echo "  docker-train    - Run training in container"
	@echo "  docker-openfoam - Run OpenFOAM in container"
	@echo ""
	@echo "Development:"
	@echo "  test            - Run tests"
	@echo "  lint            - Run code linting"
	@echo "  format          - Format code with black"
	@echo "  clean           - Clean output files"
	@echo "  clean-all       - Clean everything including caches"

# Setup and Installation
.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

.PHONY: setup
setup:
	mkdir -p $(OUTPUT_DIR) $(LOGS_DIR) $(DATA_DIR)
	mkdir -p $(OUTPUT_DIR)/plots $(OUTPUT_DIR)/model_checkpoints

# OpenFOAM targets
.PHONY: openfoam-mesh
openfoam-mesh:
	@echo "Generating OpenFOAM mesh..."
	cd $(OPENFOAM_DIR) && \
	source $$FOAM_INST_DIR/OpenFOAM-9/etc/bashrc && \
	blockMesh

.PHONY: openfoam-run
openfoam-run: openfoam-mesh
	@echo "Running OpenFOAM simulation..."
	cd $(OPENFOAM_DIR) && \
	source $$FOAM_INST_DIR/OpenFOAM-9/etc/bashrc && \
	simpleFoam

.PHONY: openfoam-clean
openfoam-clean:
	@echo "Cleaning OpenFOAM case..."
	cd $(OPENFOAM_DIR) && \
	./Allclean || (rm -rf [0-9]* processor* postProcessing constant/polyMesh log.* *.log)

# PINN Training targets
.PHONY: train
train: setup
	@echo "Training PINN model..."
	cd $(PROJECT_DIR) && \
	$(PYTHON) -c "from src import SimulationConfig, PINNRANSModel; \
	config = SimulationConfig(base_dir='$(PROJECT_DIR)'); \
	config.create_directories(); \
	model = PINNRANSModel(config); \
	model.compile_model(); \
	model.train_model()"

.PHONY: train-resume
train-resume:
	@echo "Resuming PINN training from checkpoint..."
	cd $(PROJECT_DIR) && \
	$(PYTHON) -c "from src import SimulationConfig, PINNRANSModel; \
	config = SimulationConfig(base_dir='$(PROJECT_DIR)'); \
	model = PINNRANSModel(config); \
	model.compile_model(); \
	model.train_model()"

.PHONY: plot
plot:
	@echo "Generating plots..."
	cd $(PROJECT_DIR) && \
	$(PYTHON) -c "from src import SimulationConfig, PINNRANSModel, PINNPlotter; \
	import os; \
	config = SimulationConfig(base_dir='$(PROJECT_DIR)'); \
	try: \
		model = PINNRANSModel(config); \
		plotter = PINNPlotter(config); \
		plotter.create_all_plots(model.model, '$(OUTPUT_DIR)/plots'); \
	except Exception as e: \
		print(f'Error: {e}'); \
		print('Make sure model is trained first: make train')"

# Docker targets
.PHONY: docker-build
docker-build:
	docker build -t pinn-channel-flow .

.PHONY: docker-run
docker-run:
	docker run --gpus all -it --rm \
		-v $(PROJECT_DIR)/data:/app/data \
		-v $(PROJECT_DIR)/output:/app/output \
		-v $(PROJECT_DIR)/logs:/app/logs \
		-p 8888:8888 \
		pinn-channel-flow bash

.PHONY: docker-train
docker-train:
	docker-compose --profile training up pinn-training

.PHONY: docker-openfoam
docker-openfoam:
	docker-compose --profile openfoam up openfoam

# Development targets
.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:
	flake8 src/ tests/
	pylint src/

.PHONY: format
format:
	black src/ tests/
	isort src/ tests/

# Cleaning targets
.PHONY: clean
clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(LOGS_DIR)/*
	mkdir -p $(OUTPUT_DIR)/plots $(OUTPUT_DIR)/model_checkpoints

.PHONY: clean-all
clean-all: clean openfoam-clean
	rm -rf __pycache__ .pytest_cache .coverage
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	docker system prune -f

# Jupyter notebook
.PHONY: notebook
notebook:
	cd $(PROJECT_DIR) && \
	$(PYTHON) -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Package building
.PHONY: build
build:
	$(PYTHON) -m build

.PHONY: upload
upload: build
	$(PYTHON) -m twine upload dist/*

# Environment setup
.PHONY: env
env:
	@echo "Setting up environment variables..."
	@echo "export PYTHONPATH=$(PROJECT_DIR):$$PYTHONPATH"
	@echo "export PINN_PROJECT_DIR=$(PROJECT_DIR)"
