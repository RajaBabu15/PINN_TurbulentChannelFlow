# Multi-stage Dockerfile for PINN Turbulent Channel Flow

# Stage 1: OpenFOAM base image
FROM openfoam/openfoam9-paraview56:latest as openfoam-base

# Stage 2: Python base with CUDA support
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04 as python-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    cmake \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Stage 3: Final image combining OpenFOAM and Python
FROM python-base

# Copy OpenFOAM from the base image
COPY --from=openfoam-base /opt/openfoam9 /opt/openfoam9
COPY --from=openfoam-base /home/openfoam/.bashrc /home/openfoam/.bashrc

# Set environment variables for OpenFOAM
ENV FOAM_INST_DIR=/opt
ENV WM_PROJECT_VERSION=9
ENV WM_PROJECT=OpenFOAM
ENV WM_PROJECT_DIR=$FOAM_INST_DIR/$WM_PROJECT-$WM_PROJECT_VERSION

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/output /app/logs /app/data

# Set up environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for Jupyter if needed
EXPOSE 8888

# Default command
CMD ["python", "-c", "from src import SimulationConfig; config = SimulationConfig(); config.print_config()"]
