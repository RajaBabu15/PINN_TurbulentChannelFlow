# PINN Turbulent Channel Flow

A Physics-Informed Neural Network (PINN) implementation for solving Reynolds-Averaged Navier-Stokes (RANS) equations with k-epsilon turbulence modeling in channel flows.

## Overview

This project implements a PINN-based approach to simulate turbulent channel flow using the k-epsilon turbulence model. The implementation includes:

- RANS k-epsilon equations with wall functions
- Neural network-based PDE solver using DeepXDE
- Comparison with OpenFOAM CFD results
- Comprehensive visualization tools

## Features

- **Physics-Informed Neural Networks**: Solve RANS equations directly with DeepXDE
- **k-epsilon Turbulence Model**: Standard two-equation turbulence closure
- **Wall Functions**: Proper near-wall treatment for high Reynolds number flows
- **OpenFOAM Integration**: Complete CFD case with automated simulation scripts
- **Docker Support**: Multi-stage containerized environment for reproducible experiments
- **Build System**: CMake and Makefile for streamlined development workflow
- **Automated Workflows**: Scripts for mesh generation, simulation, and post-processing
- **Comprehensive Visualization**: Built-in plotting tools for all flow fields
- **Data Export**: CSV export functionality for result analysis and comparison

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- OpenFOAM (optional, for comparison data)

### Install from source

```bash
git clone https://github.com/example/pinn-turbulent-channel-flow.git
cd pinn-turbulent-channel-flow
pip install -r requirements.txt
pip install -e .
```

### Using Docker

```bash
# Build the Docker image
docker build -t pinn-channel-flow .

# Run with GPU support
docker run --gpus all -it pinn-channel-flow

# Or use docker-compose for full setup
docker-compose up pinn-channel-flow
```

### Using Makefile (Recommended)

```bash
# Setup environment and install dependencies
make setup

# Run OpenFOAM simulation
make openfoam-run

# Train PINN model
make train

# Generate plots
make plot

# Clean all outputs
make clean
```

## Quick Start

```python
from src import SimulationConfig, PINNRANSModel, PINNPlotter

# Setup configuration
config = SimulationConfig()
config.print_config()

# Create and train model
model = PINNRANSModel(config)
model.compile_model()
model.train_model()

# Generate plots
plotter = PINNPlotter(config)
plotter.create_all_plots(model.model, "output/plots")
```

## Project Structure

```
├── src/                    # Main source code
│   ├── config/            # Configuration classes
│   │   └── config.py      # SimulationConfig class
│   ├── models/            # PINN model implementations
│   │   └── pinn_rans_model.py  # PINNRANSModel class
│   ├── physics/           # Physics equations and BCs
│   │   └── equations.py   # RANSEquations and BoundaryConditions
│   ├── visualization/     # Plotting and analysis
│   │   └── plotter.py     # PINNPlotter class
│   └── utils/             # Utility functions
├── openfoam/              # OpenFOAM case files
│   └── channelKEpsilon/   # Channel flow case
│       ├── 0/             # Initial conditions
│       ├── constant/      # Mesh and properties
│       ├── system/        # Solver settings
│       ├── run_simulation.sh  # Automated simulation script
│       └── Allclean       # Case cleanup script
├── data/                  # Reference data and datasets
├── output/                # Results and plots
│   ├── plots/            # Generated visualizations
│   ├── models/           # Saved model checkpoints
│   └── results/          # Numerical results
├── logs/                  # Training and simulation logs
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
├── pyproject.toml        # Modern Python packaging
├── CMakeLists.txt        # Build system configuration
├── Makefile              # Development workflow automation
├── Dockerfile            # Multi-stage container build
├── docker-compose.yml    # Multi-service orchestration
├── LICENSE               # MIT license
└── README.md             # This file
```

## OpenFOAM Integration

The project includes a complete OpenFOAM case setup for turbulent channel flow validation:

### Manual Execution
```bash
cd openfoam/channelKEpsilon

# Clean previous results
./Allclean

# Run the automated simulation
./run_simulation.sh
```

### Using Makefile
```bash
# Generate mesh
make openfoam-mesh

# Run simulation
make openfoam-run

# Clean case
make openfoam-clean
```

### Using Docker
```bash
# Run OpenFOAM simulation in container
docker-compose up openfoam-runner
```

## Training

The model uses a two-stage training approach:
1. **Adam optimizer**: Initial training with adaptive learning rate
2. **L-BFGS optimizer**: Fine-tuning for convergence

Training can be monitored through:
- Loss history plots
- Checkpoint files
- Training logs

## Results

The PINN model provides:
- Velocity fields (u, v)
- Pressure field (p')
- Turbulence quantities (k, ε)
- Wall shear stress
- Comparison with OpenFOAM results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pinn_channel_flow,
  title={PINN Turbulent Channel Flow},
  author={PINN Research Team},
  year={2024},
  url={https://github.com/example/pinn-turbulent-channel-flow}
}
```

## Acknowledgments

- DeepXDE library for PINN implementation
- OpenFOAM community for CFD validation
- PyTorch team for deep learning framework

## Contact

For questions and support, please contact: contact@example.com
