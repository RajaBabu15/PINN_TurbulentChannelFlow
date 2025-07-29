"""
PINN RANS Channel Flow Simulation Package

A Physics-Informed Neural Network implementation for solving 
Reynolds-Averaged Navier-Stokes equations with k-epsilon turbulence modeling
in channel flows.
"""

__version__ = "1.0.0"
__author__ = "PINN Research Team"
__email__ = "contact@example.com"

from .config.config import SimulationConfig, PlotterConfig
from .models.pinn_rans_model import PINNRANSModel
from .physics.equations import RANSEquations, BoundaryConditions
from .visualization.plotter import PINNPlotter

__all__ = [
    "SimulationConfig",
    "PlotterConfig", 
    "PINNRANSModel",
    "RANSEquations",
    "BoundaryConditions",
    "PINNPlotter",
]
