"""
Configuration module for PINN RANS Channel Flow simulation.
Contains all simulation parameters and settings.
"""

import os
import numpy as np
import deepxde as dde


class SimulationConfig:
    """Main configuration class for PINN simulation parameters."""
    
    def __init__(self, base_dir=".", use_cuda=True):
        self.base_dir = base_dir
        self.use_cuda = use_cuda
        
        # Directory structure
        self.setup_directories()
        
        # Physical parameters
        self.setup_physical_parameters()
        
        # Turbulence model constants
        self.setup_turbulence_constants()
        
        # Wall function parameters
        self.setup_wall_function_parameters()
        
        # Numerical parameters
        self.setup_numerical_parameters()
        
        # Network architecture
        self.setup_network_parameters()
        
        # Training parameters
        self.setup_training_parameters()
        
        # Plotting parameters
        self.setup_plotting_parameters()
        
        # Create geometry
        self.setup_geometry()
    
    def setup_directories(self):
        """Setup directory paths."""
        self.OUTPUT_DIR = os.path.join(self.base_dir, "output")
        self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, "model_checkpoints")
        self.LOG_DIR = os.path.join(self.base_dir, "logs")
        self.PLOT_DIR = os.path.join(self.OUTPUT_DIR, "plots")
        self.DATA_DIR = os.path.join(self.base_dir, "data")
        self.OPENFOAM_DIR = os.path.join(self.base_dir, "openfoam", "channelKEpsilon")
        
        # Log files
        self.LOG_FILE = os.path.join(self.LOG_DIR, "training_log.log")
        self.REFERENCE_DATA_FILE = os.path.join(self.DATA_DIR, "reference_output_data.csv")
        
        # Checkpoint settings
        self.CHECKPOINT_FILENAME_BASE = "rans_channel_wf"
    
    def setup_physical_parameters(self):
        """Setup fluid properties and flow parameters."""
        self.NU = 0.0002  # Kinematic viscosity (m^2/s)
        self.RHO = 1.0    # Density (kg/m^3)
        self.MU = self.RHO * self.NU  # Dynamic viscosity
        self.U_INLET = 1.0  # Inlet velocity (m/s)
        self.H = 2.0        # Channel height (m)
        self.CHANNEL_HALF_HEIGHT = self.H / 2.0  # Channel half-height
        self.L = 10.0       # Channel length (m)
        self.RE_H = self.U_INLET * self.H / self.NU  # Reynolds number
        self.EPS_SMALL = 1e-10  # Small number to prevent numerical issues
    
    def setup_turbulence_constants(self):
        """Setup k-epsilon turbulence model constants."""
        self.CMU = 0.09
        self.CEPS1 = 1.44
        self.CEPS2 = 1.92
        self.SIGMA_K = 1.0
        self.SIGMA_EPS = 1.3
    
    def setup_wall_function_parameters(self):
        """Setup wall function parameters."""
        self.KAPPA = 0.41      # von Karman constant
        self.E_WALL = 9.8      # Log-law constant for smooth walls
        self.Y_P = 0.04        # Distance from wall to apply wall function
        self.RE_TAU_TARGET = 350  # Target friction Reynolds number
        self.U_TAU_TARGET = self.RE_TAU_TARGET * self.NU / self.CHANNEL_HALF_HEIGHT
        self.YP_PLUS_TARGET = self.Y_P * self.U_TAU_TARGET / self.NU
        
        # Calculate target values for wall functions
        self.U_TARGET_WF = (self.U_TAU_TARGET / self.KAPPA) * np.log(
            max(self.E_WALL * self.YP_PLUS_TARGET, self.EPS_SMALL)
        )
        self.K_TARGET_WF = self.U_TAU_TARGET**2 / np.sqrt(self.CMU)
        self.EPS_TARGET_WF = self.U_TAU_TARGET**3 / max(self.KAPPA * self.Y_P, self.EPS_SMALL)
    
    def setup_numerical_parameters(self):
        """Setup inlet turbulence conditions."""
        self.TURBULENCE_INTENSITY = 0.05
        self.MIXING_LENGTH_SCALE = 0.07 * self.CHANNEL_HALF_HEIGHT
        self.K_INLET = 1.5 * (self.U_INLET * self.TURBULENCE_INTENSITY)**2
        self.EPS_INLET = (self.CMU**0.75) * (self.K_INLET**1.5) / self.MIXING_LENGTH_SCALE
        
        # Transformed values for neural network
        self.K_INLET_TRANSFORMED = np.log(max(self.K_INLET, self.EPS_SMALL))
        self.EPS_INLET_TRANSFORMED = np.log(max(self.EPS_INLET, self.EPS_SMALL))
        self.K_TARGET_WF_TRANSFORMED = np.log(max(self.K_TARGET_WF, self.EPS_SMALL))
        self.EPS_TARGET_WF_TRANSFORMED = np.log(max(self.EPS_TARGET_WF, self.EPS_SMALL))
    
    def setup_network_parameters(self):
        """Setup neural network architecture parameters."""
        self.NUM_LAYERS = 8
        self.NUM_NEURONS = 64
        self.ACTIVATION = "tanh"
        self.INITIALIZER = "Glorot normal"
        self.NETWORK_INPUTS = 2   # x, y coordinates
        self.NETWORK_OUTPUTS = 5  # u, v, p', k_raw, eps_raw
    
    def setup_training_parameters(self):
        """Setup training parameters."""
        self.NUM_DOMAIN_POINTS = 20000
        self.NUM_BOUNDARY_POINTS = 4000
        self.NUM_TEST_POINTS = 5000
        self.NUM_WF_POINTS_PER_WALL = 200
        
        self.LEARNING_RATE_ADAM = 1e-3
        self.ADAM_ITERATIONS = 50000
        self.LBFGS_ITERATIONS = 20000
        
        # Loss weights
        self.PDE_WEIGHTS = [1, 1, 1, 1, 1]  # continuity, mom_x, mom_y, k, eps
        self.BC_WEIGHTS = [10, 10, 10, 10, 10, 10, 10, 20, 20, 20]  # boundary conditions
        self.LOSS_WEIGHTS = self.PDE_WEIGHTS + self.BC_WEIGHTS
        
        self.SAVE_INTERVAL = 1000
        self.DISPLAY_EVERY = 1000
    
    def setup_plotting_parameters(self):
        """Setup plotting parameters."""
        self.NX_PRED = 200  # Grid points in x direction for plotting
        self.NY_PRED = 100  # Grid points in y direction for plotting
        self.CMAP_VELOCITY = 'viridis'
        self.CMAP_PRESSURE = 'coolwarm'
        self.CMAP_TURBULENCE = 'plasma'
    
    def setup_geometry(self):
        """Setup computational geometry."""
        self.GEOM = dde.geometry.Rectangle(
            xmin=[0, -self.CHANNEL_HALF_HEIGHT], 
            xmax=[self.L, self.CHANNEL_HALF_HEIGHT]
        )
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.OUTPUT_DIR, self.MODEL_DIR, self.LOG_DIR, 
            self.PLOT_DIR, self.DATA_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def print_config(self):
        """Print configuration summary."""
        print("=" * 60)
        print("PINN RANS Channel Flow Configuration")
        print("=" * 60)
        print(f"Re_H: {self.RE_H:.0f}")
        print(f"Channel dimensions: L={self.L}, H={self.H}")
        print(f"Wall function y_p: {self.Y_P} (Target y+: {self.YP_PLUS_TARGET:.2f})")
        print(f"Network: {self.NUM_LAYERS} layers, {self.NUM_NEURONS} neurons")
        print(f"Training: Adam {self.ADAM_ITERATIONS}, L-BFGS {self.LBFGS_ITERATIONS}")
        print(f"Domain points: {self.NUM_DOMAIN_POINTS}")
        print(f"Boundary points: {self.NUM_BOUNDARY_POINTS}")
        print("=" * 60)


class PlotterConfig:
    """Configuration class specifically for plotting parameters."""
    
    def __init__(self):
        self.NX_PRED = 200
        self.NY_PRED = 100
        self.CMAP_VELOCITY = 'viridis'
        self.CMAP_PRESSURE = 'coolwarm'
        self.CMAP_TURBULENCE = 'plasma'
        self.FIGURE_SIZE = (15, 10)
        self.DPI = 300
        self.SAVE_FORMAT = 'png'
