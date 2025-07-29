"""
Visualization module for PINN RANS Channel Flow results.
Contains plotting functions for field contours, profiles, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Optional, Tuple


class PINNPlotter:
    """Class for creating visualizations of PINN results."""
    
    def __init__(self, config, plotter_config=None):
        self.config = config
        self.plotter_config = plotter_config
        
        # Set plotting parameters
        plt.rcParams['figure.dpi'] = 300 if plotter_config else 150
        plt.rcParams['savefig.dpi'] = 300 if plotter_config else 150
        plt.rcParams['font.size'] = 10
    
    def create_prediction_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a grid for field predictions."""
        nx, ny = self.config.NX_PRED, self.config.NY_PRED
        x_coords = np.linspace(0, self.config.L, nx)
        y_coords = np.linspace(-self.config.CHANNEL_HALF_HEIGHT, self.config.CHANNEL_HALF_HEIGHT, ny)
        X, Y = np.meshgrid(x_coords, y_coords)
        pred_points = np.vstack((np.ravel(X), np.ravel(Y))).T
        return X, Y, pred_points
    
    def plot_field_contours(self, model, save_path: str):
        """Plot contour plots of all field variables."""
        X, Y, pred_points = self.create_prediction_grid()
        nx, ny = self.config.NX_PRED, self.config.NY_PRED
        
        # Get predictions
        predictions = model.predict(pred_points)
        
        # Extract and reshape variables
        u_pred = predictions[:, 0].reshape(ny, nx)
        v_pred = predictions[:, 1].reshape(ny, nx)
        p_prime_pred = predictions[:, 2].reshape(ny, nx)
        k_raw_pred = predictions[:, 3].reshape(ny, nx)
        eps_raw_pred = predictions[:, 4].reshape(ny, nx)
        
        # Transform k and epsilon back to physical values
        k_pred = np.exp(k_raw_pred) + self.config.EPS_SMALL
        eps_pred = np.exp(eps_raw_pred) + self.config.EPS_SMALL
        
        # Calculate eddy viscosity
        nu_t_pred = self.config.CMU * np.square(k_pred) / eps_pred
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Streamwise velocity
        cf1 = axes[0, 0].contourf(X, Y, u_pred, levels=50, cmap='viridis')
        plt.colorbar(cf1, ax=axes[0, 0], label='u (m/s)')
        axes[0, 0].set_title('Streamwise Velocity (u)')
        axes[0, 0].set_xlabel('x (m)')
        axes[0, 0].set_ylabel('y (m)')
        axes[0, 0].set_aspect('equal')
        
        # Transverse velocity
        cf2 = axes[0, 1].contourf(X, Y, v_pred, levels=50, cmap='viridis')
        plt.colorbar(cf2, ax=axes[0, 1], label='v (m/s)')
        axes[0, 1].set_title('Transverse Velocity (v)')
        axes[0, 1].set_xlabel('x (m)')
        axes[0, 1].set_ylabel('y (m)')
        axes[0, 1].set_aspect('equal')
        
        # Pressure
        cf3 = axes[0, 2].contourf(X, Y, p_prime_pred, levels=50, cmap='coolwarm')
        plt.colorbar(cf3, ax=axes[0, 2], label=\"p' (m²/s²)\")
        axes[0, 2].set_title('Kinematic Pressure')
        axes[0, 2].set_xlabel('x (m)')
        axes[0, 2].set_ylabel('y (m)')
        axes[0, 2].set_aspect('equal')
        
        # Turbulent kinetic energy (log scale)
        k_plot = np.log10(np.maximum(k_pred, self.config.EPS_SMALL))
        cf4 = axes[1, 0].contourf(X, Y, k_plot, levels=50, cmap='plasma')
        plt.colorbar(cf4, ax=axes[1, 0], label='log₁₀(k)')
        axes[1, 0].set_title('Turbulent Kinetic Energy (log scale)')
        axes[1, 0].set_xlabel('x (m)')
        axes[1, 0].set_ylabel('y (m)')
        axes[1, 0].set_aspect('equal')
        
        # Dissipation rate (log scale)
        eps_plot = np.log10(np.maximum(eps_pred, self.config.EPS_SMALL))
        cf5 = axes[1, 1].contourf(X, Y, eps_plot, levels=50, cmap='plasma')
        plt.colorbar(cf5, ax=axes[1, 1], label='log₁₀(ε)')
        axes[1, 1].set_title('Dissipation Rate (log scale)')
        axes[1, 1].set_xlabel('x (m)')
        axes[1, 1].set_ylabel('y (m)')
        axes[1, 1].set_aspect('equal')
        
        # Eddy viscosity ratio
        cf6 = axes[1, 2].contourf(X, Y, nu_t_pred / self.config.NU, levels=50, cmap='viridis')
        plt.colorbar(cf6, ax=axes[1, 2], label='νₜ / ν')
        axes[1, 2].set_title('Eddy Viscosity Ratio')
        axes[1, 2].set_xlabel('x (m)')
        axes[1, 2].set_ylabel('y (m)')
        axes[1, 2].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f\"Field contour plots saved to {save_path}\")\
    
    def plot_profile_comparison(self, model, openfoam_data: Optional[pd.DataFrame] = None, 
                              save_path: str = None):
        """Plot velocity and turbulence profiles at channel centerline."""
        # Get slice location
        x_slice = self.config.L / 2
        y_coords = np.linspace(-self.config.CHANNEL_HALF_HEIGHT, self.config.CHANNEL_HALF_HEIGHT, 
                              self.config.NY_PRED)
        slice_points = np.column_stack([np.full_like(y_coords, x_slice), y_coords])
        
        # Get PINN predictions
        pinn_pred = model.predict(slice_points)
        u_pinn = pinn_pred[:, 0]
        k_pinn = np.exp(pinn_pred[:, 3]) + self.config.EPS_SMALL
        eps_pinn = np.exp(pinn_pred[:, 4]) + self.config.EPS_SMALL
        nu_t_pinn = self.config.CMU * np.square(k_pinn) / eps_pinn
        
        # Normalize y coordinates
        y_norm = y_coords / self.config.CHANNEL_HALF_HEIGHT
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity profile
        axes[0, 0].plot(u_pinn, y_norm, 'r-', linewidth=2, label='PINN')
        if openfoam_data is not None and 'U' in openfoam_data.columns:
            axes[0, 0].plot(openfoam_data['U'], openfoam_data['y']/self.config.CHANNEL_HALF_HEIGHT, 
                           'b--', linewidth=1.5, label='OpenFOAM')
        axes[0, 0].set_xlabel('u (m/s)')
        axes[0, 0].set_ylabel('y/h')
        axes[0, 0].set_title(f'Velocity Profile at x={x_slice:.1f} m')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Turbulent kinetic energy
        axes[0, 1].plot(k_pinn, y_norm, 'r-', linewidth=2, label='PINN')
        if openfoam_data is not None and 'k' in openfoam_data.columns:
            axes[0, 1].plot(openfoam_data['k'], openfoam_data['y']/self.config.CHANNEL_HALF_HEIGHT,
                           'b--', linewidth=1.5, label='OpenFOAM')
        axes[0, 1].set_xlabel('k (m²/s²)')
        axes[0, 1].set_ylabel('y/h')
        axes[0, 1].set_title(f'TKE Profile at x={x_slice:.1f} m')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dissipation rate
        axes[1, 0].plot(eps_pinn, y_norm, 'r-', linewidth=2, label='PINN')
        if openfoam_data is not None and 'epsilon' in openfoam_data.columns:
            axes[1, 0].plot(openfoam_data['epsilon'], openfoam_data['y']/self.config.CHANNEL_HALF_HEIGHT,
                           'b--', linewidth=1.5, label='OpenFOAM')
        axes[1, 0].set_xlabel('ε (m²/s³)')
        axes[1, 0].set_ylabel('y/h')
        axes[1, 0].set_title(f'Dissipation Profile at x={x_slice:.1f} m')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        if np.all(eps_pinn > 0):
            axes[1, 0].set_xscale('log')
        
        # Eddy viscosity ratio
        axes[1, 1].plot(nu_t_pinn / self.config.NU, y_norm, 'r-', linewidth=2, label='PINN')
        if openfoam_data is not None and 'nut' in openfoam_data.columns:
            axes[1, 1].plot(openfoam_data['nut']/self.config.NU, 
                           openfoam_data['y']/self.config.CHANNEL_HALF_HEIGHT,
                           'b--', linewidth=1.5, label='OpenFOAM')
        axes[1, 1].set_xlabel('νₜ / ν')
        axes[1, 1].set_ylabel('y/h')
        axes[1, 1].set_title(f'Eddy Viscosity Ratio at x={x_slice:.1f} m')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f\"Profile comparison saved to {save_path}\")
        else:
            plt.show()
    
    def plot_wall_units(self, model, u_tau: float, save_path: str = None):
        """Plot profiles in wall units (y+, u+, k+, ε+)."""
        # Get upper wall data
        y_coords = np.linspace(0, self.config.CHANNEL_HALF_HEIGHT, self.config.NY_PRED // 2)
        x_slice = self.config.L / 2
        slice_points = np.column_stack([np.full_like(y_coords, x_slice), y_coords])
        
        # Get predictions
        pinn_pred = model.predict(slice_points)
        u_pinn = pinn_pred[:, 0]
        k_pinn = np.exp(pinn_pred[:, 3]) + self.config.EPS_SMALL
        eps_pinn = np.exp(pinn_pred[:, 4]) + self.config.EPS_SMALL
        
        # Calculate wall units
        y_dist = self.config.CHANNEL_HALF_HEIGHT - y_coords  # Distance from upper wall
        y_plus = y_dist * u_tau / self.config.NU
        u_plus = u_pinn / u_tau
        k_plus = k_pinn / (u_tau**2)
        eps_plus = eps_pinn * self.config.NU / (u_tau**4)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # u+ vs y+
        axes[0].semilogx(y_plus, u_plus, 'r-', linewidth=2, label='PINN')
        
        # Theoretical log-law
        y_plus_theory = np.logspace(1, np.log10(np.max(y_plus)), 100)
        u_plus_loglaw = (1 / self.config.KAPPA) * np.log(y_plus_theory) + 5.0  # Using B=5.0
        axes[0].semilogx(y_plus_theory, u_plus_loglaw, 'k:', label='Log-law')
        
        axes[0].set_xlabel('y⁺')
        axes[0].set_ylabel('u⁺')
        axes[0].set_title('Velocity in Wall Units')
        axes[0].legend()
        axes[0].grid(True, which='both', alpha=0.3)
        
        # k+ vs y+
        axes[1].semilogx(y_plus, k_plus, 'r-', linewidth=2, label='PINN')
        axes[1].axhline(1/np.sqrt(self.config.CMU), color='k', linestyle=':', 
                       label=f'k⁺ = 1/√Cμ ≈ {1/np.sqrt(self.config.CMU):.2f}')
        axes[1].set_xlabel('y⁺')
        axes[1].set_ylabel('k⁺')
        axes[1].set_title('TKE in Wall Units')
        axes[1].legend()
        axes[1].grid(True, which='both', alpha=0.3)
        
        # ε+ vs y+
        axes[2].semilogx(y_plus, eps_plus, 'r-', linewidth=2, label='PINN')
        eps_plus_theory = 1 / (self.config.KAPPA * y_plus_theory)
        axes[2].semilogx(y_plus_theory, eps_plus_theory, 'k:', label='ε⁺ = 1/(κy⁺)')
        axes[2].set_xlabel('y⁺')
        axes[2].set_ylabel('ε⁺')
        axes[2].set_title('Dissipation in Wall Units')
        axes[2].legend()
        axes[2].grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f\"Wall units plot saved to {save_path}\")
        else:
            plt.show()
    
    def plot_pressure_gradient(self, model, save_path: str = None):
        """Plot streamwise pressure gradient along centerline."""
        # Create centerline points
        x_coords = np.linspace(0, self.config.L, self.config.NX_PRED)
        centerline_points = np.column_stack([x_coords, np.zeros_like(x_coords)])
        
        # Get predictions
        pinn_pred = model.predict(centerline_points)
        p_prime = pinn_pred[:, 2]
        
        # Calculate gradient
        dp_dx = np.gradient(p_prime, x_coords)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, dp_dx, 'r-', linewidth=2, label='PINN')
        plt.xlabel('x (m)')
        plt.ylabel('dp\'/dx')
        plt.title('Streamwise Pressure Gradient along Centerline')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f\"Pressure gradient plot saved to {save_path}\")
        else:
            plt.show()
    
    def load_openfoam_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load OpenFOAM reference data for comparison."""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                # Assume it's a raw data file
                data = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=2)
                # Adjust column names based on your OpenFOAM output format
                if data.shape[1] >= 7:
                    data.columns = ['y', 'U', 'V', 'W', 'p', 'k', 'epsilon', 'nut'][:data.shape[1]]
            return data
        except Exception as e:
            print(f\"Error loading OpenFOAM data: {e}")
            return None
    
    def create_all_plots(self, model, output_dir: str, openfoam_data_path: str = None):
        """Create all standard plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load OpenFOAM data if available
        openfoam_data = None
        if openfoam_data_path and os.path.exists(openfoam_data_path):
            openfoam_data = self.load_openfoam_data(openfoam_data_path)
        
        # Field contours
        self.plot_field_contours(model, os.path.join(output_dir, "field_contours.png"))
        
        # Profile comparison
        self.plot_profile_comparison(model, openfoam_data, 
                                   os.path.join(output_dir, "profile_comparison.png"))
        
        # Wall units (estimate u_tau)
        u_tau_estimate = self.config.U_TAU_TARGET  # Use target value
        self.plot_wall_units(model, u_tau_estimate, 
                           os.path.join(output_dir, "wall_units.png"))
        
        # Pressure gradient
        self.plot_pressure_gradient(model, os.path.join(output_dir, "pressure_gradient.png"))
        
        print(f"All plots saved to {output_dir}")
