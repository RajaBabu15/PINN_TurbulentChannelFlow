"""
Physics equations module for RANS k-epsilon turbulence modeling.
Contains the PDE definitions and boundary conditions.
"""

import deepxde as dde
import torch
import numpy as np


class RANSEquations:
    """Class containing RANS k-epsilon equations and boundary conditions."""
    
    def __init__(self, config):
        self.config = config
    
    def pde_system(self, x, y):
        """
        Define the RANS k-epsilon PDE system.
        
        Args:
            x: spatial coordinates, shape = (N, 2), where x[:, 0] = x and x[:, 1] = y
            y: network output, shape = (N, 5), where y = (u, v, p', k_raw, eps_raw)
            
        Returns:
            List of PDE residuals [continuity, momentum_x, momentum_y, k_equation, eps_equation]
        """
        # Extract variables
        u, v, p_prime, k_raw, eps_raw = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]
        
        # Transform to positive k and epsilon using exponential
        k = dde.backend.exp(k_raw) + self.config.EPS_SMALL
        eps = dde.backend.exp(eps_raw) + self.config.EPS_SMALL
        
        # Calculate derivatives
        derivatives = self._calculate_derivatives(x, y, k, eps)
        
        # Calculate turbulent viscosity
        nu_t = self.config.CMU * dde.backend.square(k) / eps
        nu_eff = self.config.NU + nu_t
        
        # Define equations
        eq_continuity = self._continuity_equation(derivatives)
        eq_momentum_x = self._momentum_x_equation(derivatives, nu_eff, p_prime, k)
        eq_momentum_y = self._momentum_y_equation(derivatives, nu_eff, p_prime, k)
        eq_k = self._k_equation(derivatives, nu_t, k, eps)
        eq_eps = self._epsilon_equation(derivatives, nu_t, k, eps)
        
        return [eq_continuity, eq_momentum_x, eq_momentum_y, eq_k, eq_eps]
    
    def _calculate_derivatives(self, x, y, k, eps):
        """Calculate all required derivatives."""
        derivatives = {}
        
        # Velocity derivatives
        derivatives['u_x'] = dde.grad.jacobian(y, x, i=0, j=0)
        derivatives['u_y'] = dde.grad.jacobian(y, x, i=0, j=1)
        derivatives['v_x'] = dde.grad.jacobian(y, x, i=1, j=0)
        derivatives['v_y'] = dde.grad.jacobian(y, x, i=1, j=1)
        
        # Pressure derivatives
        derivatives['p_x'] = dde.grad.jacobian(y, x, i=2, j=0)
        derivatives['p_y'] = dde.grad.jacobian(y, x, i=2, j=1)
        
        # Velocity second derivatives
        derivatives['u_xx'] = dde.grad.hessian(y, x, component=0, i=0, j=0)
        derivatives['u_yy'] = dde.grad.hessian(y, x, component=0, i=1, j=1)
        derivatives['u_xy'] = dde.grad.hessian(y, x, component=0, i=0, j=1)
        derivatives['v_xx'] = dde.grad.hessian(y, x, component=1, i=0, j=0)
        derivatives['v_yy'] = dde.grad.hessian(y, x, component=1, i=1, j=1)
        derivatives['v_xy'] = dde.grad.hessian(y, x, component=1, i=0, j=1)
        
        # k and epsilon derivatives
        derivatives['k_x'] = dde.grad.jacobian(k, x, i=0, j=0)
        derivatives['k_y'] = dde.grad.jacobian(k, x, i=0, j=1)
        derivatives['eps_x'] = dde.grad.jacobian(eps, x, i=0, j=0)
        derivatives['eps_y'] = dde.grad.jacobian(eps, x, i=0, j=1)
        
        # k and epsilon second derivatives
        derivatives['k_xx'] = dde.grad.hessian(k, x, i=0, j=0)
        derivatives['k_yy'] = dde.grad.hessian(k, x, i=1, j=1)
        derivatives['eps_xx'] = dde.grad.hessian(eps, x, i=0, j=0)
        derivatives['eps_yy'] = dde.grad.hessian(eps, x, i=1, j=1)
        
        return derivatives
    
    def _continuity_equation(self, derivatives):
        """Continuity equation: ∂u/∂x + ∂v/∂y = 0"""
        return derivatives['u_x'] + derivatives['v_y']
    
    def _momentum_x_equation(self, derivatives, nu_eff, p_prime, k):
        """X-momentum equation with turbulent viscosity."""
        # Convective terms
        u = derivatives['u_x'] * 0 + derivatives['u_y'] * 0  # Get u from derivatives context
        v = derivatives['v_x'] * 0 + derivatives['v_y'] * 0  # Get v from derivatives context
        
        # Note: This is a simplified version. The actual implementation would need
        # proper handling of the convective terms and turbulent stress tensor
        convective = u * derivatives['u_x'] + v * derivatives['u_y']
        
        # Viscous terms (simplified)
        viscous = nu_eff * (derivatives['u_xx'] + derivatives['u_yy'])
        
        # Turbulence gradient term
        k_gradient = (2.0 / 3.0) * derivatives['k_x']
        
        return convective + derivatives['p_x'] - viscous + k_gradient
    
    def _momentum_y_equation(self, derivatives, nu_eff, p_prime, k):
        """Y-momentum equation with turbulent viscosity."""
        # Similar structure to x-momentum equation
        u = derivatives['u_x'] * 0 + derivatives['u_y'] * 0
        v = derivatives['v_x'] * 0 + derivatives['v_y'] * 0
        
        convective = u * derivatives['v_x'] + v * derivatives['v_y']
        viscous = nu_eff * (derivatives['v_xx'] + derivatives['v_yy'])
        k_gradient = (2.0 / 3.0) * derivatives['k_y']
        
        return convective + derivatives['p_y'] - viscous + k_gradient
    
    def _k_equation(self, derivatives, nu_t, k, eps):
        """Turbulent kinetic energy equation."""
        # Convective terms
        u = derivatives['u_x'] * 0 + derivatives['u_y'] * 0
        v = derivatives['v_x'] * 0 + derivatives['v_y'] * 0
        
        convective = u * derivatives['k_x'] + v * derivatives['k_y']
        
        # Diffusion term
        diffusivity_k = self.config.NU + nu_t / self.config.SIGMA_K
        diffusion = diffusivity_k * (derivatives['k_xx'] + derivatives['k_yy'])
        
        # Production term
        S_squared = 2 * (dde.backend.square(derivatives['u_x']) + dde.backend.square(derivatives['v_y'])) + \
                   dde.backend.square(derivatives['u_y'] + derivatives['v_x'])
        production = nu_t * S_squared
        
        return convective - diffusion - production + eps
    
    def _epsilon_equation(self, derivatives, nu_t, k, eps):
        """Turbulent dissipation rate equation."""
        # Convective terms
        u = derivatives['u_x'] * 0 + derivatives['u_y'] * 0
        v = derivatives['v_x'] * 0 + derivatives['v_y'] * 0
        
        convective = u * derivatives['eps_x'] + v * derivatives['eps_y']
        
        # Diffusion term
        diffusivity_eps = self.config.NU + nu_t / self.config.SIGMA_EPS
        diffusion = diffusivity_eps * (derivatives['eps_xx'] + derivatives['eps_yy'])
        
        # Production term
        S_squared = 2 * (dde.backend.square(derivatives['u_x']) + dde.backend.square(derivatives['v_y'])) + \
                   dde.backend.square(derivatives['u_y'] + derivatives['v_x'])
        production_k = nu_t * S_squared
        
        # Source and sink terms
        source = self.config.CEPS1 * (eps / k) * production_k
        sink = self.config.CEPS2 * (dde.backend.square(eps) / k)
        
        return convective - diffusion - source + sink


class BoundaryConditions:
    """Class for defining boundary conditions."""
    
    def __init__(self, config):
        self.config = config
    
    def define_all_bcs(self):
        """Define all boundary conditions for the problem."""
        bcs = []
        
        # Inlet conditions
        bcs.extend(self._inlet_conditions())
        
        # Outlet conditions
        bcs.extend(self._outlet_conditions())
        
        # Wall conditions
        bcs.extend(self._wall_conditions())
        
        # Wall function conditions
        bcs.extend(self._wall_function_conditions())
        
        return bcs
    
    def _inlet_conditions(self):
        """Define inlet boundary conditions."""
        bcs = []
        
        # Velocity components
        bc_u_inlet = dde.DirichletBC(
            self.config.GEOM, 
            lambda x: self.config.U_INLET, 
            self._boundary_inlet, 
            component=0
        )
        bc_v_inlet = dde.DirichletBC(
            self.config.GEOM, 
            lambda x: 0, 
            self._boundary_inlet, 
            component=1
        )
        
        # Turbulence quantities (transformed)
        bc_k_inlet = dde.DirichletBC(
            self.config.GEOM, 
            lambda x: self.config.K_INLET_TRANSFORMED, 
            self._boundary_inlet, 
            component=3
        )
        bc_eps_inlet = dde.DirichletBC(
            self.config.GEOM, 
            lambda x: self.config.EPS_INLET_TRANSFORMED, 
            self._boundary_inlet, 
            component=4
        )
        
        bcs.extend([bc_u_inlet, bc_v_inlet, bc_k_inlet, bc_eps_inlet])
        return bcs
    
    def _outlet_conditions(self):
        """Define outlet boundary conditions."""
        bc_p_outlet = dde.DirichletBC(
            self.config.GEOM, 
            lambda x: 0, 
            self._boundary_outlet, 
            component=2
        )
        return [bc_p_outlet]
    
    def _wall_conditions(self):
        """Define wall boundary conditions (no-slip)."""
        bcs = []
        
        bc_u_walls = dde.DirichletBC(
            self.config.GEOM, 
            lambda x: 0,
            self._boundary_walls,
            component=0
        )
        bc_v_walls = dde.DirichletBC(
            self.config.GEOM, 
            lambda x: 0,
            self._boundary_walls,
            component=1
        )
        
        bcs.extend([bc_u_walls, bc_v_walls])
        return bcs
    
    def _wall_function_conditions(self):
        """Define wall function boundary conditions."""
        bcs = []
        
        # Generate wall function points
        n_wf_points = self.config.NUM_WF_POINTS_PER_WALL
        x_coords = np.linspace(
            self.config.L * 0.01, 
            self.config.L * 0.99, 
            n_wf_points
        )[:, None]
        
        points_bottom = np.hstack((
            x_coords, 
            np.full_like(x_coords, -self.config.CHANNEL_HALF_HEIGHT + self.config.Y_P)
        ))
        points_top = np.hstack((
            x_coords, 
            np.full_like(x_coords, self.config.CHANNEL_HALF_HEIGHT - self.config.Y_P)
        ))
        points_wf = np.vstack((points_bottom, points_top))
        
        # Target values
        u_target_vals = np.full((points_wf.shape[0], 1), self.config.U_TARGET_WF)
        k_target_vals = np.full((points_wf.shape[0], 1), self.config.K_TARGET_WF_TRANSFORMED)
        eps_target_vals = np.full((points_wf.shape[0], 1), self.config.EPS_TARGET_WF_TRANSFORMED)
        
        # Define PointSetBCs
        bc_u_wf = dde.PointSetBC(points_wf, u_target_vals, component=0)
        bc_k_wf = dde.PointSetBC(points_wf, k_target_vals, component=3)
        bc_eps_wf = dde.PointSetBC(points_wf, eps_target_vals, component=4)
        
        bcs.extend([bc_u_wf, bc_k_wf, bc_eps_wf])
        return bcs
    
    def _boundary_inlet(self, x, on_boundary):
        """Check if point is on inlet boundary."""
        return on_boundary and np.isclose(x[0], 0)
    
    def _boundary_outlet(self, x, on_boundary):
        """Check if point is on outlet boundary."""
        return on_boundary and np.isclose(x[0], self.config.L)
    
    def _boundary_walls(self, x, on_boundary):
        """Check if point is on wall boundaries."""
        return on_boundary and (
            np.isclose(x[1], -self.config.CHANNEL_HALF_HEIGHT) or 
            np.isclose(x[1], self.config.CHANNEL_HALF_HEIGHT)
        )
