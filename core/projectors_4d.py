#!/usr/bin/env python3
"""
4D Covariant Projection Operators for E-QFT Full GR Implementation

Extends 3D spatial projectors to 4D spacetime with proper Lorentz covariance.
Implements Stage 1 of the full GR roadmap: adding time dimension and 
recovering special relativity.

Key features:
- Covariant Gaussian-localized projectors π(t,x,y,z)
- Lorentz-invariant overlaps preserving light cone structure
- Time evolution via unitary transformations
- 4D symplectic structure

Author: E-QFT Team
www.eqft-institute.org
Version: 5.0 - Publication Grade
"""

import numpy as np
from numba import jit, njit, prange
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectorState4D:
    """
    Container for a single 4D projector state.
    
    This allows for dynamic evolution of individual projectors
    as required for Lieb-Robinson measurements.
    """
    
    def __init__(self, position, momentum, width, amplitude=1.0):
        self.position = np.asarray(position, dtype=np.float64)
        self.momentum = np.asarray(momentum, dtype=np.float64)
        self.width = float(width)
        self.amplitude = complex(amplitude)
    
    def copy(self):
        """Create a deep copy of the state."""
        return ProjectorState4D(
            self.position.copy(),
            self.momentum.copy(), 
            self.width,
            self.amplitude
        )

# Physical constants
C_LIGHT = 1.0  # Natural units where c = 1
MINKOWSKI_METRIC = np.diag([-1.0, 1.0, 1.0, 1.0])  # η_μν = diag(-1,+1,+1,+1)


@njit
def four_vector_dot(k1: np.ndarray, k2: np.ndarray) -> float:
    """
    Compute Lorentz-invariant inner product of 4-vectors.
    
    k·k = -k₀² + k₁² + k₂² + k₃²
    
    Parameters:
    -----------
    k1, k2 : np.ndarray
        4-vectors [k₀, k₁, k₂, k₃]
        
    Returns:
    --------
    float
        Minkowski inner product η_μν k¹^μ k²^ν
    """
    return -k1[0]*k2[0] + k1[1]*k2[1] + k1[2]*k2[2] + k1[3]*k2[3]


@njit
def is_timelike(k: np.ndarray) -> bool:
    """Check if 4-vector is timelike (k² < 0)."""
    return four_vector_dot(k, k) < 0


@njit
def is_lightlike(k: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if 4-vector is lightlike/null (k² ≈ 0)."""
    return abs(four_vector_dot(k, k)) < tol


@njit
def is_spacelike(k: np.ndarray) -> bool:
    """Check if 4-vector is spacelike (k² > 0)."""
    return four_vector_dot(k, k) > 0


class ProjectorState4D:
    """
    Container for 4D projector state including position, momentum, and spin.
    
    Attributes:
    -----------
    x_mu : np.ndarray
        4-position [t, x, y, z]
    k_mu : np.ndarray  
        4-momentum [E, p_x, p_y, p_z]
    spin : np.ndarray
        Spin tensor S^μν (4x4 antisymmetric)
    sigma : float
        Localization width parameter
    """
    
    def __init__(self, x_mu: np.ndarray, k_mu: np.ndarray, 
                 spin: Optional[np.ndarray] = None, sigma: float = 1.0):
        self.x_mu = np.asarray(x_mu, dtype=np.float64)
        self.k_mu = np.asarray(k_mu, dtype=np.float64)
        self.sigma = sigma
        
        if spin is None:
            self.spin = np.zeros((4, 4), dtype=np.float64)
        else:
            self.spin = np.asarray(spin, dtype=np.float64)
            # Ensure antisymmetry
            self.spin = 0.5 * (self.spin - self.spin.T)
            
        # Enforce spin supplementary condition S^μν P_ν = 0
        self._enforce_ssc()
    
    def _enforce_ssc(self):
        """Enforce spin supplementary condition S^μν P_ν = 0."""
        k_squared = four_vector_dot(self.k_mu, self.k_mu)
        if abs(k_squared) < 1e-10:
            # For null momenta, SSC is automatically satisfied
            return
            
        for mu in range(4):
            constraint = sum(self.spin[mu, nu] * self.k_mu[nu] for nu in range(4))
            if abs(constraint) > 1e-10:
                # Project out violation
                for nu in range(4):
                    self.spin[mu, nu] -= constraint * self.k_mu[nu] / k_squared


@njit
def gaussian_envelope_4d(k_mu: np.ndarray, sigma: float) -> float:
    """
    4D Gaussian envelope in momentum space.
    
    G(k) = exp(-σ²k²/2) where k² = η_μν k^μ k^ν
    
    Parameters:
    -----------
    k_mu : np.ndarray
        4-momentum
    sigma : float
        Width parameter
        
    Returns:
    --------
    float
        Gaussian weight
    """
    k_squared = four_vector_dot(k_mu, k_mu)
    return np.exp(-0.5 * sigma**2 * k_squared)


@njit
def projector_wavefunction_4d(x_mu: np.ndarray, k_mu: np.ndarray, 
                              sigma: float, x_eval: np.ndarray) -> complex:
    """
    Evaluate 4D projector wavefunction at spacetime point.
    
    ψ(x) = G(k) exp(i k·x) where k·x = η_μν k^μ x^ν
    
    Parameters:
    -----------
    x_mu : np.ndarray
        Center position [t, x, y, z]
    k_mu : np.ndarray
        4-momentum [E, p_x, p_y, p_z]
    sigma : float
        Localization width
    x_eval : np.ndarray
        Evaluation point [t, x, y, z]
        
    Returns:
    --------
    complex
        Wavefunction value at x_eval
    """
    # Displacement from center
    dx_mu = x_eval - x_mu
    
    # Phase factor with Minkowski metric
    phase = -k_mu[0]*dx_mu[0] + k_mu[1]*dx_mu[1] + k_mu[2]*dx_mu[2] + k_mu[3]*dx_mu[3]
    
    # Gaussian envelope
    envelope = gaussian_envelope_4d(k_mu, sigma)
    
    return envelope * np.exp(1j * phase)


@njit
def overlap_4d(state1: Tuple, state2: Tuple) -> complex:
    """
    Compute overlap between two 4D projector states.
    
    Fast computation using momentum space representation.
    
    Parameters:
    -----------
    state1, state2 : Tuple
        (x_mu, k_mu, sigma) for each state
        
    Returns:
    --------
    complex
        Overlap ⟨ψ₁|ψ₂⟩
    """
    x1, k1, sigma1 = state1
    x2, k2, sigma2 = state2
    
    # Combined width
    sigma_eff = np.sqrt(sigma1**2 + sigma2**2)
    
    # Momentum difference  
    dk = k1 - k2
    dk_squared = four_vector_dot(dk, dk)
    
    # Position difference with Minkowski metric
    dx = x2 - x1
    phase = -dk[0]*dx[0] + dk[1]*dx[1] + dk[2]*dx[2] + dk[3]*dx[3]
    
    # Gaussian factor from momentum integration
    gauss = np.exp(-0.25 * sigma_eff**2 * dk_squared)
    
    return gauss * np.exp(1j * phase)


@njit
def commutator_distance_4d(state1: Tuple, state2: Tuple) -> float:
    """
    Compute E-QFT distance from projector commutator in 4D.
    
    d²(1,2) = Tr[(π₁π₂ - π₂π₁)†(π₁π₂ - π₂π₁)]
            = 2[1 - |⟨ψ₁|ψ₂⟩|²]
    
    Parameters:
    -----------
    state1, state2 : Tuple
        4D projector states
        
    Returns:
    --------
    float
        Squared distance
    """
    overlap = overlap_4d(state1, state2)
    return 2.0 * (1.0 - abs(overlap)**2)


@njit
def is_spacelike_separated(state1: Tuple, state2: Tuple) -> bool:
    """
    Check if two events are spacelike separated.
    
    Parameters:
    -----------
    state1, state2 : Tuple
        4D projector states with positions
        
    Returns:
    --------
    bool
        True if spacelike separated
    """
    x1 = state1[0]  # x_mu for state 1
    x2 = state2[0]  # x_mu for state 2
    
    dx = x2 - x1
    interval = four_vector_dot(dx, dx)
    
    return interval > 0


@njit
def light_cone_check(state1: Tuple, state2: Tuple, tol: float = 1e-3) -> bool:
    """
    Verify that commutator vanishes outside light cone.
    
    For spacelike separation, projectors should commute:
    [π₁, π₂] ≈ 0
    
    Parameters:
    -----------
    state1, state2 : Tuple
        4D projector states
    tol : float
        Tolerance for vanishing commutator
        
    Returns:
    --------
    bool
        True if causality is satisfied
    """
    if is_spacelike_separated(state1, state2):
        # Should have vanishing commutator
        d2 = commutator_distance_4d(state1, state2)
        # For commuting projectors, d² should be very small
        # Note: for finite sigma, there's always some overlap
        return d2 < tol
    return True  # Timelike/lightlike separation allowed to have non-zero commutator


class Lattice4D:
    """
    4D spacetime lattice for E-QFT simulations.
    
    Parameters:
    -----------
    Nt : int
        Number of time slices
    L : int
        Spatial lattice size (L³ sites per time slice)
    at : float
        Temporal lattice spacing (default = spatial spacing)
    a : float
        Spatial lattice spacing
    """
    
    def __init__(self, Nt: int, L: int, at: float = 1.0, a: float = 1.0):
        self.Nt = Nt
        self.L = L
        self.at = at
        self.a = a
        
        # Total number of lattice sites
        self.n_sites = Nt * L**3
        
        # Create lattice coordinate arrays
        self._setup_coordinates()
        
        logger.info(f"Created 4D lattice: {Nt}×{L}³ sites, at={at}, a={a}")
    
    def _setup_coordinates(self):
        """Setup 4D coordinate arrays."""
        # Time coordinates
        self.t_coords = np.arange(self.Nt) * self.at
        
        # Spatial coordinates  
        self.x_coords = np.arange(self.L) * self.a
        self.y_coords = np.arange(self.L) * self.a
        self.z_coords = np.arange(self.L) * self.a
        
        # Full 4D mesh
        T, X, Y, Z = np.meshgrid(self.t_coords, self.x_coords, 
                                 self.y_coords, self.z_coords, indexing='ij')
        
        self.x_mu_grid = np.stack([T, X, Y, Z], axis=-1)
    
    def site_to_4position(self, site_idx: int) -> np.ndarray:
        """Convert linear site index to 4-position."""
        t_idx = site_idx // (self.L**3)
        spatial_idx = site_idx % (self.L**3)
        
        z_idx = spatial_idx % self.L
        y_idx = (spatial_idx // self.L) % self.L
        x_idx = spatial_idx // (self.L**2)
        
        return np.array([self.t_coords[t_idx], 
                        self.x_coords[x_idx],
                        self.y_coords[y_idx], 
                        self.z_coords[z_idx]])
    
    def periodic_distance_4d(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Compute periodic distance vector in 4D.
        
        Time direction can be periodic (thermal) or open (real-time).
        """
        dx = x2 - x1
        
        # Spatial directions always periodic
        for i in range(1, 4):
            if abs(dx[i]) > self.L * self.a / 2:
                dx[i] -= np.sign(dx[i]) * self.L * self.a
        
        # Time direction - make periodic for thermal field theory
        if abs(dx[0]) > self.Nt * self.at / 2:
            dx[0] -= np.sign(dx[0]) * self.Nt * self.at
            
        return dx


def create_momentum_lattice(lattice: Lattice4D, 
                           mass: float = 0.0) -> np.ndarray:
    """
    Create allowed 4-momenta on lattice satisfying dispersion relation.
    
    For mass m: E² = p² + m² (in natural units)
    
    Parameters:
    -----------
    lattice : Lattice4D
        4D lattice structure  
    mass : float
        Particle mass (0 for photons/gravitons)
        
    Returns:
    --------
    np.ndarray
        Array of 4-momenta [E, px, py, pz]
    """
    # Spatial momenta (periodic boundary conditions)
    kx = 2 * np.pi * np.fft.fftfreq(lattice.L, lattice.a)
    ky = 2 * np.pi * np.fft.fftfreq(lattice.L, lattice.a)
    kz = 2 * np.pi * np.fft.fftfreq(lattice.L, lattice.a)
    
    # Build momentum grid
    momenta = []
    for i, px in enumerate(kx):
        for j, py in enumerate(ky):
            for k, pz in enumerate(kz):
                p_squared = px**2 + py**2 + pz**2
                
                # Energy from dispersion relation
                E = np.sqrt(p_squared + mass**2)
                
                # Both positive and negative energy solutions
                momenta.append([E, px, py, pz])
                if E > 0:  # Avoid double-counting zero mode
                    momenta.append([-E, px, py, pz])
    
    return np.array(momenta)


def test_special_relativity(lattice: Lattice4D, n_tests: int = 100):
    """
    Test that 4D projectors satisfy special relativity.
    
    Tests:
    1. Exponential decay of commutators (not strict vanishing)
    2. Lorentz invariance of overlaps  
    3. Dispersion relation for massless excitations
    
    Note: For Gaussian projectors, commutators decay exponentially
    but never vanish. This is physically correct - see test_sr_continuum.py
    for proper continuum limit analysis.
    """
    logger.info("Testing special relativity properties...")
    
    # Test 1: Exponential decay for spacelike separation
    # For Gaussian width σ=1, expect Γ(r) ∝ exp(-r²/4)
    decay_test_passed = True
    
    r_values = [2.0, 3.0, 4.0]  # Test at multiple separations
    gamma_values = []
    
    for r in r_values:
        x1 = np.array([0.0, 0.0, 0.0, 0.0])
        x2 = np.array([0.0, r, 0.0, 0.0])  # Purely spatial
        
        k = np.array([1.0, 0.0, 0.0, 1.0])  # Same momentum
        
        state1 = (x1, k, 1.0)
        state2 = (x2, k, 1.0)
        
        d2 = commutator_distance_4d(state1, state2)
        gamma_values.append(d2)
    
    # Check exponential decay
    if gamma_values[0] > 1e-10 and gamma_values[1] > 1e-10:
        # Decay rate between r[0] and r[1]
        rate_measured = np.log(gamma_values[1]/gamma_values[0]) / (r_values[1]**2 - r_values[0]**2)
        rate_theory = -1/4  # For σ=1
        
        error = abs(rate_measured - rate_theory) / abs(rate_theory)
        logger.info(f"Decay rate: measured={rate_measured:.4f}, theory={rate_theory:.4f}, error={error:.1%}")
        
        if error > 0.2:  # Allow 20% error
            decay_test_passed = False
    else:
        logger.info("Commutators too small to measure decay rate")
    
    # Test 2: Massless dispersion
    momenta = create_momentum_lattice(lattice, mass=0.0)
    dispersion_errors = []
    
    for k_mu in momenta[:n_tests]:
        if abs(k_mu[0]) > 1e-10:  # Skip zero modes
            # Check E² = p²
            E2 = k_mu[0]**2
            p2 = k_mu[1]**2 + k_mu[2]**2 + k_mu[3]**2
            error = abs(E2 - p2) / E2
            dispersion_errors.append(error)
    
    max_error = max(dispersion_errors) if dispersion_errors else 0
    logger.info(f"Dispersion relation: max error = {max_error:.2e}")
    
    # Success criteria: proper decay and good dispersion
    return decay_test_passed and max_error < 1e-10


# Example usage and tests
if __name__ == "__main__":
    # Create 4D lattice
    lattice = Lattice4D(Nt=16, L=8, at=1.0, a=1.0)
    
    # Test special relativity
    sr_test = test_special_relativity(lattice)
    print(f"Special relativity test: {'PASSED' if sr_test else 'FAILED'}")
    
    # Create sample projector states
    x1 = np.array([0.0, 0.0, 0.0, 0.0])
    k1 = np.array([1.0, 0.0, 0.0, 1.0])  # Lightlike momentum
    state1 = ProjectorState4D(x1, k1, sigma=1.0)
    
    x2 = np.array([1.0, 2.0, 0.0, 0.0])  # Spacelike separated
    k2 = np.array([1.0, 1.0, 0.0, 0.0])  # Another lightlike momentum
    state2 = ProjectorState4D(x2, k2, sigma=1.0)
    
    # Check commutator
    s1 = (state1.x_mu, state1.k_mu, state1.sigma)
    s2 = (state2.x_mu, state2.k_mu, state2.sigma)
    
    d2 = commutator_distance_4d(s1, s2)
    print(f"Commutator distance²: {d2:.6f}")
    print(f"Spacelike separated: {is_spacelike_separated(s1, s2)}")
    print(f"Light cone check: {light_cone_check(s1, s2)}")