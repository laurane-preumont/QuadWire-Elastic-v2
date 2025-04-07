"""
Shape Optimization Module

This module implements shape optimization algorithms including displacement
minimization and counter-deformation optimization using tuple-based inputs.
"""


import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as spla
import time
import os
from typing import List, Tuple, Optional, Union, Any, TypeVar, Protocol
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from shape import derivateOffset
from shape.derivateOffset import fixed_structure

# Type aliases
ArrayFloat = npt.NDArray[np.float64]
SparseMatrix = Union[sp.csr_matrix, sp.csc_matrix]
T = TypeVar('T')

# Type definitions for tuple inputs
MeshingTuple = Tuple[float, float, float, int, int, int, str, str, str, bool]  # L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
LoadingTuple = Tuple[float, str]  # dT, loadType
DiscretizationTuple = Tuple[int, int]  # elemOrder, quadOrder
MaterialTuple = Tuple[float, float, float, str]  # E, nu, alpha, optimizedBehavior
PlottingTuple = Tuple[bool, str, float]  # toPlot, clrmap, scfplot


class TerminationReason(Enum):
    """Possible reasons for algorithm termination."""
    MAX_ITERATIONS = "Maximum iterations reached"
    COST_TOLERANCE = "Cost function tolerance reached"
    FLAT_GRADIENT = "Gradient became flat (converged)"
    RELATIVE_IMPROVEMENT = "Relative improvement tolerance reached"


@dataclass
class OptimizationState:
    """Holds current state of optimization."""
    param: ArrayFloat
    offset: ArrayFloat
    us: ArrayFloat
    yKy: SparseMatrix
    yfbc: ArrayFloat
    Jval: float
    J_prime: ArrayFloat
    mu: float
    iteration: int
    param_history: List[ArrayFloat]
    mu_history: List[float]
    J_history: List[float]
    Jprime_history: List[ArrayFloat]
    G_history: List[ArrayFloat]


############################
# Cost function definition #
############################

class CostFunction(Protocol):
    """Protocol defining interface for cost functions."""

    def __call__(self, us: ArrayFloat, **kwargs: Any) -> float:
        """Calculate cost value."""
        ...

    def gradient(self, us: ArrayFloat, p: ArrayFloat, offset: ArrayFloat, loading: LoadingTuple, discretization: DiscretizationTuple, material: MaterialTuple, **kwargs: Any) -> ArrayFloat:
        """Calculate cost function gradient."""
        ...

    def adjoint_rhs(self, us: ArrayFloat, **kwargs: Any) -> ArrayFloat:
        """Calculate right-hand side of adjoint equation."""
        ...


class DisplacementMinimizationCost:
    """Cost function for minimizing displacements using L2-norm squared."""

    def __init__(self):
        self.displacement_scale = None
        self.initialized = False

    def __call__(self, us: ArrayFloat, **kwargs: Any) -> float:
        if not self.initialized:
            self.displacement_scale = max(np.sqrt(np.mean(np.square(us))), 1e-8)
            self.initialized = True
            print(f"Initializing displacement scale: {self.displacement_scale:.2e}")

        us_scaled = us / self.displacement_scale
        return float(np.sum(us_scaled * us_scaled))


    def adjoint_rhs(self, us: ArrayFloat, **kwargs: Any) -> ArrayFloat:
        if not self.initialized:
            raise ValueError("Cost function not initialized - call it first")
        us_scaled = us / self.displacement_scale
        return -2 * us_scaled / self.displacement_scale    # derivative of |u/scale|² is 2/scale*|u/scale|

    def gradient(self, us: ArrayFloat, p: ArrayFloat, offset: ArrayFloat, loading: LoadingTuple, discretization: DiscretizationTuple, material: MaterialTuple, **kwargs: Any) -> ArrayFloat:
        required_keys = {'meshing'}
        if not all(key in kwargs for key in required_keys):
            raise ValueError(f"Missing required parameters: {required_keys - set(kwargs.keys())}")
        meshing = kwargs['meshing']
        _, _, _, _, nLayers_v, *_ = meshing
        J_prime = np.zeros(nLayers_v, dtype=np.float64)

        print("\nGradient computation:")
        for i in range(nLayers_v):
            Aprime_i = derivateOffset.derive_yKy(meshing, offset, loading, discretization, material, i)
            fprime_i = derivateOffset.derive_yfbc(meshing, offset, loading, discretization, material, i)

            J_prime[i] = np.dot(p, Aprime_i @ us[:, np.newaxis] - fprime_i)

        return J_prime

class CounterDeformationCost:
    """Base cost function for counter-deformation optimization."""

    def _compute_parameters(self, meshing, offset, offset0, Y):
        """Helper method to compute common parameters."""
        param = offset2param(meshing, offset)
        param0 = offset2param(meshing, offset0)
        param_2uslike = param2us(meshing, param, Y)
        param0_2uslike = param2us(meshing, param0, Y)
        return param, param0, param_2uslike, param0_2uslike

    def _check_required_keys(self, kwargs):
        """Validate required parameters are present."""
        required_keys = {'meshing', 'Y', 'initial_offset', 'current_offset'}
        if not all(key in kwargs for key in required_keys):
            raise ValueError(f"Missing required parameters: {required_keys - set(kwargs.keys())}")
        return kwargs['meshing'], kwargs['Y'], kwargs['initial_offset'], kwargs['current_offset']

    def _compute_difference(self, us, param_2uslike, param0_2uslike):
        """Compute the difference term."""
        if not sp.issparse(us):
            us = sp.csr_matrix(us)
        if not sp.issparse(param_2uslike):
            param_2uslike = sp.csr_matrix(param_2uslike)
        if not sp.issparse(param0_2uslike):
            param0_2uslike = sp.csr_matrix(param0_2uslike)

        us = us.reshape(-1, 1)
        param_2uslike = param_2uslike.reshape(-1, 1)
        param0_2uslike = param0_2uslike.reshape(-1, 1)

        n = us.shape[0]
        if param_2uslike.shape != (n, 1):
            raise ValueError(f"param_2uslike shape {param_2uslike.shape} does not match us shape {us.shape}")
        if param0_2uslike.shape != (n, 1):
            raise ValueError(f"param0_2uslike shape {param0_2uslike.shape} does not match us shape {us.shape}")

        diff = us + param_2uslike - param0_2uslike
        return diff

    def __call__(self, us: ArrayFloat, **kwargs: Any) -> float:
        """Calculate cost function value using sparse operations."""
        meshing, Y, offset0, offset = self._check_required_keys(kwargs)
        _, _, param_2uslike, param0_2uslike = self._compute_parameters(meshing, offset, offset0, Y)
        diff = self._compute_difference(us, param_2uslike, param0_2uslike)
        diff_squared = diff.multiply(diff)
        sum = (diff_squared).sum()
        return float(sum)

    def adjoint_rhs(self, us: ArrayFloat, **kwargs: Any) -> ArrayFloat:
        meshing, Y, offset0, offset = self._check_required_keys(kwargs)
        _, _, param_2uslike, param0_2uslike = self._compute_parameters(meshing, offset, offset0, Y)
        diff = self._compute_difference(us, param_2uslike, param0_2uslike)
        rhs = -2 * diff
        return rhs.todense()

    def gradient(self, us: ArrayFloat, p: ArrayFloat, offset: ArrayFloat,
                loading: LoadingTuple, discretization: DiscretizationTuple,
                material: MaterialTuple, **kwargs: Any) -> ArrayFloat:
        meshing, Y, offset0, _ = self._check_required_keys(kwargs)
        _, _, _, _, nLayers_v, *_ = meshing
        param, param0, param_2uslike, param0_2uslike = self._compute_parameters(
            meshing, offset, offset0, Y)

        diff = self._compute_difference(us, param_2uslike, param0_2uslike)

        J_prime = np.zeros(nLayers_v, dtype=np.float64)

        for i in range(nLayers_v):
            Aprime_i = derivateOffset.derive_yKy(meshing, offset, loading, discretization, material, i)
            fprime_i = derivateOffset.derive_yfbc(meshing, offset, loading, discretization, material, i)
            implicit_term_i = -np.dot(p, Aprime_i @ us[:, np.newaxis] - fprime_i)

            dM_dparam_i = d_param2us(meshing, param - param0, Y, i)
            explicit_term_i = 2 * dM_dparam_i.T @ diff

            J_prime[i] = implicit_term_i + explicit_term_i

        return J_prime


class ScaledCounterDeformationCost(CounterDeformationCost):
    """Counter-deformation cost function with dimensionless scaling.
    Normalizes displacements and offsets using their RMS values to improve
    numerical conditioning during optimization.
    """

    def __init__(self):
        super().__init__()
        self.displacement_scale = None
        self.offset_scale = None
        self.initialized = False

    def _initialize_scaling(self, us: ArrayFloat, param: ArrayFloat) -> None:
        """Initialize scale factors using RMS values with minimum thresholds.

        Args:
            us: Current displacement solution vector
            param: Current offset parameters

        Scale factors are computed as max(RMS, threshold) to avoid division by
        very small values while preserving relative magnitudes.
        """
        if not self.initialized:
            self.displacement_scale = max(np.sqrt(np.mean(np.square(us))), 1e-8)
            self.offset_scale = max(np.sqrt(np.mean(np.square(param))), 1e-5)
            self.initialized = True
            print(f"Initialized scaling factors:")
            print(f"Displacement scale: {self.displacement_scale:.2e}")
            print(f"Offset scale: {self.offset_scale:.2e}")

    def _compute_parameters(self, meshing, offset, offset0, Y, us=None):
       """Compute parameters and convert to dimensionless quantities.

       Args:
           meshing: Mesh parameters
           offset: Current offset values [m]
           offset0: Target offset values [m]
           Y: Assembly matrix
           us: Optional displacement solution [m] for initial scaling

       Returns:
           param: Raw parameters [m]
           param0: Raw target parameters [m]
           param_2uslike: Dimensionless mapped current parameters [-]
           param0_2uslike: Dimensionless mapped target parameters [-]
       """
       param = offset2param(meshing, offset)
       param0 = offset2param(meshing, offset0)

       if not self.initialized and us is not None:
           self.displacement_scale = max(np.sqrt(np.mean(np.square(us))), 1e-8)
           self.offset_scale = max(np.sqrt(np.mean(np.square(param))), 1e-5)
           self.initialized = True

       if not self.initialized:
           raise ValueError("Scaling not initialized - must provide us on first call")

       param_2uslike = param2us(meshing, param, Y) / self.offset_scale
       param0_2uslike = param2us(meshing, param0, Y) / self.offset_scale

       return param, param0, param_2uslike, param0_2uslike


    def __call__(self, us: ArrayFloat, **kwargs: Any) -> float:
        """Compute cost using dimensionless quantities."""
        meshing, Y, offset0, offset = self._check_required_keys(kwargs)
        param, param0, param_2uslike, param0_2uslike = self._compute_parameters(
            meshing, offset, offset0, Y, us)

        self._initialize_scaling(us, param)

        # Scale current us
        us_scaled = us / self.displacement_scale

        diff_scaled = self._compute_difference(us_scaled, param_2uslike, param0_2uslike)

        diff_squared = diff_scaled.multiply(diff_scaled)
        sum_scaled = float(diff_squared.sum())

        return sum_scaled

    def adjoint_rhs(self, us: ArrayFloat, **kwargs: Any) -> ArrayFloat:
        """Compute scaled adjoint RHS.

        Returns:
            -2 * (diff/S_o) * (S_d/S_o) for consistent scaling
        """
        meshing, Y, offset0, offset = self._check_required_keys(kwargs)
        param, param0, param_2uslike, param0_2uslike = self._compute_parameters(
            meshing, offset, offset0, Y)

        self._initialize_scaling(us, param)

        # Get scaled difference
        us_scaled = us / self.displacement_scale
        diff_scaled = self._compute_difference(us_scaled, param_2uslike, param0_2uslike)

        rhs = -2 * diff_scaled
        return rhs.todense()

    def gradient(self, us: ArrayFloat, p: ArrayFloat, offset: ArrayFloat,
                loading: LoadingTuple, discretization: DiscretizationTuple,
                material: MaterialTuple, **kwargs: Any) -> ArrayFloat:
        """Compute scaled gradient ensuring consistent units.

        Scales adjoint variable p by (S_d/S_o) to maintain correct relationship
        between cost function and its gradient.
        """
        meshing, Y, offset0, _ = self._check_required_keys(kwargs)
        param, param0, param_2uslike, param0_2uslike = self._compute_parameters(
            meshing, offset, offset0, Y)
        nLayers_v = param.shape[0]

        self._initialize_scaling(us, param)

        # Get scaled difference
        us_scaled = us / self.displacement_scale
        diff_scaled = self._compute_difference(us_scaled, param_2uslike, param0_2uslike)

        J_prime = np.zeros(nLayers_v, dtype=np.float64)

        for i in range(nLayers_v):
            Aprime_i = derivateOffset.derive_yKy(meshing, offset, loading, discretization, material, i)
            fprime_i = derivateOffset.derive_yfbc(meshing, offset, loading, discretization, material, i)
            implicit_term_i = -np.dot(p, Aprime_i @ us[:, np.newaxis] - fprime_i)

            dM_dparam_i = d_param2us(meshing, param - param0, Y, i)
            explicit_term_i = 2 * dM_dparam_i.T @ diff_scaled

            J_prime[i] = implicit_term_i + explicit_term_i

        return J_prime

########################
# conversion operators #
########################

def param2overhang(param: ArrayFloat) -> ArrayFloat:
    """
    Convert layerwise offset parameters to overhang values.

    Args:
        param: Array of layerwise parameters [n]

    Returns:
        Array of overhang values [n]

    Mathematical expression:
        overhang_i = param_i - param_{i-1}
    """
    return np.hstack((0, np.diff(param)))


def offset2param(meshing: MeshingTuple, offset: ArrayFloat) -> ArrayFloat:
    """Convert offset values to average layer parameters."""
    _, _, _, nLayers_h, nLayers_v, nNodes, *_ = meshing
    return np.average(offset[1].reshape(nLayers_v, nNodes * nLayers_h), axis=1)


def param2offset(meshing: MeshingTuple, param: ArrayFloat) -> ArrayFloat:
    """Convert layer parameters to offset shape."""
    _, _, _, nLayers_h, _, nNodes, *_ = meshing
    nNodes_h = nNodes * nLayers_h

    offset = np.zeros((2, param.size * nNodes_h), dtype=np.float64)
    offset[1, :] = np.repeat(param, nNodes_h)
    return offset


def offset2u(offset) -> ArrayFloat:
    """Convert offset values to the same shape as us."""
    offset_tnb = np.vstack((offset, np.zeros((1, offset.shape[1]))))
    offset_tnb_4 = np.tile(offset_tnb, 4)
    ulike = sp.csr_matrix(offset_tnb_4.flatten()[:, np.newaxis])
    return ulike   # column vector


def param2u(meshing: MeshingTuple, param: ArrayFloat) -> ArrayFloat:
    """Convert layer parameters to equivalent u shape."""
    return offset2u(param2offset(meshing, param))

def u2us(u, Y):
    YtY = Y.T @ Y  # This is square and positive definite
    Ytu = Y.T @ u
    us = spsolve(YtY, Ytu)  # us = (Y.T @ Y)^-1 @ Y.T @ u
    return us # shape (n,)

def param2us(meshing, param, Y):  # operator M in latex
    """Convert parameter values to the same shape as us"""
    ulike = param2u(meshing, param)
    uslike = u2us(ulike, Y)  # Y depends on param
    return sp.csr_matrix(uslike).reshape(-1,1)

def offset2us(offset, Y):
    ulike = offset2u(offset)  # offset2u
    uslike = u2us(ulike, Y)  # u2us --> c'est uniquement cette opération qui dépend de param via Y
    return uslike


def us2param(meshing, us, Y):
    """Convert us values to the same shape as offset with layerwise average."""
    _, _, _, nLayers_h, nLayers_v, nNodes, *_ = meshing
    if len(us.shape) < 2 :
        us = us[:, np.newaxis]
    u = Y @ us   # us2u
    nNodesTot = u.shape[0] // 12
    U = u.reshape((3, 4, nNodesTot))
    U_avg = np.average(U, axis=1)  # average of node displacements
    U_layers = U_avg.reshape(3, nLayers_v, nNodes)  # split data per layer
    paramlike = np.average(U_layers, axis=2)
    return paramlike[1]



######################################
# derivative of conversion operators #
######################################

def d_param2u(meshing, param, i):
    """ derivative of param2u with respect to param_i
        Note: Linear function
    """
    e_i = param * 0
    e_i[i] = 1
    return param2u(meshing, e_i)

def d_param2us(meshing: MeshingTuple, param: ArrayFloat, Y: sp.spmatrix, i: int) -> sp.spmatrix:
    """
    Compute derivative of param2us with respect to param[i].
    check_param2us_derivative shows that this derivative computation is correct.

    Shape analysis:
    - d(param2u)/d(param[i]): (n_u, 1)
    - d(u2us)/d(u): (n_us, n_u)
    - Result: (n_us, 1)
    """
    # Get d(param2u)/d(param[i])
    dparam2u = d_param2u(meshing, param, i)  # Shape: (n_u, 1)

    # d(u2us)/d(u) = (Y^T Y)^(-1) Y^T
    YtY = Y.T @ Y
    du2us = spla.spsolve(YtY, Y.T.toarray())  # Shape: (n_us, n_u)

    # Apply chain rule
    return du2us @ dparam2u  # Shape: (n_us, n_u) × (n_u, 1) = (n_us, 1)

##################################
# Functions to check derivatives #
##################################

# Run checks with :    param2u_results, param2us_results = check_all_derivatives(meshing, param, Y)

def check_derivative_finiteDiff(func, func_derivative, x, i, epsilon=1e-6, rtol=1e-5, atol=1e-8, **kwargs):
    """
    Generic function to check a derivative against finite differences.

    Args:
        func: Function to check derivative of
        func_derivative: Function computing analytical derivative
        x: Point at which to check derivative
        i: Index of parameter to check derivative for
        epsilon: Step size for finite difference
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        **kwargs: Additional arguments passed to both functions

    Returns:
        (is_correct, error, analytical, numerical)
    """
    # Create perturbed input
    x_eps = x.copy()
    x_eps[i] += epsilon

    # Evaluate function at both points
    f_x = func(x, **kwargs)
    f_x_eps = func(x_eps, **kwargs)

    # Compute finite difference
    finite_diff = (f_x_eps - f_x) / epsilon

    # Get analytical derivative
    analytical_diff = func_derivative(x, i, **kwargs)

    # Convert to dense for comparison if needed
    if sp.issparse(finite_diff):
        finite_diff = finite_diff.toarray()
    if sp.issparse(analytical_diff):
        analytical_diff = analytical_diff.toarray()

    # Compute error
    abs_error = np.linalg.norm(finite_diff - analytical_diff)
    rel_error = abs_error / (np.linalg.norm(analytical_diff) + 1e-10)

    # Check if within tolerance
    is_correct = (abs_error <= atol) or (rel_error <= rtol)

    return is_correct, rel_error, analytical_diff, finite_diff


def check_param2u_derivative(meshing, param, epsilon=1e-6):
    """Check param2u derivative."""
    def func(p, meshing=meshing):
        return param2u(meshing, p)

    def func_derivative(p, i, meshing=meshing):
        return d_param2u(meshing, p, i)

    results = []
    for i in range(len(param)):
        is_correct, error, analytical, numerical = check_derivative_finiteDiff(
            func, func_derivative, param, i, epsilon=epsilon
        )
        results.append({
            'param_idx': i,
            'is_correct': is_correct,
            'error': error,
            'analytical': analytical,
            'numerical': numerical
        })
        print(f"param2u derivative check for param[{i}]: "
              f"{'✓' if is_correct else '✗'} (rel_error={error:.2e})")
    return results

def check_param2us_derivative(meshing, param, Y, epsilon=1e-6):
    """Check param2us derivative."""
    def func(p, meshing=meshing, Y=Y):
        return param2us(meshing, p, Y)

    def func_derivative(p, i, meshing=meshing, Y=Y):
        return d_param2us(meshing, p, Y, i)

    results = []
    for i in range(len(param)):
        is_correct, error, analytical, numerical = check_derivative_finiteDiff(
            func, func_derivative, param, i, epsilon=epsilon
        )
        results.append({
            'param_idx': i,
            'is_correct': is_correct,
            'error': error,
            'analytical': analytical,
            'numerical': numerical
        })
        print(f"param2us derivative check for param[{i}]: "
              f"{'✓' if is_correct else '✗'} (rel_error={error:.2e})")
    return results


def check_all_derivatives(meshing, param, Y, epsilon=1e-6):
    """Run all derivative checks and print a summary."""
    print("\nChecking param2u derivatives:")
    param2u_results = check_param2u_derivative(meshing, param, epsilon)

    print("\nChecking param2us derivatives:")
    param2us_results = check_param2us_derivative(meshing, param, Y, epsilon)

    # Print summary
    print("\nSummary:")
    print("param2u: ", sum(r['is_correct'] for r in param2u_results), "/", len(param2u_results), "tests passed")
    print("param2us: ", sum(r['is_correct'] for r in param2us_results), "/", len(param2us_results), "tests passed")

    return param2u_results, param2us_results

####################
# gradient descent #
####################

def compute_adjoint_state(yKy: SparseMatrix, cost_function: CostFunction, us: ArrayFloat, **cost_kwargs: Any) -> ArrayFloat:
    """Compute the adjoint state by solving the adjoint equation."""
    rhs = cost_function.adjoint_rhs(us, **cost_kwargs)
    print("\nAdjoint computation:")
    print(f"|rhs|: {norm(rhs):.6e}")
    print(f"Condition number estimate of yKy: {norm(yKy.todense()) * norm(np.linalg.inv(yKy.todense())):.6e}")
    print(yKy.todense().shape)
    print(np.sum((yKy-yKy.T).data))
    plt.spy(yKy, markersize=1), plt.show()


    yKy_copy = yKy.copy()
    rhs_copy = rhs.copy()
    try:
        p = spla.spsolve(yKy.T, rhs)
        print(f"|p|: {norm(p):.6e}")
        if not np.all(np.isfinite(p)):
            raise ValueError("Adjoint solution contains NaN or Inf values")
        # Verify adjoint equation
        #residual = norm(yKy.T @ p - rhs)
        #residual = norm(yKy.T @ p - rhs) / norm(rhs)
        residual = norm(yKy_copy.T @ p - rhs_copy) / norm(rhs_copy)
        print(f"Adjoint equation residual normé: {residual :.6e}")
        return p
    except spla.MatrixRankWarning:
        raise ValueError("Stiffness matrix is singular or badly conditioned")
    except RuntimeError as e:
        raise RuntimeError(f"Sparse solver failed: {str(e)}")

def project_onto_constraints_ortho(param: ArrayFloat, overhang_max: float) -> ArrayFloat:
    """Project parameters onto feasible region."""
    nLayers_v = len(param)

    A = sp.eye(nLayers_v, format='csr')
    A[1:, :-1] -= sp.eye(nLayers_v - 1, format='csr')
    A = A[1:]

    overhang = A @ param
    signs = 2 * (overhang > 0) - 1.0
    A = sp.diags(signs) @ A

    b = (overhang_max-1e-8) * np.ones(nLayers_v - 1)
    z = spla.spsolve(A @ A.T, A @ param - b)
    return param - A.T @ z


def G(param: ArrayFloat, overhang_max: float, tolerance: float = 1e-9) -> ArrayFloat:
    """Compute overhang constraints: G(param) < 0."""
    if overhang_max <= 0:
        raise ValueError("Maximum overhang must be positive")

    overhang = param2overhang(param)
    overhang_abs = np.abs(overhang[1:]) + tolerance
    return overhang_abs - overhang_max


def analyze_convergence(J_history: List[float], window_size: int) -> Tuple[float, bool]:
    """
    Analyze convergence behavior.

    Args:
        J_history: History of cost function values
        window_size: Number of iterations to analyze

    Returns:
        Tuple[float, bool]: (mean relative improvement, oscillation flag)
    """
    if len(J_history) < window_size + 1:
        return 0.0, False

    # Calculate improvements
    J_arr = np.array(J_history)
    # Get window slice and calculate differences
    window = J_arr[-window_size:]
    improvements = np.diff(window)  # len = window_size - 1

    # Calculate relative improvements using the correct corresponding values
    rel_improvements = improvements / np.maximum(np.abs(window[:-1]), 1e-10)

    # Analyze oscillation
    signs = np.sign(rel_improvements)
    sign_changes = np.sum(np.abs(np.diff(signs))) / (len(signs) - 1)

    return float(np.mean(rel_improvements)), bool(sign_changes > 0.5)


def check_relative_convergence(J_history: List[float], successful_steps: List[bool], rel_tolerance: float = 1e-8, window_size: int = 5) -> bool:
    """
    Check relative improvement convergence considering only successful steps.

    Returns True if the mean absolute relative improvement over successful steps
    within the window is smaller than the tolerance, indicating convergence.

    Args:
        J_history: History of cost function values
        successful_steps: List of booleans indicating step success
        rel_tolerance: Relative tolerance for convergence
        window_size: Number of iterations to consider

    Returns:
        bool: True if converged, False otherwise
    """
    if len(J_history) < window_size + 1:
        return False

    # Get the window of steps
    window_steps = successful_steps[-window_size:]
    if not any(window_steps):  # If no successful steps in window
        return False   # should be True!

    # Get indices of successful steps in the window
    successful_indices = [i for i, success in enumerate(window_steps) if success]
    if len(successful_indices) < 2:  # Need at least 2 successful steps to measure improvement
        return False

    # Get the corresponding J values for successful steps
    J_successful = [J_history[-(window_size - i)] for i in successful_indices]

    # Calculate improvements between consecutive successful steps
    improvements = np.diff(J_successful)
    rel_improvements = np.abs(improvements / np.maximum(np.abs(J_successful[:-1]), 1e-10))

    return float(np.mean(rel_improvements)) < rel_tolerance


def update_step_size(mu: float, J_new: float, J_old: float,
                    min_step: float = 1e-10, max_step: float = 1e5) -> float:
    """Update step size based on success of previous iteration."""
    alpha_plus = 1.5
    alpha_minus = 0.5
    print('Step ' + ('successful: increasing' if J_new <= J_old else 'failed: decreasing') + ' step size')

    relative_change = abs(J_new - J_old) / abs(J_old)
    if J_new <= J_old:
        if relative_change > 1e-3:
            return float(np.clip(mu * 2.0, min_step, max_step))  # Increased from 1.5 # TODO: choose alpha parameters
        return float(np.clip(mu * 1.8, min_step, max_step))  # Increased from 1.3
    if relative_change > 1e-2:
        return float(np.clip(mu * 0.5, min_step, max_step))  # Less aggressive reduction
    return float(np.clip(mu * 0.7, min_step, max_step))  # Less aggressive reduction



def compute_next_param_gradient(param: ArrayFloat, mu: float, J_prime: ArrayFloat, gradient_clip: Optional[float] = None) -> ArrayFloat:
    """Compute next parameter values using gradient descent step."""
    param_scale = norm(param)       # Get current scale of parameters
    grad_scale = norm(J_prime)      # Get current scale of gradient

    if grad_scale > 1e-10:
        # Normalize gradient to have unit norm
        normalized_gradient = J_prime / grad_scale

        # Limit step size relative to current parameter scale
        max_step = 0.1 * param_scale
        actual_step = min(mu, max_step)
        if actual_step < mu :
            print('step size 10% param scale')
        return param - actual_step * normalized_gradient


def projected_gradient_algorithm(meshing, offset, loading, discretization, material, cost_function, additive, max_iterations, tolerance, rel_tolerance=1e-8, previous_state=None, **kwargs):
    """Projected gradient algorithm"""
    overhang_max = 1.0
    gradient_tolerance = 1e-12

    cost_function_kwargs = {
        'meshing': meshing,
        'initial_offset': offset,
        'current_offset': offset.copy()  # Initialisation explicite
    }

    if previous_state is None:
        # Configuration initiale
        initialization = fixed_structure(meshing, loading, discretization, material, additive)
        U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(initialization,
            meshing, offset, loading, discretization, material, additive)

        # Update keys with shape_structure results
        cost_function_kwargs.update({
            'Y': Y,
            'Assemble': Assemble
        })

        # Initial evaluation
        param = offset2param(meshing, offset)
        Jval = cost_function(us, **cost_function_kwargs)
        p = compute_adjoint_state(yKy, cost_function, us, **cost_function_kwargs)
        J_prime = cost_function.gradient(us, p, offset, loading, discretization, material, **cost_function_kwargs)

        # Initialize step size
        initial_grad_norm = norm(J_prime)
        if initial_grad_norm > 1e-10:
            param_scale = norm(param)
            mu = 10 * param_scale / initial_grad_norm     # TODO: choose between 0.001 and 0.5
        else:
            mu = 1

        k = 0

        # Initialize history tracking
        param_values: List[ArrayFloat] = []
        mu_values: List[float] = []
        J_values: List[float] = []
        Jprime_values: List[ArrayFloat] = []
        G_values: List[ArrayFloat] = []

        successful_steps: List[bool] = []
    else:
        # Resume from previous state
        param = previous_state.param.copy()
        offset = previous_state.offset.copy()
        us = previous_state.us.copy()
        yKy = previous_state.yKy.copy()
        yfbc = previous_state.yfbc.copy()
        Jval = previous_state.Jval
        J_prime = previous_state.J_prime.copy()
        mu = previous_state.mu
        k = previous_state.iteration

        # Restore history
        param_values = previous_state.param_history.copy()
        mu_values = previous_state.mu_history.copy()
        J_values = previous_state.J_history.copy()
        Jprime_values = previous_state.Jprime_history.copy()
        G_values = previous_state.G_history.copy()

        successful_steps = []
        initialization = fixed_structure(meshing, loading, discretization, material, additive)
        U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(initialization,
            meshing, offset, loading, discretization, material, additive)

        # Update keys with shape_structure results
        cost_function_kwargs.update({
            'Y': Y,
            'Assemble': Assemble
        })

        print(f'Resuming optimization from iteration {k}')
        print(f'Previous cost value: {Jval:.6e}')

    # Start optimization loop
    tic = time.time()
    termination_reason = None

    # # Initial configuration  (comment out because already ran from previous_state setup)
    # U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(initialization,
    #             meshing, offset, loading, discretization, material, additive)

    # Initial evaluation
    param = offset2param(meshing, offset)
    Jval = cost_function(us, **cost_function_kwargs)  # Current cost value
    p = compute_adjoint_state(yKy, cost_function, us, **cost_function_kwargs)
    J_prime = cost_function.gradient(us, p, offset, loading, discretization, material, **cost_function_kwargs)

    print("\n=== Initial State ===")
    print(f"Initial cost: {Jval:.6e}")
    print(f"Initial gradient norm: {norm(J_prime):.6e}")
    print(f"Initial param norm: {norm(param):.6e}")

    while k < max_iterations:
        print(f"\n=== Iteration {k} ===")

        J_old = Jval  # Store current cost before update
        print(f"Current cost: {J_old:.6e}")
        print(f"Current gradient norm: {norm(J_prime):.6e}")
        print(f"Current step size (mu): {mu:.6e}")
        print(f"Current param norm: {norm(param):.6e}")

        # Store current state
        param_values.append(param.copy())
        mu_values.append(mu)
        J_values.append(Jval)
        Jprime_values.append(J_prime.copy())
        G_values.append(G(param, overhang_max))

        # Check termination conditions
        if Jval <= tolerance:
            termination_reason = TerminationReason.COST_TOLERANCE
            break

        if check_relative_convergence(J_values, successful_steps, rel_tolerance):
            termination_reason = TerminationReason.RELATIVE_IMPROVEMENT
            break

        if norm(J_prime) <= gradient_tolerance:
            termination_reason = TerminationReason.FLAT_GRADIENT
            break

        # Update parameters
        param_new = compute_next_param_gradient(param, mu, J_prime)

        # Project if constraints are violated
        if not np.all(G(param_new, overhang_max) < 0):
            print('Projecting onto constraints')
            param_new = project_onto_constraints_ortho(param_new, overhang_max)
        print(f"\nProposed step:")
        print(f"New param norm: {norm(param_new):.6e}")
        print(f"Param change: {norm(param_new - param):.6e}")

        # Update shape
        offset_new = param2offset(meshing, param_new)

        try:
            results_new = derivateOffset.shape_structure(initialization,
                meshing, offset, loading, discretization, material, additive)
            U_new, us_new, yKy_new, Assemble_new, Y_new, yfbc_new, X, Elems = results_new

            # Update cost function kwargs with new offset
            cost_function_kwargs.update({
                            'current_offset': offset_new,
                            'Y': Y_new,
                            'Assemble': Assemble_new
                        })
            # Evaluate new cost
            J_new = cost_function(us_new, **cost_function_kwargs)

            print(f"New cost (J_new): {J_new:.6e}")
            print(f"Cost change: {J_new - J_old:.6e}")

            # Track step success
            step_successful = J_new <= J_old
            successful_steps.append(step_successful)

            if step_successful:
                # Update state for next iteration
                Jval, param, offset = J_new, param_new, offset_new
                us, yKy, yfbc = us_new, yKy_new, yfbc_new

                # Update offset in cost function kwargs
                cost_function_kwargs['current_offset'] = offset  # Use current_offset

                # Compute new gradient
                p = compute_adjoint_state(yKy, cost_function, us, **cost_function_kwargs)
                J_prime = cost_function.gradient(us, p, offset, loading, discretization, material, **cost_function_kwargs)

        except Exception as e:
            print(f'Erreur à l\'itération {k}:')
            print(f'Type d\'erreur : {type(e).__name__}')
            print(f'Message d\'erreur : {str(e)}')

            # Restauration des paramètres précédents
            cost_function_kwargs.update({
                'current_offset': offset,
                'Y': Y,
                'Assemble': Assemble
            })

            step_successful = False
            successful_steps.append(step_successful)
            J_new = J_old

        mu = update_step_size(mu, J_new, J_old)

        k += 1

    # Set final termination reason if not already set
    if termination_reason is None:
        termination_reason = TerminationReason.MAX_ITERATIONS

    # Create final state
    final_state = OptimizationState(param=param, offset=offset, us=us, yKy=yKy, yfbc=yfbc, Jval=Jval, J_prime=J_prime, mu=mu, iteration=k, param_history=param_values, mu_history=mu_values, J_history=J_values, Jprime_history=Jprime_values,
        G_history=G_values)

    # Compute final metrics
    computation_time = time.time() - tic
    success_rate = np.mean(successful_steps) if successful_steps else 0.0

    print(f'Optimization completed:')
    print(f'  Final cost: {J_values[-1]:.6e}')
    print(f'  Computation time: {computation_time:.2f} seconds')
    print(f'  Termination reason: {termination_reason.value}')

    return {'param_values': param_values, 'J_values': J_values, 'Jprime_values': Jprime_values, 'mu_values': mu_values, 'G_values': G_values, 'termination_reason': termination_reason, 'iterations': k, 'final_cost': J_values[-1],
        'computation_time': computation_time, 'current_state': final_state, 'success_rate': success_rate}

###################
# Analyse results #
###################

def analyze_optimization_results(results: dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """Create comprehensive convergence analysis plots."""
    iterations = np.arange(len(results['J_values']))
    J_values = np.array(results['J_values'])

    rel_improvements = np.zeros_like(J_values)
    rel_improvements[1:] = (J_values[:-1] - J_values[1:]) / np.maximum(np.abs(J_values[:-1]), 1e-10) * 100

    gradient_norms = np.array([norm(jp) for jp in results['Jprime_values']])
    param_array = np.array(results['param_values'])

    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Cost function convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(iterations, J_values, 'b-', label='Cost Function')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost Value (log scale)')
    ax1.set_title('Cost Function Convergence')
    ax1.grid(True)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(iterations[1:], rel_improvements[1:], 'r--', alpha=0.5, label='Relative Improvement')
    ax1_twin.set_ylabel('Relative Improvement (%)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Gradient norm convergence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(iterations, gradient_norms, 'r-', label='Gradient Norm')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm (log scale)')
    ax2.set_title('Gradient Convergence')
    ax2.grid(True)
    ax2.legend()

    # Parameter evolution
    ax3 = fig.add_subplot(gs[1, :])
    num_params = param_array.shape[1]
    for i in range(num_params):
        ax3.plot(iterations, param_array[:, i], alpha=0.7, label=f'Param {i + 1}')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Parameter Evolution')
    if num_params <= 10:
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)

    # Step size evolution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.semilogy(iterations, results['mu_values'], 'g-', label='Step Size (μ)')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Step Size')
    ax4.set_title('Step Size Evolution')
    ax4.grid(True)
    ax4.legend()

    # Target geometry
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def plot_convergence_metrics(results: dict[str, Any], metrics: List[str], figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> plt.Figure:
    """Create custom convergence plots for specified metrics."""
    valid_metrics = {'cost': ('Cost Function', 'J_values', 'log'), 'gradient': ('Gradient Norm', 'Jprime_values', 'log'), 'step_size': ('Step Size', 'mu_values', 'log'),
        'parameters': ('Parameter Values', 'param_values', 'linear')}

    # Validate metrics
    for metric in metrics:
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Valid options are {list(valid_metrics.keys())}")

    n_plots = len(metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    iterations = np.arange(len(results['J_values']))

    for ax, metric in zip(axes, metrics):
        title, data_key, scale = valid_metrics[metric]

        if isinstance(data_key, list):
            # Handle multiple data series
            for key, label in zip(data_key, ['α⁺', 'α⁻']):
                ax.plot(iterations, results[key], label=label)
        else:
            if data_key == 'Jprime_values':
                values = [norm(jp) for jp in results[data_key]]
            elif data_key == 'param_values':
                param_array = np.array(results[data_key])
                for i in range(param_array.shape[1]):
                    ax.plot(iterations, param_array[:, i], label=f'Param {i + 1}', alpha=0.7)
            else:
                values = results[data_key]
                ax.plot(iterations, values)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(title)
        if scale == 'log':
            ax.set_yscale('log')
        ax.grid(True)
        if metric in ['alpha', 'parameters']:
            ax.legend()
        ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig

def save_optimization_report(results: dict[str, Any], filepath: str, include_plots: bool = True) -> None:
    """Generate and save a comprehensive optimization report."""
    import json
    from datetime import datetime

    # Convert NumPy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    report = {
        'timestamp': datetime.now().isoformat(),

        'convergence_metrics': {
            'initial_cost': float(results['J_values'][0]),
            'final_cost': float(results['J_values'][-1]),
            'cost_reduction': float((results['J_values'][0] - results['J_values'][-1]) / results['J_values'][0] * 100),
            'final_gradient_norm': float(norm(results['Jprime_values'][-1]))
        },
        'plotting_data': {
            'iterations': convert_to_serializable(np.arange(len(results['J_values']))),
            'cost_values': convert_to_serializable(results['J_values']),
            'parameter_values': convert_to_serializable(results['param_values'])
        },
        'optimization_results': {
            'final_cost': float(results['final_cost']),
            'iterations': int(results['iterations']),
            'computation_time': float(results['computation_time']),
            'termination_reason': results['termination_reason'].value,
            'success_rate': float(results['success_rate'])
        }
    }

    # Add parameter statistics
    param_array = np.array(results['param_values'])
    report['parameter_statistics'] = {
        'initial_params': convert_to_serializable(param_array[0]),
        'final_params': convert_to_serializable(param_array[-1]),
        'param_changes': convert_to_serializable(param_array[-1] - param_array[0])
    }

    def convert_numpy(obj):
        """Convert NumPy objects to JSON serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.float_)):
            return obj.item()
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif hasattr(obj, 'value'):  # For enum values
            return obj.value
        elif isinstance(obj, OptimizationState):
            return {
                'param': convert_numpy(obj.param),
                'offset': convert_numpy(obj.offset),
                'us': convert_numpy(obj.us),
                'yKy': convert_numpy(obj.yKy.toarray()),
                'yfbc': convert_numpy(obj.yfbc),
                'Jval': float(obj.Jval),
                'J_prime': convert_numpy(obj.J_prime),
                'mu': float(obj.mu),
                'iteration': int(obj.iteration)
            }
        return obj

    if include_plots:
        plot_dir = filepath.rsplit('.', 1)[0] + '_plots'
        os.makedirs(plot_dir, exist_ok=True)

        # Save plots
        fig = analyze_optimization_results(results)
        fig.savefig(os.path.join(plot_dir, 'convergence_analysis.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        metrics_fig = plot_convergence_metrics(results, ['cost', 'gradient', 'step_size'])
        metrics_fig.savefig(os.path.join(plot_dir, 'metrics.png'), bbox_inches='tight', dpi=300)
        plt.close(metrics_fig)

        report['plots'] = {
            'convergence_analysis': os.path.join(plot_dir, 'convergence_analysis.png'),
            'metrics': os.path.join(plot_dir, 'metrics.png')
        }

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)


################
# Plot results #
################
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Project colors
COLORS = {
    'primary': (0, 62/255, 92/255),     # bleu303
    'primary_pale': (0, 62/255, 92/255, 0.2),
    'secondary': (213/255, 43/255, 30/255),  # rouge485
    'secondary_pale': (213/255, 43/255, 30/255, 0.2),
    'tertiary': (0, 104/255, 128/255)   # bleu315
}

# Set global style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300
})


def plot_cost_convergence(iterations, J_values, Jprime_values):
    """Plot cost function convergence with relative improvements."""
    fig, ax1 = plt.subplots(figsize=(7, 3.5))  # 55% width, 50% height of plot_param_states

    # rel_improvements = np.zeros_like(J_values)
    # rel_improvements[1:] = (J_values[:-1] - J_values[1:]) / np.maximum(np.abs(J_values[:-1]), 1e-10) * 100
    gradient_norms = np.array([norm(jp) for jp in Jprime_values])

    # Cost function plot
    ax1.semilogy(iterations, J_values/J_values[0], '-', color=COLORS['secondary'],
                 label='Fonction coût normalisée', linewidth=1)
    ax1.set_xlabel('Itérations')
    ax1.set_ylabel('Fonction coût') # échelle log

    ax1.grid(True, linestyle=':')

    # Relative improvement plot
    ax2 = ax1.twinx()
    ax2.plot(iterations[1:], gradient_norms[1:]/gradient_norms[1], '--',
             color=COLORS['primary_pale'], alpha=0.5,
             #label='Amélioration relative', linewidth=0.5)
             label='Gradient normalisé', linewidth=0.5)
    ax2.set_ylabel('Evolution du gradient')#'Amélioration relative (%)')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig


def plot_parameter_evolution(iterations, param_array, difference=False):
    """Plot parameter evolution using color gradient."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))  # 50% height of plot_param_states

    num_params = param_array.shape[1]
    mix = 0.2
    color1 = np.array(COLORS['primary']) * (1-mix) + np.array([1, 1, 1]) * mix  # bleu303 with some white to make it lighter
    color2 = np.array(COLORS['tertiary']) * mix + np.array([0, 0, 0])  # bleu315 with some black to make it darker
    alpha_color = 0.5
    if difference :
        param_array -= param_array[0,:]

    linestyles = ['-', '--', '-.', ':']  # Repeated line styles for more distinction

    for i in range(num_params):
        alpha = i / (num_params - 1)
        color = color1 * (1 - alpha) + color2 * alpha

        linestyle = linestyles[i % len(linestyles)]
        ax.plot(iterations, param_array[:, i],
                color=tuple(np.append(color, 1-alpha_color*alpha)), # from 1 to 0.5 linearly
                #alpha=0.8,  # Add transparency
                linewidth=0.5+(1-alpha), # decrease from 1.5 to 0.5 linearly
                linestyle=linestyle,
                label=f'Paramètre {i + 1}')

    x_values = param_array[0, :][::2]
    ax.set_xlabel('Itérations')
    if difference :
        ax.set_ylabel(r'Evolution : $\delta_i^0-\delta_i^k$')
    else :
        plt.yticks(x_values, labels=[f'{x:.0f}' for x in x_values])
        ax.set_ylabel(r'Paramètres $\delta_i$')
    ax.grid(True, linestyle=':')
    if num_params <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig

def plotparam(param_values, ttl='', gap=5, difference=False):
   param_values_array = np.array(param_values)
   if difference:
       param_values_array -= param_values_array[0]
   nLayers_v = len(param_values_array[0])
   num_sets = len(param_values_array)

   color_firstLayer = COLORS['primary']
   color_lastLayer = COLORS['secondary']
   colors = [(1-t)*np.array(COLORS['primary_pale']) + t*np.array(COLORS['secondary_pale'])
            for t in np.linspace(0, 1, num_sets-2)]

   fig, ax = plt.subplots(figsize=(6, 8))

   # First set
   ax.plot(param_values_array[0], np.arange(nLayers_v),
       label='k=0', color=color_firstLayer, linewidth=1.0,
       linestyle='-', marker='o', zorder=4)

   # Intermediate sets
   for i, (row, color) in enumerate(zip(param_values_array[1:-1], colors), start=1):
       ax.plot(row, np.arange(nLayers_v),
           label=f'k={i}'*(i%gap==0), color=color,
           linestyle='--', marker='o', zorder=2)
       if not difference:
           for layer in range(nLayers_v):
               rect = Rectangle((row[layer]-0.5, layer-0.5), 1, 1,
                   linewidth=1.2, edgecolor=color, facecolor='none', zorder=0)
               ax.add_patch(rect)

   # Last set
   ax.plot(param_values_array[-1], range(nLayers_v),
       label=f'k={num_sets-1}', color=color_lastLayer,
       linewidth=1.0, linestyle='-', marker='o', zorder=3)

   if not difference:
       for layer in range(nLayers_v):
           for params, color in [(param_values_array[0], color_firstLayer),
                               (param_values_array[-1], color_lastLayer)]:
               rect = Rectangle((params[layer]-0.5, layer-0.5), 1, 1,
                   linewidth=1.2, edgecolor=color, facecolor='none', zorder=0)
               ax.add_patch(rect)

   ax.set_yticks(np.arange(nLayers_v))
   if not difference:
       x_values = np.concatenate([param_values_array[0]])[::2]
       plt.xticks(x_values, labels=[f'{x:.0f}' for x in x_values])

   ax.set_title(ttl)
   if difference:
       ax.set_xlabel(r'Ecarts aux $\delta_i^0$')
       ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
   else:
       ax.set_xlabel(r'Paramètres $\delta_i$')
       ax.legend(loc='upper right')
   ax.set_ylabel(r'Couches $i$')
   ax.grid(True, linestyle=':', alpha=0.6)

   plt.tight_layout()
   return fig


def plot_param_states(initial_param, final_param, param_def_initial, param_def_final, title='Parameter States Comparison', scale=10, difference=False):
    """Plot comparison of parameter states with clearer visibility."""
    nLayers_v = len(initial_param)

    # Create figure with specified size
    fig, ax = plt.subplots(figsize=(6, 8))

    # Define updated styles according to the graphic charter
    styles = {
        'target': {
            'color': COLORS['primary'],
            'marker': 'o',
            'linestyle': '-',
            'linewidth': 1.0,
            'label': 'Géométrie cible' if difference else 'État initial',
            'zorder': 4
        },
        'def_target': {
            'color': COLORS['primary_pale'],
            'marker': 's',
            'linestyle': '--',
            'linewidth': 1.0,
            'label': 'Déformée initiale' if difference else 'Déformée initiale',
            'zorder': 3
        },
        'optim': {
            'color': COLORS['secondary'],
            'marker': 'o',
            'linestyle': '-',
            'linewidth': 1.0,
            'label': 'Contre-déformée' if difference else 'État final',
            'zorder': 2
        },
        'def_optim': {
            'color': COLORS['secondary_pale'],
            'marker': 's',
            'linestyle': '--',
            'linewidth': 1.0,
            'label': 'Déformée finale' if difference else 'Déformée finale',
            'zorder': 1
        }
    }
    comparison = initial_param * int(difference)
    y_range = np.arange(nLayers_v)
    for state, params, style in [
        ('target', initial_param - comparison, styles['target']),
        ('def_target', initial_param + scale * param_def_initial - comparison, styles['def_target']),
        ('optim', final_param - comparison, styles['optim']),
        ('def_optim', final_param + scale * param_def_final - comparison, styles['def_optim'])
    ]:
        ax.plot(params, y_range, **style)

        if not difference :
            # Add rectangles for each state
            for layer in range(nLayers_v):
                rect = Rectangle((params[layer] - 0.5, layer - 0.5), 1, 1,
                                 linewidth=1.2,
                                 edgecolor=style['color'],
                                 facecolor='none',
                                 zorder=0)
                ax.add_patch(rect)

    ax.set_title(title)
    if difference :
        ax.set_xlabel(r'Ecarts aux $\delta_i^0$')
    else :
        ax.set_xlabel(r'Paramètres $\delta_i$')
    ax.set_ylabel(r'Couches $i$')
    ax.set_yticks(y_range)
    if not difference :
        x_values = np.concatenate([initial_param])[::2]
        plt.xticks(x_values, labels=[f'{x:.0f}' for x in x_values])
    ax.grid(True, linestyle=':', alpha=0.6)

    # Add legend inside top-right corner
    if difference :
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    else :
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig



# def plot_geometry_comparison(X_target, U0, U_final, Elems, meshing, scale=100, discretization=None):
#     """Plot 3D geometry comparison."""
#     L, Hn, Hb, *_ = meshing
#     elemOrder = discretization[0] if discretization else 1
#     scfplot = 10
#
#     # Create figure
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_box_aspect((1, scfplot, scfplot))
#
#     # Get basis vectors (copied from usePlotMesh)
#     from modules import mesh, fem
#     Xunc, uncElems = mesh.uncouple_nodes(X_target, Elems)
#     t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)
#     Me = sp.csr_matrix((np.ones(Elems.size), (Elems.flatten(), np.arange(Elems.size))))
#     nm = Me @ n
#     nm = nm / np.linalg.norm(nm, axis=1)[:, np.newaxis]
#     bm = Me @ b
#     bm = bm / np.linalg.norm(bm, axis=1)[:, np.newaxis]
#
#     # Target shape (x0 from usePlotMesh)
#     x0 = X_target[:, :, np.newaxis] + 0.5 * (
#         Hn * nm[:, :, np.newaxis] * np.array([[[-1, 1, -1, 1]]]) +
#         Hb * bm[:, :, np.newaxis] * np.array([[[1, 1, -1, -1]]])
#     )
#     srf1, _ = plot.plotMesh(ax, L, x0, Elems, color='royalblue', edgecolor='royalblue', outer=False)
#     srf1.set_alpha(0.3)
#
#     # Deformed shapes
#     x_init = x0 + scale * np.moveaxis(U0, (0, 1, 2), (1, 2, 0))
#     x_final = x0 + scale * np.moveaxis(U_final, (0, 1, 2), (1, 2, 0))
#
#     srf2, _ = plot.plotMesh(ax, L, x_init, Elems, color='lightblue', edgecolor='lightblue', outer=False)
#     srf2.set_alpha(0.3)
#
#     srf3, _ = plot.plotMesh(ax, L, x_final, Elems, color='lightgreen', edgecolor='lightgreen', outer=False)
#     srf3.set_alpha(0.3)
#
#     # Add legend
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='royalblue', alpha=0.3, label='Target'),
#         Patch(facecolor='lightblue', alpha=0.3, label='Initial Deformed'),
#         Patch(facecolor='lightgreen', alpha=0.3, label='Final Deformed')
#     ]
#     ax.legend(handles=legend_elements)
#
#     ax.set_xlabel('Axis t')
#     ax.set_ylabel('Axis n')
#     ax.set_zlabel('Axis b')
#
#     return fig
#
# def compare_geometries(meshing, loading, discretization, material, results,
#                       scale=100, save_path=None):
#     """
#     Compare optimized solutions in both parameter space and 3D geometry space.
#     """
#     # Get initial and final parameters
#     initial_param = results['param_values'][0]
#     final_param = results['param_values'][-1]
#
#     # Convert to offsets
#     initial_offset = param2offset(meshing, initial_param)
#     final_offset = param2offset(meshing, final_param)
#
#     # Generate target geometry and solutions
#     from shape.shapeOffset import generate_trajectory
#     X_target, Elems, U0 = generate_trajectory(meshing, initial_offset)
#
#     # Get mechanical solutions
#     initialization = fixed_structure(meshing, loading, discretization, material, additive)
#     U0, us0, yKy0, Assemble0, Y0, yfbc0, X0, _ = derivateOffset.shape_structure(initialization,
#             meshing, initial_offset, loading, discretization, material, additive)
#
#     U_final, us_final, yKy_final, Assemble_final, Y_final, yfbc_final, X_final, _ = derivateOffset.shape_structure(initialization,
#             meshing, final_offset, loading, discretization, material, additive)
#
#     param_def_initial = us2param(meshing, us0, Y0)
#     param_def_final = us2param(meshing, us_final, Y_final)
#
#     # Plot comparisons
#     plot_param_states(initial_param, final_param,
#                      param_def_initial, param_def_final,
#                      'Parameter States Comparison')
#
#     fig = plot_geometry_comparison(X_target, U0, U_final, Elems,
#                                  meshing, scale, discretization)
#
#     if save_path:
#         fig.savefig(save_path + '_geom.png', dpi=300, bbox_inches='tight')
#
#     # Calculate metrics
#     metrics = {
#         'param_metrics': {
#             'initial_diff_norm': float(np.linalg.norm(param_def_initial - initial_param)),
#             'final_diff_norm': float(np.linalg.norm(param_def_final - final_param)),
#             'improvement': float((1 - np.linalg.norm(param_def_final - final_param)/
#                                 np.linalg.norm(param_def_initial - initial_param)) * 100),
#         },
#         'geometry_metrics': {
#             'initial_max_displacement': float(np.max(np.abs(U0))),
#             'final_max_displacement': float(np.max(np.abs(U_final))),
#             'initial_error': float(np.linalg.norm(np.mean(U0, axis=1).T)),
#             'final_error': float(np.linalg.norm(np.mean(U_final, axis=1).T))
#         }
#     }
#
#     return metrics
#
# def analyze_layer_errors(X, final_offset, us_final, initial_offset, meshing, Y, Assemble,
#                         scale_factor=5, save_path=None):
#     """
#     Analyze errors layer by layer.
#
#     Parameters
#     ----------
#     Same as compare_geometries()
#
#     Returns
#     -------
#     fig : matplotlib figure
#         Layer analysis plots
#     metrics : dict
#         Layer-specific error metrics
#     """
#     L, Hn, Hb, nLayers_h, nLayers_v, nNodes, *_ = meshing
#
#     # Reshape displacements by layer
#     u_final = Y @ us_final
#     U_final = scale_factor * u_final.reshape((3, 4, -1))
#
#     # Initialize arrays for layer metrics
#     layer_errors = np.zeros(nLayers_v)
#     layer_improvements = np.zeros(nLayers_v)
#
#     # Create figure
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,12))
#
#     # Analyze each layer
#     for i in range(nLayers_v):
#         # Get layer indices
#         start_idx = i * nNodes * nLayers_h
#         end_idx = (i + 1) * nNodes * nLayers_h
#
#         # Extract layer geometry
#         X_layer = X[:,start_idx:end_idx]
#         U_layer = U_final[:,:,start_idx:end_idx]
#
#         # Calculate layer error
#         error = norm(U_layer - X_layer, axis=1)
#         layer_errors[i] = np.mean(error)
#
#         # Calculate improvement from initial
#         initial_error = norm(X_layer, axis=1)
#         layer_improvements[i] = (np.mean(initial_error) - np.mean(error))/np.mean(initial_error) * 100
#
#     # Plot layer errors
#     ax1.bar(np.arange(nLayers_v), layer_errors)
#     ax1.set_title('Error by Layer')
#     ax1.set_xlabel('Layer')
#     ax1.set_ylabel('Mean Error')
#
#     # Plot layer improvements
#     ax2.bar(np.arange(nLayers_v), layer_improvements)
#     ax2.set_title('Improvement by Layer')
#     ax2.set_xlabel('Layer')
#     ax2.set_ylabel('Percent Improvement')
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#
#     # Calculate layer-specific metrics
#     worst_layer = int(np.argmax(layer_errors))
#     best_layer = int(np.argmin(layer_errors))
#
#     metrics = {
#         'layer_errors': layer_errors,
#         'layer_improvements': layer_improvements,
#         'worst_layer': worst_layer,
#         'best_layer': best_layer,
#         'error_std': float(np.std(layer_errors)),
#         'improvement_std': float(np.std(layer_improvements))
#     }
#
#     return fig, metrics
#
# # %% Display
#
# def plot_layer_offsets_with_improvement(results, color_scheme=True):
#    initial_params = results['parameter_statistics']['initial_params']
#    final_params = results['parameter_statistics']['final_params']
#    cost_history = results['J_values']  # Assuming J_values is stored
#    layers = range(1, len(initial_params) + 1)
#
#    # Colors from project scheme
#    colors = {
#        'primary': '#003e5c',  # bleu303
#        'secondary': '#d52b1e', # rouge485
#        'tertiary': '#006880'  # bleu315
#    }
#
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
#
#    # Layer offsets
#    ax1.plot(initial_params, layers, 'o-', label='Initial', color=colors['primary'])
#    ax1.plot(final_params, layers, 'o-', label='Optimized', color=colors['secondary'])
#    ax1.set_ylabel('Layer Number')
#    ax1.set_xlabel('Offset')
#    ax1.grid(True, linestyle=':')
#    ax1.legend()
#    ax1.invert_yaxis()
#    ax1.set_ylim(len(initial_params) + 0.5, 0.5)
#
#    # Improvement rate
#    iterations = range(len(cost_history))
#    relative_improvement = [(cost_history[0] - cost)/cost_history[0] * 100
#                          for cost in cost_history]
#
#    ax2.plot(iterations, relative_improvement, '-', color=colors['tertiary'])
#    ax2.set_xlabel('Iteration')
#    ax2.set_ylabel('Improvement (%)')
#    ax2.grid(True, linestyle=':')
#
#    plt.tight_layout()
#    plt.show()

# Usage
import json

results_file = 'optimization_results.json'
def plot_optimization_results(results_file):
    """
    Plot optimization results with project color scheme.
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract data
    iterations = range(results['optimization_results']['iterations'])
    J_values = np.array(results['convergence_metrics']['cost_values'])
    param_array = np.array(results['parameter_statistics']['param_values'])

    # Project colors
    colors = {
        'primary': '#003e5c',    # bleu303
        'primary_pale': '#003e5c33',  # bleu303pale
        'secondary': '#d52b1e',  # rouge485
        'tertiary': '#006880'    # bleu315
    }

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1.5])

    # Cost function evolution
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(iterations, J_values, '-', color=colors['primary'],
                 linewidth=2, label='Cost Function')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost Value (log scale)')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    # Parameters evolution
    ax2 = fig.add_subplot(gs[1])
    layers = range(1, len(param_array[0]) + 1)

    # Plot initial and final states
    ax2.plot(param_array[0], layers, 'o-', color=colors['primary'],
             label='Initial Parameters', linewidth=2)
    ax2.plot(param_array[-1], layers, 'o-', color=colors['secondary'],
             label='Final Parameters', linewidth=2)

    # Add intermediate states with transparency
    num_intermediate = min(8, len(param_array)-2)  # Limit number of intermediates
    if num_intermediate > 0:
        step = (len(param_array)-2) // num_intermediate
        for i in range(1, len(param_array)-1, step):
            ax2.plot(param_array[i], layers, '-', color=colors['primary_pale'],
                    alpha=0.2, linewidth=1)

    ax2.set_ylabel('Layer Number')
    ax2.set_xlabel('Offset Parameter')
    ax2.grid(True, linestyle=':')
    ax2.legend()
    ax2.invert_yaxis()

    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    fig = plot_optimization_results("optimization_results.json")
    plt.show()