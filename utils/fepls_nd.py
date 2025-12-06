"""
FEPLS (Functional Extreme Partial Least Squares) routines for multi-dimensional data
Supports 1D, 2D, and 3D functional covariates

The main function fepls_nd automatically detects the dimensionality and applies
the appropriate computation.
"""

import numpy as np
import numba
from typing import Optional, Tuple, Union


@numba.njit(parallel=True, fastmath=False)
def inner_product_nd(x: np.ndarray, y: np.ndarray) -> float:
    """
    we compute the inner product between two multi-dimensional arrays
    supports 1D, 2D, and 3D arrays
    """
    if x.ndim == 1:
        # 1D case: standard dot product
        return np.dot(x, y)
    elif x.ndim == 2:
        # 2D case: Frobenius inner product
        return np.sum(x * y)
    elif x.ndim == 3:
        # 3D case: sum over all dimensions
        return np.sum(x * y)
    else:
        # general case: flatten and dot
        return np.dot(x.flatten(), y.flatten())


@numba.njit(parallel=True, fastmath=False)
def norm_nd(x: np.ndarray) -> float:
    """
    we compute the norm of a multi-dimensional array
    supports 1D, 2D, and 3D arrays
    """
    if x.ndim == 1:
        return np.linalg.norm(x)
    elif x.ndim == 2:
        # Frobenius norm
        return np.sqrt(np.sum(x * x))
    elif x.ndim == 3:
        # 3D norm: sqrt of sum of squares
        return np.sqrt(np.sum(x * x))
    else:
        # general case: flatten and norm
        return np.linalg.norm(x.flatten())


@numba.njit(parallel=True, fastmath=False)
def normalize_nd(x: np.ndarray) -> np.ndarray:
    """
    we normalize a multi-dimensional array to unit norm
    returns a copy of x normalized
    """
    n = norm_nd(x)
    if n < 1e-10:
        # we return zero array if norm is too small
        return np.zeros_like(x)
    if x.ndim == 1:
        return x / n
    elif x.ndim == 2:
        return x / n
    elif x.ndim == 3:
        return x / n
    else:
        return x.flatten() / n


@numba.njit(parallel=True, fastmath=False)
def fepls_1d(X: np.ndarray, Y: np.ndarray, y_matrix: np.ndarray, tau: float) -> np.ndarray:
    """
    we compute FEPLS direction for 1D functional data
    X: shape (N, n, d) where N=batch, n=samples, d=dimension
    Y: shape (N, n)
    y_matrix: shape (N, n)
    returns: shape (N, d)
    """
    N = X.shape[0]
    n = X.shape[1]
    d = X.shape[2]
    out = np.zeros((N, d))
    
    for j in numba.prange(d):
        aux = np.multiply(X[:, :, j], Y ** tau)
        out2 = np.multiply(aux, np.greater_equal(Y, y_matrix))
        out[:, j] = np.sum(out2, axis=1) / n
    
    norms = np.sqrt(np.sum(out ** 2, axis=1) / d)
    
    # we fix division by zero
    for i in numba.prange(N):
        if norms[i] < 1e-10:
            norms[i] = 1.0
    
    out2 = out * (norms.reshape((norms.size, 1))) ** (-1)
    return out2


@numba.njit(parallel=True, fastmath=False)
def fepls_2d_separate(X: np.ndarray, Y: np.ndarray, y_matrix: np.ndarray, tau1: float, tau2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    we compute TWO separate FEPLS directions for 2D functional data
    beta1 for dimension d1 (using tau1), beta2 for dimension d2 (using tau2)
    
    X: shape (N, n, d1, d2) where N=batch, n=samples, d1,d2=spatial dimensions
    Y: shape (N, n)
    y_matrix: shape (N, n)
    tau1: test function parameter for dimension d1 (used to compute beta1)
    tau2: test function parameter for dimension d2 (used to compute beta2)
    returns: (beta1, beta2) where beta1 has shape (N, d1) and beta2 has shape (N, d2)
    """
    N = X.shape[0]
    n = X.shape[1]
    d1 = X.shape[2]
    d2 = X.shape[3]
    
    beta1 = np.zeros((N, d1))
    beta2 = np.zeros((N, d2))
    
    # we compute beta1: marginalize over d2, use tau1
    for i in numba.prange(N):
        for j1 in range(d1):
            # we sum over d2 dimension
            aux = np.zeros(n)
            for j2 in range(d2):
                aux += X[i, :, j1, j2]
            # we apply tau1 weighting
            aux_weighted = np.multiply(aux, Y[i, :] ** tau1)
            aux_filtered = np.multiply(aux_weighted, np.greater_equal(Y[i, :], y_matrix[i, :]))
            beta1[i, j1] = np.sum(aux_filtered) / (n * d2)
    
    # we compute beta2: marginalize over d1, use tau2
    for i in numba.prange(N):
        for j2 in range(d2):
            # we sum over d1 dimension
            aux = np.zeros(n)
            for j1 in range(d1):
                aux += X[i, :, j1, j2]
            # we apply tau2 weighting
            aux_weighted = np.multiply(aux, Y[i, :] ** tau2)
            aux_filtered = np.multiply(aux_weighted, np.greater_equal(Y[i, :], y_matrix[i, :]))
            beta2[i, j2] = np.sum(aux_filtered) / (n * d1)
    
    # we normalize separately
    for i in numba.prange(N):
        norm1 = np.linalg.norm(beta1[i, :])
        if norm1 < 1e-10:
            norm1 = 1.0
        beta1[i, :] = beta1[i, :] / norm1
        
        norm2 = np.linalg.norm(beta2[i, :])
        if norm2 < 1e-10:
            norm2 = 1.0
        beta2[i, :] = beta2[i, :] / norm2
    
    return beta1, beta2


@numba.njit(parallel=True, fastmath=False)
def fepls_2d(X: np.ndarray, Y: np.ndarray, y_matrix: np.ndarray, tau1: float, tau2: float, position_dependent: bool = False) -> np.ndarray:
    """
    we compute FEPLS direction for 2D functional data with two tau coefficients
    returns a SINGLE 2D direction (matrix d1 x d2)
    
    X: shape (N, n, d1, d2) where N=batch, n=samples, d1,d2=spatial dimensions
    Y: shape (N, n)
    y_matrix: shape (N, n)
    tau1: test function parameter for dimension d1
    tau2: test function parameter for dimension d2
    position_dependent: if True, tau varies with position; if False, uses average tau
    returns: shape (N, d1, d2) - single 2D direction
    """
    N = X.shape[0]
    n = X.shape[1]
    d1 = X.shape[2]
    d2 = X.shape[3]
    out = np.zeros((N, d1, d2))
    
    # we compute weighted sum over samples with dimension-specific tau
    for i in numba.prange(N):
        for j1 in range(d1):
            for j2 in range(d2):
                if position_dependent:
                    # we use position-dependent weighting: interpolate between tau1 and tau2
                    # normalized position: 0 to 1
                    pos1 = j1 / max(1, d1 - 1) if d1 > 1 else 0.0
                    pos2 = j2 / max(1, d2 - 1) if d2 > 1 else 0.0
                    # we weight tau1 by position in d1, tau2 by position in d2
                    tau_combined = tau1 * pos1 + tau2 * pos2
                else:
                    # we use average tau
                    tau_combined = (tau1 + tau2) / 2.0
                aux = np.multiply(X[i, :, j1, j2], Y[i, :] ** tau_combined)
                out2 = np.multiply(aux, np.greater_equal(Y[i, :], y_matrix[i, :]))
                out[i, j1, j2] = np.sum(out2) / n
    
    # we normalize
    for i in numba.prange(N):
        norm = norm_nd(out[i, :, :])
        if norm < 1e-10:
            norm = 1.0
        out[i, :, :] = out[i, :, :] / norm
    
    return out


@numba.njit(parallel=True, fastmath=False)
def fepls_3d(X: np.ndarray, Y: np.ndarray, y_matrix: np.ndarray, tau1: float, tau2: float, tau3: float, position_dependent: bool = False) -> np.ndarray:
    """
    we compute FEPLS direction for 3D functional data with three tau coefficients
    X: shape (N, n, d1, d2, d3) where N=batch, n=samples, d1,d2,d3=spatial dimensions
    Y: shape (N, n)
    y_matrix: shape (N, n)
    tau1: test function parameter for dimension d1
    tau2: test function parameter for dimension d2
    tau3: test function parameter for dimension d3
    position_dependent: if True, tau varies with position; if False, uses average tau
    returns: shape (N, d1, d2, d3)
    """
    N = X.shape[0]
    n = X.shape[1]
    d1 = X.shape[2]
    d2 = X.shape[3]
    d3 = X.shape[4]
    out = np.zeros((N, d1, d2, d3))
    
    # we compute weighted sum over samples with dimension-specific tau
    for i in numba.prange(N):
        for j1 in range(d1):
            for j2 in range(d2):
                for j3 in range(d3):
                    if position_dependent:
                        # we use position-dependent weighting
                        pos1 = j1 / max(1, d1 - 1) if d1 > 1 else 0.0
                        pos2 = j2 / max(1, d2 - 1) if d2 > 1 else 0.0
                        pos3 = j3 / max(1, d3 - 1) if d3 > 1 else 0.0
                        # we weight each tau by its corresponding position
                        tau_combined = (tau1 * pos1 + tau2 * pos2 + tau3 * pos3) / 3.0
                    else:
                        # we use average tau
                        tau_combined = (tau1 + tau2 + tau3) / 3.0
                    aux = np.multiply(X[i, :, j1, j2, j3], Y[i, :] ** tau_combined)
                    out2 = np.multiply(aux, np.greater_equal(Y[i, :], y_matrix[i, :]))
                    out[i, j1, j2, j3] = np.sum(out2) / n
    
    # we normalize
    for i in numba.prange(N):
        norm = norm_nd(out[i, :, :, :])
        if norm < 1e-10:
            norm = 1.0
        out[i, :, :, :] = out[i, :, :, :] / norm
    
    return out


def fepls_2d_combined(beta1: np.ndarray, beta2: np.ndarray) -> np.ndarray:
    """
    we combine two separate 1D directions into a single 2D direction
    beta1: shape (N, d1) or (d1,)
    beta2: shape (N, d2) or (d2,)
    returns: shape (N, d1, d2) or (d1, d2) - outer product beta1 @ beta2^T
    """
    if beta1.ndim == 1 and beta2.ndim == 1:
        # single batch case
        return np.outer(beta1, beta2)
    elif beta1.ndim == 2 and beta2.ndim == 2:
        # multiple batches case
        N = beta1.shape[0]
        d1 = beta1.shape[1]
        d2 = beta2.shape[1]
        out = np.zeros((N, d1, d2))
        for i in range(N):
            out[i, :, :] = np.outer(beta1[i, :], beta2[i, :])
        return out
    else:
        raise ValueError(f"Incompatible shapes: beta1 {beta1.shape}, beta2 {beta2.shape}")


def fepls_nd(X: np.ndarray, Y: np.ndarray, y_matrix: np.ndarray, tau, position_dependent: bool = False, separate_directions: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    we compute FEPLS direction for multi-dimensional functional data
    automatically detects dimensionality and applies appropriate computation
    
    Parameters:
    -----------
    X : np.ndarray
        functional covariates
        - 1D: shape (N, n, d)
        - 2D: shape (N, n, d1, d2)
        - 3D: shape (N, n, d1, d2, d3)
        where N=batch size, n=number of samples, d/d1/d2/d3=spatial dimensions
    Y : np.ndarray
        scalar responses, shape (N, n)
    y_matrix : np.ndarray
        threshold matrix, shape (N, n)
    tau : float or tuple/list
        test function parameter(s) (regular variation index)
        - 1D: single float
        - 2D: tuple/list of 2 floats (tau1, tau2) or single float (used for both)
        - 3D: tuple/list of 3 floats (tau1, tau2, tau3) or single float (used for all)
    position_dependent : bool, default=False
        if True, tau varies with spatial position (only for 2D/3D)
        if False, uses average of tau values
    
    Returns:
    --------
    beta_hat : np.ndarray
        estimated FEPLS direction
        - 1D: shape (N, d)
        - 2D: shape (N, d1, d2)
        - 3D: shape (N, d1, d2, d3)
    """
    if X.ndim < 3:
        raise ValueError(f"X must have at least 3 dimensions, got {X.ndim}")
    
    # we detect dimensionality and parse tau
    if X.ndim == 3:
        # 1D case: (N, n, d)
        if isinstance(tau, (tuple, list)):
            tau = tau[0]  # we take first element if tuple/list provided
        return fepls_1d(X, Y, y_matrix, float(tau))
    elif X.ndim == 4:
        # 2D case: (N, n, d1, d2)
        if isinstance(tau, (tuple, list)):
            if len(tau) >= 2:
                tau1, tau2 = float(tau[0]), float(tau[1])
            else:
                tau1 = tau2 = float(tau[0])
        else:
            tau1 = tau2 = float(tau)
        
        if separate_directions:
            # we return two separate directions (beta1, beta2)
            return fepls_2d_separate(X, Y, y_matrix, tau1, tau2)
        else:
            # we return a single 2D direction (matrix d1 x d2)
            return fepls_2d(X, Y, y_matrix, tau1, tau2, position_dependent)
    elif X.ndim == 5:
        # 3D case: (N, n, d1, d2, d3)
        if isinstance(tau, (tuple, list)):
            if len(tau) >= 3:
                tau1, tau2, tau3 = float(tau[0]), float(tau[1]), float(tau[2])
            elif len(tau) == 2:
                tau1, tau2 = float(tau[0]), float(tau[1])
                tau3 = (tau1 + tau2) / 2.0  # we average if only 2 provided
            else:
                tau1 = tau2 = tau3 = float(tau[0])
        else:
            tau1 = tau2 = tau3 = float(tau)
        return fepls_3d(X, Y, y_matrix, tau1, tau2, tau3, position_dependent)
    else:
        raise ValueError(f"X has {X.ndim} dimensions, but only 1D, 2D, and 3D are supported")


@numba.njit(parallel=True, fastmath=False)
def projection_1d(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    we compute projection <X, beta> for 1D case
    X: shape (n, d)
    beta: shape (d,)
    returns: shape (n,)
    """
    n = X.shape[0]
    d = X.shape[1]
    out = np.zeros(n)
    for i in numba.prange(n):
        out[i] = np.dot(X[i, :], beta) / d
    return out


@numba.njit(parallel=True, fastmath=False)
def projection_2d_separate(X: np.ndarray, beta1: np.ndarray, beta2: np.ndarray) -> np.ndarray:
    """
    we compute projection <X, beta1 ⊗ beta2> for 2D case with two separate directions
    X: shape (n, d1, d2)
    beta1: shape (d1,)
    beta2: shape (d2,)
    returns: shape (n,) - projection using tensor product beta1 ⊗ beta2
    """
    n = X.shape[0]
    d1 = X.shape[1]
    d2 = X.shape[2]
    out = np.zeros(n)
    for i in numba.prange(n):
        # we compute <X[i], beta1 ⊗ beta2> = sum_{j1, j2} X[i, j1, j2] * beta1[j1] * beta2[j2]
        for j1 in range(d1):
            for j2 in range(d2):
                out[i] += X[i, j1, j2] * beta1[j1] * beta2[j2]
        out[i] = out[i] / (d1 * d2)
    return out


@numba.njit(parallel=True, fastmath=False)
def projection_2d(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    we compute projection <X, beta> for 2D case
    X: shape (n, d1, d2)
    beta: shape (d1, d2)
    returns: shape (n,)
    """
    n = X.shape[0]
    d1 = X.shape[1]
    d2 = X.shape[2]
    out = np.zeros(n)
    for i in numba.prange(n):
        out[i] = np.sum(X[i, :, :] * beta) / (d1 * d2)
    return out


@numba.njit(parallel=True, fastmath=False)
def projection_3d(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    we compute projection <X, beta> for 3D case
    X: shape (n, d1, d2, d3)
    beta: shape (d1, d2, d3)
    returns: shape (n,)
    """
    n = X.shape[0]
    d1 = X.shape[1]
    d2 = X.shape[2]
    d3 = X.shape[3]
    out = np.zeros(n)
    for i in numba.prange(n):
        out[i] = np.sum(X[i, :, :, :] * beta) / (d1 * d2 * d3)
    return out


def projection_nd(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    we compute projection <X, beta> for multi-dimensional case
    automatically detects dimensionality
    
    Parameters:
    -----------
    X : np.ndarray
        functional covariates
        - 1D: shape (n, d)
        - 2D: shape (n, d1, d2)
        - 3D: shape (n, d1, d2, d3)
    beta : np.ndarray
        FEPLS direction
        - 1D: shape (d,)
        - 2D: shape (d1, d2)
        - 3D: shape (d1, d2, d3)
    
    Returns:
    --------
    proj : np.ndarray
        projections, shape (n,)
    """
    if X.ndim == 2 and beta.ndim == 1:
        # 1D case
        return projection_1d(X, beta)
    elif X.ndim == 3 and beta.ndim == 2:
        # 2D case
        return projection_2d(X, beta)
    elif X.ndim == 4 and beta.ndim == 3:
        # 3D case
        return projection_3d(X, beta)
    else:
        # general case: flatten and compute
        X_flat = X.reshape(X.shape[0], -1)
        beta_flat = beta.flatten()
        d = beta_flat.shape[0]
        return np.dot(X_flat, beta_flat) / d


# we provide backward compatibility wrapper
def fepls(X: np.ndarray, Y: np.ndarray, y_matrix: np.ndarray, tau) -> Optional[np.ndarray]:
    """
    backward compatibility wrapper for fepls_nd
    accepts single tau (float) or multiple tau (tuple/list) for multi-dimensional data
    """
    return fepls_nd(X, Y, y_matrix, tau)

