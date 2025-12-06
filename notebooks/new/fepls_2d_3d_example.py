# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: '1.18.1'
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
"""
Example script demonstrating FEPLS with 2D and 3D functional data

This script shows how to:
1. Create 2D/3D functional data from raw data
2. Apply FEPLS to multi-dimensional data
3. Visualize the results
"""

# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# we add utils to path
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(os.path.dirname(current_dir), 'utils')
sys.path.append(utils_dir)

from fepls_nd import fepls_nd, projection_nd

# %%
def create_2d_functional_data(n_samples: int = 100, d1: int = 20, d2: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    we create synthetic 2D functional data
    each sample is a 2D image/array (e.g., spatial data, spectrogram)
    
    Parameters:
    -----------
    n_samples : int
        number of samples
    d1, d2 : int
        spatial dimensions
    
    Returns:
    --------
    X : np.ndarray, shape (1, n_samples, d1, d2)
        functional covariates (2D arrays)
    Y : np.ndarray, shape (1, n_samples)
        scalar responses (heavy-tailed)
    """
    # we create synthetic 2D patterns
    X = np.zeros((1, n_samples, d1, d2))
    Y = np.zeros((1, n_samples))
    
    # we create a base pattern (e.g., a 2D Gaussian)
    x1 = np.linspace(-2, 2, d1)
    x2 = np.linspace(-2, 2, d2)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    base_pattern = np.exp(-(X1**2 + X2**2))
    
    for i in range(n_samples):
        # we add noise and variation
        noise = np.random.randn(d1, d2) * 0.1
        intensity = np.random.gamma(2, 2)  # heavy-tailed intensity
        X[0, i, :, :] = intensity * base_pattern + noise
        
        # we create heavy-tailed response
        # Y is related to the intensity of the pattern
        Y[0, i] = intensity + np.random.randn() * 0.5
    
    # we make Y heavy-tailed
    Y = np.abs(Y) + 0.1
    Y = Y ** (1 / 0.3)  # we create heavy tail with gamma ~ 0.3
    
    return X, Y


def create_3d_functional_data(n_samples: int = 100, d1: int = 10, d2: int = 10, d3: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    we create synthetic 3D functional data
    each sample is a 3D volume (e.g., spatio-temporal data, 3D image)
    
    Parameters:
    -----------
    n_samples : int
        number of samples
    d1, d2, d3 : int
        spatial dimensions
    
    Returns:
    --------
    X : np.ndarray, shape (1, n_samples, d1, d2, d3)
        functional covariates (3D arrays)
    Y : np.ndarray, shape (1, n_samples)
        scalar responses (heavy-tailed)
    """
    # we create synthetic 3D patterns
    X = np.zeros((1, n_samples, d1, d2, d3))
    Y = np.zeros((1, n_samples))
    
    # we create a base 3D pattern (e.g., a 3D Gaussian)
    x1 = np.linspace(-2, 2, d1)
    x2 = np.linspace(-2, 2, d2)
    x3 = np.linspace(-2, 2, d3)
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    base_pattern = np.exp(-(X1**2 + X2**2 + X3**2))
    
    for i in range(n_samples):
        # we add noise and variation
        noise = np.random.randn(d1, d2, d3) * 0.1
        intensity = np.random.gamma(2, 2)  # heavy-tailed intensity
        X[0, i, :, :, :] = intensity * base_pattern + noise
        
        # we create heavy-tailed response
        Y[0, i] = intensity + np.random.randn() * 0.5
    
    # we make Y heavy-tailed
    Y = np.abs(Y) + 0.1
    Y = Y ** (1 / 0.3)  # we create heavy tail with gamma ~ 0.3
    
    return X, Y


# %%
def example_2d_fepls():
    """we demonstrate FEPLS on 2D functional data"""
    print("=" * 60)
    print("Example 1: FEPLS on 2D Functional Data")
    print("=" * 60)
    
    # we create 2D data
    n_samples = 200
    d1, d2 = 20, 20
    X, Y = create_2d_functional_data(n_samples=n_samples, d1=d1, d2=d2)
    
    print(f"X shape: {X.shape} (N={X.shape[0]}, n={X.shape[1]}, d1={X.shape[2]}, d2={X.shape[3]})")
    print(f"Y shape: {Y.shape}")
    print(f"Y statistics: min={np.min(Y):.3f}, max={np.max(Y):.3f}, mean={np.mean(Y):.3f}")
    
    # we set up FEPLS parameters with two tau coefficients
    tau1 = -1.0  # tau for dimension d1
    tau2 = -0.5  # tau for dimension d2
    k = 20  # number of extremes
    Y_sorted = np.sort(Y[0, :])[::-1]
    y_threshold = Y_sorted[k]
    y_matrix = y_threshold * np.ones_like(Y)
    
    print(f"\nFEPLS parameters:")
    print(f"  tau1 = {tau1} (for dimension d1)")
    print(f"  tau2 = {tau2} (for dimension d2)")
    print(f"  k = {k} (using top {k} extremes)")
    print(f"  threshold = {y_threshold:.3f}")
    
    # we compute FEPLS direction with two tau
    print("\nComputing FEPLS direction with two tau coefficients...")
    beta_hat = fepls_nd(X, Y, y_matrix, (tau1, tau2))
    
    if beta_hat is not None:
        print(f"beta_hat shape: {beta_hat.shape}")
        print(f"beta_hat norm: {np.linalg.norm(beta_hat[0, :, :].flatten()):.6f}")
        print(f"beta_hat statistics: min={np.min(beta_hat):.3f}, max={np.max(beta_hat):.3f}, mean={np.mean(beta_hat):.3f}")
        
        # we visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # plot 1: example 2D sample
        im1 = axes[0].imshow(X[0, 0, :, :], cmap='viridis', aspect='auto')
        axes[0].set_title('Example 2D Sample')
        axes[0].set_xlabel('d2')
        axes[0].set_ylabel('d1')
        plt.colorbar(im1, ax=axes[0])
        
        # plot 2: FEPLS direction
        im2 = axes[1].imshow(beta_hat[0, :, :], cmap='RdBu_r', aspect='auto')
        axes[1].set_title('FEPLS Direction (2D)')
        axes[1].set_xlabel('d2')
        axes[1].set_ylabel('d1')
        plt.colorbar(im2, ax=axes[1])
        
        # plot 3: projections vs Y
        proj = projection_nd(X[0, :, :, :], beta_hat[0, :, :])
        axes[2].scatter(proj, Y[0, :], alpha=0.5, s=20)
        axes[2].set_xlabel('Projection <X, beta>')
        axes[2].set_ylabel('Y (Response)')
        axes[2].set_title('Projections vs Response')
        axes[2].grid(True, alpha=0.3)
        
        # we highlight extremes
        extreme_idx = Y[0, :] >= y_threshold
        axes[2].scatter(proj[extreme_idx], Y[0, extreme_idx], 
                       color='red', alpha=0.7, s=30, label='Extremes', zorder=5)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('fepls_2d_example.png', dpi=150, bbox_inches='tight')
        print("\nSaved visualization to 'fepls_2d_example.png'")
        plt.show()
        
        # we also compute with position-dependent tau
        print("\nComputing FEPLS with position-dependent tau...")
        beta_hat_pos = fepls_nd(X, Y, y_matrix, (tau1, tau2), position_dependent=True)
        if beta_hat_pos is not None:
            print(f"Position-dependent beta_hat norm: {np.linalg.norm(beta_hat_pos[0, :, :].flatten()):.6f}")
            # we compare the two approaches
            diff = np.linalg.norm(beta_hat[0, :, :] - beta_hat_pos[0, :, :])
            print(f"Difference between average and position-dependent: {diff:.6f}")
    else:
        print("ERROR: Could not compute FEPLS direction")


# %%
def example_3d_fepls():
    """we demonstrate FEPLS on 3D functional data"""
    print("\n" + "=" * 60)
    print("Example 2: FEPLS on 3D Functional Data")
    print("=" * 60)
    
    # we create 3D data
    n_samples = 150
    d1, d2, d3 = 10, 10, 10
    X, Y = create_3d_functional_data(n_samples=n_samples, d1=d1, d2=d2, d3=d3)
    
    print(f"X shape: {X.shape} (N={X.shape[0]}, n={X.shape[1]}, d1={X.shape[2]}, d2={X.shape[3]}, d3={X.shape[4]})")
    print(f"Y shape: {Y.shape}")
    print(f"Y statistics: min={np.min(Y):.3f}, max={np.max(Y):.3f}, mean={np.mean(Y):.3f}")
    
    # we set up FEPLS parameters with three tau coefficients
    tau1 = -1.0  # tau for dimension d1
    tau2 = -0.5  # tau for dimension d2
    tau3 = -0.8  # tau for dimension d3
    k = 15  # number of extremes
    Y_sorted = np.sort(Y[0, :])[::-1]
    y_threshold = Y_sorted[k]
    y_matrix = y_threshold * np.ones_like(Y)
    
    print(f"\nFEPLS parameters:")
    print(f"  tau1 = {tau1} (for dimension d1)")
    print(f"  tau2 = {tau2} (for dimension d2)")
    print(f"  tau3 = {tau3} (for dimension d3)")
    print(f"  k = {k} (using top {k} extremes)")
    print(f"  threshold = {y_threshold:.3f}")
    
    # we compute FEPLS direction with three tau
    print("\nComputing FEPLS direction with three tau coefficients...")
    beta_hat = fepls_nd(X, Y, y_matrix, (tau1, tau2, tau3))
    
    if beta_hat is not None:
        print(f"beta_hat shape: {beta_hat.shape}")
        print(f"beta_hat norm: {np.linalg.norm(beta_hat[0, :, :, :].flatten()):.6f}")
        print(f"beta_hat statistics: min={np.min(beta_hat):.3f}, max={np.max(beta_hat):.3f}, mean={np.mean(beta_hat):.3f}")
        
        # we visualize 3D data by showing slices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # we show middle slices of example sample
        mid_d1, mid_d2, mid_d3 = d1 // 2, d2 // 2, d3 // 2
        
        # example sample slices
        im1 = axes[0, 0].imshow(X[0, 0, mid_d1, :, :], cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Example Sample: slice d1={}'.format(mid_d1))
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(X[0, 0, :, mid_d2, :], cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Example Sample: slice d2={}'.format(mid_d2))
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(X[0, 0, :, :, mid_d3], cmap='viridis', aspect='auto')
        axes[0, 2].set_title('Example Sample: slice d3={}'.format(mid_d3))
        plt.colorbar(im3, ax=axes[0, 2])
        
        # FEPLS direction slices
        im4 = axes[1, 0].imshow(beta_hat[0, mid_d1, :, :], cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('FEPLS Direction: slice d1={}'.format(mid_d1))
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(beta_hat[0, :, mid_d2, :], cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('FEPLS Direction: slice d2={}'.format(mid_d2))
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(beta_hat[0, :, :, mid_d3], cmap='RdBu_r', aspect='auto')
        axes[1, 2].set_title('FEPLS Direction: slice d3={}'.format(mid_d3))
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('fepls_3d_example.png', dpi=150, bbox_inches='tight')
        print("\nSaved visualization to 'fepls_3d_example.png'")
        
        # we also plot projections
        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        proj = projection_nd(X[0, :, :, :, :], beta_hat[0, :, :, :])
        ax.scatter(proj, Y[0, :], alpha=0.5, s=20)
        extreme_idx = Y[0, :] >= y_threshold
        ax.scatter(proj[extreme_idx], Y[0, extreme_idx], 
                  color='red', alpha=0.7, s=30, label='Extremes', zorder=5)
        ax.set_xlabel('Projection <X, beta>')
        ax.set_ylabel('Y (Response)')
        ax.set_title('Projections vs Response (3D)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fepls_3d_projections.png', dpi=150, bbox_inches='tight')
        print("Saved projections plot to 'fepls_3d_projections.png'")
        plt.show()
    else:
        print("ERROR: Could not compute FEPLS direction")


# %%
if __name__ == "__main__":
    # we run examples
    example_2d_fepls()
    example_3d_fepls()
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

