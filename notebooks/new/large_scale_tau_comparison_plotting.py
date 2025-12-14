# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
"""
Large-scale tau comparison with hypothesis verification and plotting
Tests multiple tau values, verifies hypotheses, and creates comparative plots
"""

# %%
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import List, Tuple, Dict, Optional
import numba

# we add current directory to path to import from adapted.py and large_scale_hypothesisverif.py
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
sys.path.append(current_dir)

# %%
# we import FEPLS functions from adapted.py
try:
    import adapted
    fepls_original = adapted.fepls
    bitcoin_concomittant_corr_original = adapted.bitcoin_concomittant_corr
    get_hill_estimator = adapted.get_hill_estimator
    Exponential_QQ_Plot_1D = adapted.Exponential_QQ_Plot_1D
    plot_quantile_conditional_on_sample_new = adapted.plot_quantile_conditional_on_sample_new
except ImportError:
    # we try alternative import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("adapted", os.path.join(current_dir, "adapted.py"))
    adapted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapted)
    fepls_original = adapted.fepls
    bitcoin_concomittant_corr_original = adapted.bitcoin_concomittant_corr
    get_hill_estimator = adapted.get_hill_estimator
    Exponential_QQ_Plot_1D = adapted.Exponential_QQ_Plot_1D
    plot_quantile_conditional_on_sample_new = adapted.plot_quantile_conditional_on_sample_new

# we create safe wrappers that handle negative Y values and negative tau
def compute_beta_with_tau(X_data, Y_data, tau, y_matrix):
    """we compute beta using fepls (1D) for given tau"""
    X_fepls = np.expand_dims(X_data, axis=0)  # shape (1, n, d)
    Y_fepls = np.expand_dims(Y_data, axis=0)  # shape (1, n)
    
    Y_abs = np.abs(Y_fepls)
    y_matrix_abs = np.expand_dims(y_matrix, axis=0) if y_matrix.ndim == 1 else y_matrix
    y_matrix_abs = np.abs(y_matrix_abs)
    epsilon = 1e-8 if tau >= 0 else 1e-6
    Y_abs = np.maximum(Y_abs, epsilon)
    y_matrix_abs = np.maximum(y_matrix_abs, epsilon)
    
    beta = fepls_safe(X_fepls, Y_abs, y_matrix_abs, tau)
    if beta is not None:
        beta = beta[0, :]  # shape (d,)
        # we normalize
        norm = np.linalg.norm(beta)
        if norm > 1e-10:
            beta = beta / norm
        return beta
    return None

def fepls_safe(X, Y, y_matrix, tau):
    """we wrap fepls to handle negative Y values and negative tau by using absolute value"""
    # we use absolute value of Y to avoid NaN with negative exponents
    # this is safe because we're interested in the magnitude of extremes
    Y_abs = np.abs(Y)
    y_matrix_abs = np.abs(y_matrix)
    # we ensure Y_abs is strictly positive (add small epsilon to avoid zeros)
    # when tau is negative, zeros would cause Inf, so we need a larger epsilon
    # for tau=-3, epsilon=1e-6 gives (1e-6)**(-3) = 1e18 which can cause overflow
    # we use epsilon that ensures (epsilon)**(-abs(tau)) < 1e10 to avoid overflow
    if tau < 0:
        # we want epsilon such that epsilon**(-abs(tau)) < 1e10
        # epsilon > (1e-10)**(1/abs(tau))
        max_power = max(abs(tau), 1.0)
        epsilon = max(1e-6, (1e-10)**(1.0/max_power))
    else:
        epsilon = 1e-8
    Y_abs = np.maximum(Y_abs, epsilon)
    y_matrix_abs = np.maximum(y_matrix_abs, epsilon)
    
    try:
        result = fepls_original(X, Y_abs, y_matrix_abs, tau)
        # we check for NaN/Inf and try to fix
        if result is not None:
            # we check for NaN/Inf before and after
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                # we renormalize if needed
                for i in range(result.shape[0]):
                    norm = np.linalg.norm(result[i, :])
                    if norm > 1e-10:
                        result[i, :] = result[i, :] / norm
                    else:
                        # if norm is too small, we set to uniform vector
                        result[i, :] = np.ones(result.shape[1]) / np.sqrt(result.shape[1])
            # we also check for suspiciously large values (potential overflow)
            if np.any(np.abs(result) > 1e10):
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                # we renormalize
                for i in range(result.shape[0]):
                    norm = np.linalg.norm(result[i, :])
                    if norm > 1e-10:
                        result[i, :] = result[i, :] / norm
                    else:
                        result[i, :] = np.ones(result.shape[1]) / np.sqrt(result.shape[1])
        return result
    except Exception as e:
        # we try with original Y if it's all positive
        if np.all(Y >= 0):
            try:
                # we still need to handle zeros for negative tau
                Y_safe = np.maximum(Y, epsilon)
                y_matrix_safe = np.maximum(y_matrix, epsilon)
                result = fepls_original(X, Y_safe, y_matrix_safe, tau)
                if result is not None:
                    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                return result
            except Exception:
                return None
        return None

def bitcoin_concomittant_corr_safe(X, Y, tau, m):
    """we wrap bitcoin_concomittant_corr to handle negative Y values and negative tau"""
    # we need to temporarily replace fepls with fepls_safe inside bitcoin_concomittant_corr
    # because bitcoin_concomittant_corr calls fepls internally, and we need to protect against NaN
    original_fepls = adapted.fepls  # we save the original fepls from adapted module
    
    Y_abs = np.abs(Y)
    # we use a larger epsilon for negative tau to avoid overflow/NaN
    # for tau=-3, epsilon=1e-6 gives (1e-6)**(-3) = 1e18 which can cause overflow
    # we use epsilon that ensures (epsilon)**(-abs(tau)) < 1e10 to avoid overflow
    if tau < 0:
        # we want epsilon such that epsilon**(-abs(tau)) < 1e10
        # epsilon > (1e-10)**(1/abs(tau))
        max_power = max(abs(tau), 1.0)
        epsilon = max(1e-6, (1e-10)**(1.0/max_power))
    else:
        epsilon = 1e-8
    Y_abs = np.maximum(Y_abs, epsilon)
    
    # we temporarily replace fepls in adapted module with fepls_safe
    # so that bitcoin_concomittant_corr uses the safe version that handles NaN
    adapted.fepls = fepls_safe
    
    try:
        result = bitcoin_concomittant_corr_original(X, Y_abs, tau, m)
        # we check for NaN/Inf more carefully
        if result is not None:
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            # we also check if any values are suspiciously large (potential overflow)
            if np.any(np.abs(result) > 1e10):
                print(f"    WARNING: correlation curve contains very large values (potential overflow) for tau={tau:.2f}")
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            result = np.zeros(m)
        
        if np.all(result == 0.0):
            if np.all(Y >= 0):
                try:
                    Y_safe = np.maximum(Y, epsilon)
                    result = bitcoin_concomittant_corr_original(X, Y_safe, tau, m)
                    if result is not None:
                        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        result = np.zeros(m)
                except Exception:
                    result = np.zeros(m)
    except Exception as e:
        print(f"    WARNING: Exception in bitcoin_concomittant_corr for tau={tau:.2f}: {e}")
        if np.all(Y >= 0):
            try:
                Y_safe = np.maximum(Y, epsilon)
                result = bitcoin_concomittant_corr_original(X, Y_safe, tau, m)
                if result is not None:
                    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    result = np.zeros(m)
            except Exception:
                result = np.zeros(m)
        else:
            result = np.zeros(m)
    finally:
        # we restore the original fepls in adapted module
        adapted.fepls = original_fepls
    
    return result


# we use the safe versions
fepls = fepls_safe
bitcoin_concomittant_corr = bitcoin_concomittant_corr_safe

# we import hypothesis verification functions from large_scale_hypothesisverif.py
try:
    from large_scale_hypothesisverif import (
        load_stooq_file, create_functional_data, build_functional_dataset,
        compute_fepls_direction, estimate_gamma, estimate_kappa
    )
except ImportError:
    # we try alternative import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("large_scale_hypothesisverif", 
                                                   os.path.join(current_dir, "large_scale_hypothesisverif.py"))
    large_scale_hypothesisverif = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(large_scale_hypothesisverif)
    load_stooq_file = large_scale_hypothesisverif.load_stooq_file
    create_functional_data = large_scale_hypothesisverif.create_functional_data
    build_functional_dataset = large_scale_hypothesisverif.build_functional_dataset
    compute_fepls_direction = large_scale_hypothesisverif.compute_fepls_direction
    estimate_gamma = large_scale_hypothesisverif.estimate_gamma
    estimate_kappa = large_scale_hypothesisverif.estimate_kappa

# %% [markdown]
# # Configuration

# %%

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "tau_comparison_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

# we define tau values to test (single values, not tuples)
TAU_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
TAU_GRID = TAU_VALUES

# we set random seed
np.random.seed(42)


# %% [markdown]
# # Helper functions for plotting

# %%
def plot_single_tau_analysis(
    X_fepls: np.ndarray,
    Y_fepls: np.ndarray,
    beta: np.ndarray,
    tau: float,
    best_k: int,
    gamma_hat: float,
    kappa_hat: float,
    hypothesis_value: float,
    hypothesis_valid: bool,
    pair_name: str,
    save_path: str,
    corr_curve: np.ndarray = None,  # we can pass precomputed corr_curve
) -> None:
    """we create a comprehensive plot for a single tau value with one beta direction"""
    n_samples = Y_fepls.shape[1]
    d = beta.shape[0]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # we compute correlation curve for this tau if not provided
    m_threshold = int(n_samples / 5)
    if corr_curve is None:
        # we recalculate corr_curve with the correct tau to ensure consistency
        corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau, m_threshold)
        # we check for NaN/Inf in correlation curve and replace them
        if np.any(np.isnan(corr_curve)) or np.any(np.isinf(corr_curve)):
            corr_curve = np.nan_to_num(corr_curve, nan=0.0, posinf=0.0, neginf=0.0)
    
    Y_sorted = np.sort(Y_fepls[0])[::-1]
    
    # Plot 1: Correlation curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(corr_curve, 'b-', linewidth=2)
    ax1.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best k={best_k}')
    ax1.set_title(f'Tail Correlation vs k (tau={tau:.1f})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Exceedances (k)')
    ax1.set_ylabel('Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hill Plot
    ax2 = fig.add_subplot(gs[0, 1])
    hill_est = get_hill_estimator(Y_sorted)
    ax2.plot(hill_est, 'g-', linewidth=2)
    ax2.axhline(y=gamma_hat, color='r', linestyle='--', linewidth=2, label=f'gamma_hat={gamma_hat:.4f}')
    ax2.set_title(f'Hill Plot (Tail Index)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Gamma')
    ax2.set_xlim(10, m_threshold)
    ax2.set_ylim(0, max(1.0, gamma_hat * 2))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Exponential QQ Plot
    ax3 = fig.add_subplot(gs[0, 2])
    qq_data = Exponential_QQ_Plot_1D(Y_fepls, best_k)
    if len(qq_data) > 1:
        slope, intercept, r_val, _, _ = linregress(qq_data[:, 0], qq_data[:, 1])
        ax3.scatter(qq_data[:, 0], qq_data[:, 1], alpha=0.6, s=50)
        ax3.plot(qq_data[:, 0], intercept + slope * qq_data[:, 0], 'r-', 
                linewidth=2, label=f'R²={r_val**2:.3f}')
    ax3.set_title(f'Exponential QQ Plot (k={best_k})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Theoretical Quantile')
    ax3.set_ylabel('Sample Quantile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Beta (single direction)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(beta, color='blue', linewidth=2, label=f'Beta (tau={tau:.1f})')
    ax4.set_title(f'FEPLS Direction Beta (tau={tau:.1f})', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Weight')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Conditional Quantile with Scatter
    ax5 = fig.add_subplot(gs[1, 1:])
    # we compute projections using beta - MUST divide by d to match plot_quantile_conditional_on_sample_new
    proj_vals = np.dot(X_fepls[0, :, :], beta) / d
    # we use Silverman's rule of thumb for bandwidth selection (more robust)
    # h = 1.06 * std * n^(-1/5) for univariate, but we use a simpler adaptive version
    proj_std = np.std(proj_vals)
    proj_iqr = np.percentile(proj_vals, 75) - np.percentile(proj_vals, 25)
    # we use the smaller of std and IQR/1.34 (robust scale estimate) for bandwidth
    proj_scale = min(proj_std, proj_iqr / 1.34) if proj_iqr > 0 else proj_std
    h_univ = 1.06 * proj_scale * (n_samples ** (-0.2)) if n_samples > 1 else 0.2 * proj_std
    # we ensure h_univ is not too small or too large
    h_univ = max(h_univ, proj_std * 0.1) if proj_std > 0 else 0.01
    h_univ = min(h_univ, proj_std * 2.0) if proj_std > 0 else 0.1
    
    # for functional bandwidth, we use mean std of functional data
    func_stds = [np.std(X_fepls[0, i, :]) for i in range(n_samples)]
    func_mean_std = np.mean(func_stds) if func_stds else 0.01
    h_func = 1.06 * func_mean_std * (n_samples ** (-0.2)) if n_samples > 1 else 0.2 * func_mean_std
    h_func = max(h_func, func_mean_std * 0.1) if func_mean_std > 0 else 0.01
    h_func = min(h_func, func_mean_std * 2.0) if func_mean_std > 0 else 0.1
    
    h_univ_vec = h_univ * np.ones(n_samples)
    h_func_vec = h_func * np.ones(n_samples)
    
    Y_vals = Y_fepls[0]
    
    # we identify extreme and non-extreme points based on best_k threshold
    Y_sorted_idx = np.argsort(Y_vals)[::-1]
    extreme_threshold = Y_vals[Y_sorted_idx[best_k]] if best_k < len(Y_vals) else np.median(Y_vals)
    is_extreme = Y_vals >= extreme_threshold
    
    try:
        # we use beta directly for plot_quantile_conditional_on_sample_new
        quantiles, s_grid = plot_quantile_conditional_on_sample_new(
            X_fepls, Y_fepls,
            dimred=beta,
            x_func=beta,
            alpha=0.95,
            h_univ_vector=h_univ_vec,
            h_func_vector=h_func_vec
        )
        
        # we check for NaN/Inf in quantiles
        if np.any(np.isnan(quantiles)) or np.any(np.isinf(quantiles)):
            quantiles = np.nan_to_num(quantiles, nan=np.nanpercentile(Y_vals, 95), posinf=np.nanpercentile(Y_vals, 95), neginf=0.0)
        
        # we scatter non-extreme points in blue
        ax5.scatter(proj_vals[~is_extreme], Y_vals[~is_extreme], 
                   alpha=0.4, s=20, color='blue', label='Non-extreme', zorder=1)
        # we scatter extreme points in red
        ax5.scatter(proj_vals[is_extreme], Y_vals[is_extreme], 
                   alpha=0.7, s=25, color='red', label='Extreme', zorder=2)
        # we plot quantile curves - only plot valid (non-NaN) values
        valid_mask_0 = ~np.isnan(quantiles[:, 0]) & ~np.isinf(quantiles[:, 0])
        valid_mask_1 = ~np.isnan(quantiles[:, 1]) & ~np.isinf(quantiles[:, 1])
        if np.any(valid_mask_0):
            ax5.plot(s_grid[valid_mask_0], quantiles[valid_mask_0, 0], 
                    label='Univariate Est.', linestyle='--', linewidth=2, zorder=3, color='green')
        if np.any(valid_mask_1):
            ax5.plot(s_grid[valid_mask_1], quantiles[valid_mask_1, 1], 
                    label='Functional Est.', linewidth=2, zorder=3, color='orange')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Error computing quantiles: {e}', 
                transform=ax5.transAxes, ha='center', fontsize=10)
        import traceback
        traceback.print_exc()
    ax5.set_title(f'Conditional 95% Quantile with Scatter (tau={tau:.1f}, k={best_k})', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Projection <X, Beta>/d')
    ax5.set_ylabel('Y (Response)')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Hypothesis verification summary
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    status_color = 'green' if hypothesis_valid else 'red'
    status_text = 'VALID' if hypothesis_valid else 'INVALID'
    
    summary_text = f"""
    Hypothesis Verification for tau = {tau:.1f}
    ========================================================================
    gamma_hat = {gamma_hat:.6f}
    kappa_hat = {kappa_hat:.6f}
    2*(kappa + tau)*gamma = {hypothesis_value:.6f}
    
    Conditions:
    - Positive: {hypothesis_value > 0.0} (must be > 0)
    - Less than 1: {hypothesis_value < 1.0} (must be < 1)
    
    Status: {status_text}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor=status_color, alpha=0.2))
    
    fig.suptitle(f'{pair_name} - tau = {tau:.1f}', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tau_comparison(
    tau_results: List[Dict],
    pair_name: str,
    save_path: str,
) -> None:
    """we create a comparison plot across all tau values"""
    valid_taus = [r for r in tau_results if r['hypothesis_valid']]
    invalid_taus = [r for r in tau_results if not r['hypothesis_valid']]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Hypothesis value vs tau
    ax1 = axes[0, 0]
    if valid_taus:
        valid_tau_vals = [r['tau'] for r in valid_taus]
        valid_hyp_vals = [r['hypothesis_value'] for r in valid_taus]
        ax1.scatter(valid_tau_vals, valid_hyp_vals, c='green', s=100, 
                   marker='o', label='Valid', zorder=3)
    if invalid_taus:
        invalid_tau_vals = [r['tau'] for r in invalid_taus]
        invalid_hyp_vals = [r['hypothesis_value'] for r in invalid_taus]
        ax1.scatter(invalid_tau_vals, invalid_hyp_vals, c='red', s=100,
                   marker='x', label='Invalid', zorder=3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax1.axhline(y=1, color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax1.axhspan(0, 1, alpha=0.1, color='green', label='Valid region')
    ax1.set_xlabel('tau')
    ax1.set_ylabel('2*(kappa + tau)*gamma')
    ax1.set_title('Hypothesis Value vs tau', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best correlation vs tau
    ax2 = axes[0, 1]
    all_tau_vals = [r['tau'] for r in tau_results]
    all_max_corr = [r['max_correlation'] for r in tau_results]
    colors = ['green' if r['hypothesis_valid'] else 'red' for r in tau_results]
    ax2.scatter(all_tau_vals, all_max_corr, c=colors, s=100, alpha=0.6, marker='o', label='Max Correlation')
    ax2.set_xlabel('tau')
    ax2.set_ylabel('Max Correlation')
    ax2.set_title('Max Correlation vs tau', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best k vs tau
    ax3 = axes[1, 0]
    all_best_k = [r['best_k'] for r in tau_results]
    ax3.scatter(all_tau_vals, all_best_k, c=colors, s=100, alpha=0.6)
    ax3.set_xlabel('tau')
    ax3.set_ylabel('Best k')
    ax3.set_title('Best k vs tau', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if tau_results:
        table_data = []
        headers = ['tau', '2*(k+τ)*γ', 'Max Corr', 'Best k', 'Status']
        for r in sorted(tau_results, key=lambda x: x['tau']):
            status = '✓ VALID' if r['hypothesis_valid'] else '✗ INVALID'
            table_data.append([
                f"{r['tau']:+.2f}",
                f"{r['hypothesis_value']:.4f}",
                f"{r['max_correlation']:.4f}",
                f"{r['best_k']}",
                status
            ])
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.15, 0.25, 0.2, 0.15, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # we color valid rows green
        for i, r in enumerate(sorted(tau_results, key=lambda x: x['tau'])):
            if r['hypothesis_valid']:
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#90EE90')
    
    fig.suptitle(f'{pair_name} - Tau Comparison', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# %% [markdown]
# # Main analysis function

# %%
def analyze_pair_with_multiple_tau(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    pair_name: str,
    tau_grid: List[float],
    global_beta_signatures: Optional[dict] = None,
) -> None:
    """we analyze a pair with multiple tau values - single beta vector"""
    n, d = X_data.shape
    
    # we work with 1D data - compute single beta vector of dimension d
    print(f"  Using 1D data: (n={n}, d={d}) - computing beta vector of dimension {d}")
    
    # we reshape for FEPLS functions (expects (N, n, d) format for 1D)
    X_fepls = np.expand_dims(X_data, axis=0)  # shape (1, n, d)
    Y_fepls = np.expand_dims(Y_data, axis=0)  # shape (1, n)
    
    n_samples = Y_fepls.shape[1]
    m_threshold = int(n_samples / 5)
    
    # we estimate gamma and kappa once (they don't depend on tau)
    # we use average tau for hypothesis verification
    print(f"  Estimating gamma and kappa for {pair_name}...")
    gamma_hat = estimate_gamma(Y_data)
    if gamma_hat is None:
        print(f"  ERROR: Could not estimate gamma for {pair_name}")
        return
    
    # we need an initial beta_hat to estimate kappa (use tau=1.0 as default)
    X_fepls_1d = np.expand_dims(X_data, axis=0)
    beta_hat_init_1d = compute_fepls_direction(X_data, Y_data, tau=1.0)
    if beta_hat_init_1d is None:
        print(f"  ERROR: Could not compute initial beta_hat for {pair_name}")
        return
    
    kappa_hat = estimate_kappa(X_data, Y_data, beta_hat_init_1d)
    if kappa_hat is None:
        print(f"  ERROR: Could not estimate kappa for {pair_name}")
        return
    
    print(f"  gamma_hat = {gamma_hat:.6f}, kappa_hat = {kappa_hat:.6f}")
    
    # we test each tau value
    tau_results = []
    valid_tau_count = 0
    beta_hat_history = []  # we track beta_hat values to detect duplicates
    
    for tau in tau_grid:
        print(f"  Testing tau = {tau:.2f}...")
        
        # we compute best_k using tau
        corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau, m_threshold)
        corr_curve = np.nan_to_num(corr_curve, nan=0.0, posinf=0.0, neginf=0.0)
        
        # we check for NaN/Inf in correlation curve and replace them
        if np.any(np.isnan(corr_curve)) or np.any(np.isinf(corr_curve)):
            print(f"    WARNING: correlation curve contains NaN/Inf for tau={tau:.2f}, replacing with 0")
            corr_curve = np.nan_to_num(corr_curve, nan=0.0, posinf=0.0, neginf=0.0)
        
        # we check if correlation curve is all zeros (which might indicate a problem)
        if np.all(corr_curve == 0.0):
            print(f"    WARNING: correlation curve is all zeros for tau={tau:.2f}, this might indicate a problem")
        
        # we debug: print max correlation to verify it changes with tau
        if len(corr_curve) > 0:
            print(f"    DEBUG: max correlation for tau={tau:.2f}: {np.max(corr_curve):.6f}")
            print(f"    DEBUG: correlation curve first 5 values for tau={tau:.2f}: {corr_curve[:5]}")
        
        # we find best k using sharpness (sharpest peak)
        valid_k_start = 10
        if len(corr_curve) > valid_k_start:
            valid_curve = corr_curve[valid_k_start:]
            # we ensure no NaN/Inf (should already be handled, but double-check)
            valid_curve = np.nan_to_num(valid_curve, nan=0.0, posinf=0.0, neginf=0.0)
            
            # we calculate sharpness for all points in valid range
            sharpness_values = np.zeros(len(valid_curve))
            for i in range(1, len(valid_curve) - 1):
                # sharpness = 2*C[k] - C[k-1] - C[k+1] (local convexity)
                sharpness_values[i] = 2 * valid_curve[i] - valid_curve[i-1] - valid_curve[i+1]
            
            # we find the sharpest peak (highest sharpness)
            # if all sharpness are negative or zero, we use max correlation instead
            if np.all(sharpness_values <= 0):
                # we fall back to max correlation
                best_k_idx = np.argmax(valid_curve)
                best_k = best_k_idx + valid_k_start
                max_corr = valid_curve[best_k_idx]
                sharpness = 0.0
            else:
                best_k_idx = np.argmax(sharpness_values)
                best_k = best_k_idx + valid_k_start
                max_corr = valid_curve[best_k_idx]
                sharpness = sharpness_values[best_k_idx]
        else:
            best_k = valid_k_start if len(corr_curve) > valid_k_start else 2
            max_corr = corr_curve[best_k] if best_k < len(corr_curve) else 0.0
            sharpness = 0.0
        
        print(f"    DEBUG: best_k for tau={tau:.2f}: {best_k}, max_corr={max_corr:.6f}")
        
        # we compute beta using tau
        Y_sorted = np.sort(Y_fepls[0])[::-1]
        y_n = Y_sorted[best_k] if best_k < len(Y_sorted) else Y_sorted[0]
        y_matrix = y_n * np.ones_like(Y_fepls)
        
        print(f"    DEBUG: y_n (threshold) for tau={tau:.2f}: {y_n:.6f}")
        
        try:
            # we verify that tau is actually used by checking the weights
            Y_abs = np.abs(Y_fepls)
            epsilon = 1e-8 if tau >= 0 else 1e-6
            Y_abs = np.maximum(Y_abs, epsilon)
            # we check weights for extreme values
            extreme_mask = Y_fepls[0] >= y_n
            if np.any(extreme_mask):
                weights_sample = Y_abs[0, extreme_mask]**tau
                print(f"    DEBUG: sample weights (Y**tau) for tau={tau:.2f}: min={np.min(weights_sample):.6e}, max={np.max(weights_sample):.6e}, mean={np.mean(weights_sample):.6e}")
            
            # we compute beta using tau
            beta = compute_beta_with_tau(X_fepls[0, :, :], Y_fepls[0, :], tau, y_matrix[0, :])
            
            if beta is None:
                print(f"    ERROR: Could not compute beta for tau={tau:.2f}, skipping")
                continue
            
            # we check for NaN/Inf in beta
            if np.any(np.isnan(beta)) or np.any(np.isinf(beta)):
                print(f"    WARNING: beta contains NaN/Inf for tau={tau:.2f}, trying to fix...")
                beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
                norm = np.linalg.norm(beta)
                if norm > 1e-10:
                    beta = beta / norm
                else:
                    print(f"    ERROR: beta norm is zero for tau={tau:.2f}, skipping")
                    continue
            
            # we debug: print beta norm and first few values to verify they differ across pairs
            beta_norm = np.linalg.norm(beta)
            beta_first = beta[:min(3, len(beta))] if len(beta) > 0 else []
            # we compute a hash-like signature to detect if beta is the same across pairs
            beta_signature = np.sum(beta * np.arange(1, len(beta) + 1))  # weighted sum as signature
            # we compute signatures for X and Y to verify they differ across pairs
            X_flat = X_fepls.flatten()
            Y_flat = Y_fepls.flatten()
            X_signature = np.sum(X_flat * np.arange(1, len(X_flat) + 1))
            Y_signature = np.sum(Y_flat * np.arange(1, len(Y_flat) + 1))
            print(f"    DEBUG: beta norm for tau={tau:.2f}: {beta_norm:.6f}, shape: {beta.shape}, first values: {beta_first}, signature: {beta_signature:.6f}")
            print(f"    DEBUG: X_fepls shape: {X_fepls.shape}, X_fepls mean: {np.mean(X_fepls):.6f}, X_fepls std: {np.std(X_fepls):.6f}, X_signature: {X_signature:.6f}")
            print(f"    DEBUG: Y_fepls shape: {Y_fepls.shape}, Y_fepls mean: {np.mean(Y_fepls):.6f}, Y_fepls std: {np.std(Y_fepls):.6f}, Y_signature: {Y_signature:.6f}")
            
            # we check if this beta signature matches any previous pair (global check)
            if global_beta_signatures is not None:
                for (prev_pair, prev_tau), prev_beta_sig in global_beta_signatures.items():
                    if prev_pair != pair_name:  # we only check across different pairs
                        # we check if signatures are very close (within 1e-3)
                        if abs(beta_signature - prev_beta_sig) < 1e-3:
                            print(f"    ⚠️  WARNING: beta signature for {pair_name} tau={tau:.2f} is VERY SIMILAR to {prev_pair} tau={prev_tau}!")
                            print(f"       beta signatures: {beta_signature:.6f} vs {prev_beta_sig:.6f} (diff: {abs(beta_signature - prev_beta_sig):.6e})")
                
                # we store this pair's beta signature
                global_beta_signatures[(pair_name, tau)] = beta_signature
            
            # we check if this beta is identical to a previous one
            for prev_tau, prev_beta in beta_hat_history:
                if np.allclose(beta, prev_beta, atol=1e-6):
                    print(f"    WARNING: beta for tau={tau:.2f} is identical to tau={prev_tau:.2f} (within tolerance)")
                    break
            
            beta_hat_history.append((tau, beta.copy()))
        except Exception as e:
            print(f"    ERROR computing beta for tau={tau:.2f}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # we verify hypothesis using tau
        hypothesis_value = 2.0 * (kappa_hat + tau) * gamma_hat
        condition_pos = hypothesis_value > 0.0
        condition_upper = hypothesis_value < 1.0
        hypothesis_valid = condition_pos and condition_upper
        
        if hypothesis_valid:
            valid_tau_count += 1
            print(f"    ✓ VALID: 2*(kappa+tau)*gamma = {hypothesis_value:.6f}")
        else:
            print(f"    ✗ INVALID: 2*(kappa+tau)*gamma = {hypothesis_value:.6f}")
        
        # we store results
        tau_results.append({
            'tau': tau,
            'beta': beta,
            'best_k': best_k,
            'max_correlation': max_corr,
            'sharpness': sharpness,
            'hypothesis_value': hypothesis_value,
            'hypothesis_valid': hypothesis_valid,
        })
        
        # we create plot for ALL tau (not just valid ones), but we keep the valid/invalid flag
        plot_path = os.path.join(SAVE_DIR, f"{pair_name}_tau_{tau:.1f}.png")
        # we pass the precomputed corr_curve to avoid recalculating it (which might give different results)
        plot_single_tau_analysis(
            X_fepls, Y_fepls, beta, tau, best_k,
            gamma_hat, kappa_hat, hypothesis_value, hypothesis_valid,
            pair_name, plot_path, corr_curve=corr_curve
        )
    
    # we create comparison plot
    comparison_path = os.path.join(SAVE_DIR, f"{pair_name}_tau_comparison.png")
    plot_tau_comparison(tau_results, pair_name, comparison_path)
    
    print(f"  Completed {pair_name}: {valid_tau_count}/{len(tau_grid)} valid tau values")
    print(f"  Plots saved to {SAVE_DIR}")


# %% [markdown]
# # Main execution

# %%
def main():
    """we run tau comparison analysis for selected pairs"""
    print("=" * 80)
    print("Large-scale Tau Comparison with Hypothesis Verification")
    print("=" * 80)
    
    # we create a global dictionary to track beta signatures across all pairs
    global_beta_signatures = {}  # key: (pair_name, tau), value: beta_signature
    
    # we load functional dataset
    print("\nLoading functional dataset...")
    func_data, tickers = build_functional_dataset()
    print(f"Loaded {len(tickers)} assets")
    
    # we generate all possible pairs from available tickers
    # we use itertools.permutations to get all ordered pairs (X, Y) where X != Y
    all_possible_pairs = list(itertools.permutations(tickers, 2))
    print(f"\nGenerated {len(all_possible_pairs)} possible pairs from {len(tickers)} tickers")
    
    # we can optionally limit the number of pairs to analyze (set to None for all pairs)
    max_pairs_to_analyze = None  # we set to None to analyze all pairs, or set a number to limit
    if max_pairs_to_analyze is not None and max_pairs_to_analyze < len(all_possible_pairs):
        pairs_to_analyze = all_possible_pairs[:max_pairs_to_analyze]
        print(f"Limiting analysis to first {max_pairs_to_analyze} pairs")
    else:
        pairs_to_analyze = all_possible_pairs
        print(f"Analyzing all {len(pairs_to_analyze)} pairs")
    
    print(f"\nAnalyzing {len(pairs_to_analyze)} pairs with tau grid: {len(TAU_GRID)} values")
    print(f"  tau values: {TAU_VALUES}")
    
    for ticker_X, ticker_Y in pairs_to_analyze:
        if ticker_X not in func_data or ticker_Y not in func_data:
            print(f"Skipping {ticker_X} -> {ticker_Y}: data not available")
            continue
        
        # we align data
        dates_X = func_data[ticker_X]['dates']
        dates_Y = func_data[ticker_Y]['dates']
        common_dates = dates_X.intersection(dates_Y)
        
        if len(common_dates) < 100:
            print(f"Skipping {ticker_X} -> {ticker_Y}: insufficient common dates ({len(common_dates)})")
            continue
        
        idx_X = dates_X.isin(common_dates)
        idx_Y = dates_Y.isin(common_dates)
        
        if hasattr(idx_X, 'values'):
            idx_X = idx_X.values
        if hasattr(idx_Y, 'values'):
            idx_Y = idx_Y.values
        idx_X = np.asarray(idx_X, dtype=bool)
        idx_Y = np.asarray(idx_Y, dtype=bool)
        
        X_data = func_data[ticker_X]['curves'][idx_X]
        Y_data = func_data[ticker_Y]['max_return'][idx_Y]
        
        n = min(X_data.shape[0], Y_data.shape[0])
        if n < 50:
            print(f"Skipping {ticker_X} -> {ticker_Y}: insufficient samples ({n})")
            continue
        
        X_array = X_data[:n]
        Y_array = Y_data[:n]
        
        pair_name = f"{ticker_X.replace('.hu.txt', '')}_{ticker_Y.replace('.hu.txt', '')}"
        
        # we debug: print data statistics to verify they differ across pairs
        print(f"\n{'='*80}")
        print(f"Analyzing pair: {pair_name} (n={n})")
        print(f"  X_data shape: {X_array.shape}, X_data mean: {np.mean(X_array):.6f}, X_data std: {np.std(X_array):.6f}")
        print(f"  Y_data shape: {Y_array.shape}, Y_data mean: {np.mean(Y_array):.6f}, Y_data std: {np.std(Y_array):.6f}")
        print(f"{'='*80}")
        
        try:
            analyze_pair_with_multiple_tau(X_array, Y_array, pair_name, TAU_GRID, global_beta_signatures)
        except Exception as e:
            print(f"ERROR analyzing {pair_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Plots saved to: {SAVE_DIR}")
    print(f"{'='*80}")


# %%
if __name__ == "__main__":
    main()

