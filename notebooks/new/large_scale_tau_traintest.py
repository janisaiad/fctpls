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
    from adapted import (
        fepls as fepls_original, bitcoin_concomittant_corr as bitcoin_concomittant_corr_original,
        get_hill_estimator, Exponential_QQ_Plot_1D, plot_quantile_conditional_on_sample_new
    )
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
def fepls_safe(X, Y, y_matrix, tau):
    """we wrap fepls to handle negative Y values by using absolute value"""
    # we use absolute value of Y to avoid NaN with negative exponents
    # this is safe because we're interested in the magnitude of extremes
    Y_abs = np.abs(Y)
    y_matrix_abs = np.abs(y_matrix)
    try:
        result = fepls_original(X, Y_abs, y_matrix_abs, tau)
        # we check for NaN/Inf
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return None
        return result
    except Exception:
        return None

def bitcoin_concomittant_corr_safe(X, Y, tau, m):
    """we wrap bitcoin_concomittant_corr to handle negative Y values"""
    # we use absolute value of Y to avoid NaN with negative exponents
    Y_abs = np.abs(Y)
    try:
        result = bitcoin_concomittant_corr_original(X, Y_abs, tau, m)
        # we check for NaN/Inf and replace with 0
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result
    except Exception:
        return np.zeros(m)

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

# we define tau grid to test
TAU_GRID = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

# we set random seed
np.random.seed(42)


# %% [markdown]
# # Helper functions for plotting

# %%
def plot_single_tau_analysis(
    X_fepls: np.ndarray,
    Y_fepls: np.ndarray,
    beta_hat: np.ndarray,
    tau: float,
    best_k: int,
    gamma_hat: float,
    kappa_hat: float,
    hypothesis_value: float,
    hypothesis_valid: bool,
    pair_name: str,
    save_path: str,
) -> None:
    """we create a comprehensive plot for a single tau value"""
    n_samples = Y_fepls.shape[1]
    d_points = X_fepls.shape[2]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # we compute correlation curve for this tau
    m_threshold = int(n_samples / 5)
    corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau, m_threshold)
    
    Y_sorted = np.sort(Y_fepls[0])[::-1]
    
    # Plot 1: Correlation curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(corr_curve, 'b-', linewidth=2)
    ax1.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best k={best_k}')
    ax1.set_title(f'Tail Correlation vs k (tau={tau})', fontsize=12, fontweight='bold')
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
    
    # Plot 4: Beta Curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(beta_hat, color='purple', linewidth=2)
    ax4.set_title(f'FEPLS Direction Beta(t) (tau={tau})', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Intraday Time (Index)')
    ax4.set_ylabel('Weight')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Conditional Quantile with Scatter
    ax5 = fig.add_subplot(gs[1, 1:])
    h_univ = 0.2 * np.std(np.dot(X_fepls[0], beta_hat) / d_points)
    h_func = 0.2 * np.mean(np.std(X_fepls[0], axis=0))
    h_univ_vec = h_univ * np.ones(n_samples)
    h_func_vec = h_func * np.ones(n_samples)
    
    # we compute projections for scatter plot
    proj_vals = np.dot(X_fepls[0], beta_hat) / d_points
    Y_vals = Y_fepls[0]
    
    # we identify extreme and non-extreme points based on best_k threshold
    Y_sorted_idx = np.argsort(Y_vals)[::-1]
    extreme_threshold = Y_vals[Y_sorted_idx[best_k]] if best_k < len(Y_vals) else np.median(Y_vals)
    is_extreme = Y_vals >= extreme_threshold
    
    try:
        quantiles, s_grid = plot_quantile_conditional_on_sample_new(
            X_fepls, Y_fepls,
            dimred=beta_hat,
            x_func=beta_hat,
            alpha=0.95,
            h_univ_vector=h_univ_vec,
            h_func_vector=h_func_vec
        )
        # we scatter non-extreme points in blue
        ax5.scatter(proj_vals[~is_extreme], Y_vals[~is_extreme], 
                   alpha=0.4, s=20, color='blue', label='Non-extreme', zorder=1)
        # we scatter extreme points in red
        ax5.scatter(proj_vals[is_extreme], Y_vals[is_extreme], 
                   alpha=0.7, s=25, color='red', label='Extreme', zorder=2)
        # we plot quantile curves
        ax5.plot(s_grid, quantiles[:, 0], label='Univariate Est.', linestyle='--', linewidth=2, zorder=3)
        ax5.plot(s_grid, quantiles[:, 1], label='Functional Est.', linewidth=2, zorder=3)
    except Exception as e:
        ax5.text(0.5, 0.5, f'Error computing quantiles: {e}', 
                transform=ax5.transAxes, ha='center')
    ax5.set_title(f'Conditional 95% Quantile with Scatter (tau={tau}, k={best_k})', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Projection <X, Beta>')
    ax5.set_ylabel('Y (Response)')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Hypothesis verification summary
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    status_color = 'green' if hypothesis_valid else 'red'
    status_text = 'VALID' if hypothesis_valid else 'INVALID'
    
    summary_text = f"""
    Hypothesis Verification for tau = {tau}
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
    
    fig.suptitle(f'{pair_name} - tau = {tau}', fontsize=16, fontweight='bold')
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
    
    # Plot 2: Train and Test correlation vs tau
    ax2 = axes[0, 1]
    all_tau_vals = [r['tau'] for r in tau_results]
    all_max_corr_train = [r['max_correlation_train'] for r in tau_results]
    all_test_corr = [r['test_correlation'] for r in tau_results]
    colors = ['green' if r['hypothesis_valid'] else 'red' for r in tau_results]
    ax2.scatter(all_tau_vals, all_max_corr_train, c=colors, s=100, alpha=0.6, 
               marker='o', label='Train Correlation')
    ax2.scatter(all_tau_vals, all_test_corr, c=colors, s=100, alpha=0.6, 
               marker='x', label='Test Correlation')
    ax2.set_xlabel('tau')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Train vs Test Correlation vs tau', fontweight='bold')
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
    
    if valid_taus:
        table_data = []
        headers = ['tau', '2*(k+τ)*γ', 'Train Corr', 'Test Corr', 'Best k', 'Status']
        for r in sorted(tau_results, key=lambda x: x['tau']):
            status = '✓ VALID' if r['hypothesis_valid'] else '✗ INVALID'
            table_data.append([
                f"{r['tau']:+.2f}",
                f"{r['hypothesis_value']:.4f}",
                f"{r['max_correlation_train']:.4f}",
                f"{r['test_correlation']:.4f}",
                f"{r['best_k']}",
                status
            ])
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.12, 0.2, 0.15, 0.15, 0.12, 0.26])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # we color valid rows green
        for i, r in enumerate(tau_results):
            if r['hypothesis_valid']:
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#90EE90')
    else:
        ax4.text(0.5, 0.5, 'No valid tau values found', 
                transform=ax4.transAxes, ha='center', fontsize=14)
    
    fig.suptitle(f'{pair_name} - Tau Comparison', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# %% [markdown]
# # Train/Test split function

# %%
def split_train_test(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """we split data into train and test sets"""
    n = X_data.shape[0]
    n_train = int(n * train_ratio)
    
    # we use random permutation for train/test split
    rng = np.random.RandomState(random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X_data[train_indices]
    Y_train = Y_data[train_indices]
    X_test = X_data[test_indices]
    Y_test = Y_data[test_indices]
    
    return X_train, Y_train, X_test, Y_test


# %% [markdown]
# # Main analysis function

# %%
def analyze_pair_with_multiple_tau(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    pair_name: str,
    tau_grid: List[float],
    train_ratio: float = 0.8,
) -> None:
    """we analyze a pair with multiple tau values, verify hypotheses, and create plots"""
    n, d = X_data.shape
    
    # we split into train and test sets
    print(f"  Splitting data into train ({train_ratio*100:.0f}%) and test ({(1-train_ratio)*100:.0f}%)...")
    X_train, Y_train, X_test, Y_test = split_train_test(X_data, Y_data, train_ratio=train_ratio)
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # we check Y data: log returns can be negative, but max_return should be positive
    # we verify that Y contains valid values (no NaN/Inf)
    if np.any(np.isnan(Y_train)) or np.any(np.isinf(Y_train)):
        print(f"  WARNING: Y_train contains NaN/Inf values")
    if np.any(np.isnan(Y_test)) or np.any(np.isinf(Y_test)):
        print(f"  WARNING: Y_test contains NaN/Inf values")
    print(f"  Y_train stats: min={np.min(Y_train):.6f}, max={np.max(Y_train):.6f}, mean={np.mean(Y_train):.6f}")
    print(f"  Y_test stats: min={np.min(Y_test):.6f}, max={np.max(Y_test):.6f}, mean={np.mean(Y_test):.6f}")
    
    # we reshape for FEPLS functions (expects (N, n, d) format)
    X_fepls_train = np.expand_dims(X_train, axis=0)
    Y_fepls_train = np.expand_dims(Y_train, axis=0)
    X_fepls_test = np.expand_dims(X_test, axis=0)
    Y_fepls_test = np.expand_dims(Y_test, axis=0)
    
    n_samples_train = Y_fepls_train.shape[1]
    d_points = X_fepls_train.shape[2]
    m_threshold = int(n_samples_train / 5)
    
    # we estimate gamma and kappa on TRAIN set only (they don't depend on tau)
    print(f"  Estimating gamma and kappa on TRAIN set for {pair_name}...")
    gamma_hat = estimate_gamma(Y_train)
    if gamma_hat is None:
        print(f"  ERROR: Could not estimate gamma for {pair_name}")
        return
    
    # we need an initial beta_hat to estimate kappa on TRAIN set (use tau=1.0 as default)
    beta_hat_init = compute_fepls_direction(X_train, Y_train, tau=1.0)
    if beta_hat_init is None:
        print(f"  ERROR: Could not compute initial beta_hat for {pair_name}")
        return
    
    kappa_hat = estimate_kappa(X_train, Y_train, beta_hat_init)
    if kappa_hat is None:
        print(f"  ERROR: Could not estimate kappa for {pair_name}")
        return
    
    print(f"  gamma_hat = {gamma_hat:.6f}, kappa_hat = {kappa_hat:.6f}")
    
    # we test each tau value
    tau_results = []
    valid_tau_count = 0
    
    for tau in tau_grid:
        print(f"  Testing tau = {tau}...")
        
        # we compute correlation curve for this tau on TRAIN set
        corr_curve = bitcoin_concomittant_corr(X_fepls_train, Y_fepls_train, tau, m_threshold)
        
        # we check for NaN/Inf in correlation curve
        if np.any(np.isnan(corr_curve)) or np.any(np.isinf(corr_curve)):
            print(f"    WARNING: correlation curve contains NaN/Inf for tau={tau}, skipping")
            continue
        
        # we find best k using sharpness (sharpest peak) on TRAIN set
        valid_k_start = 5
        if len(corr_curve) > valid_k_start:
            valid_curve = corr_curve[valid_k_start:]
            # we filter out NaN/Inf values
            valid_mask = ~(np.isnan(valid_curve) | np.isinf(valid_curve))
            if np.sum(valid_mask) < 3:  # we need at least 3 valid points for sharpness
                print(f"    WARNING: insufficient valid correlation values for tau={tau}, skipping")
                continue
            # we calculate sharpness for all points in valid range
            sharpness_values = np.zeros(len(valid_curve))
            for i in range(1, len(valid_curve) - 1):
                if valid_mask[i] and valid_mask[i-1] and valid_mask[i+1]:
                    # sharpness = 2*C[k] - C[k-1] - C[k+1] (local convexity)
                    sharpness_values[i] = 2 * valid_curve[i] - valid_curve[i-1] - valid_curve[i+1]
                else:
                    sharpness_values[i] = -np.inf  # we mark invalid points
            # we find the sharpest peak (highest sharpness) among valid points
            valid_sharpness_mask = ~(np.isnan(sharpness_values) | np.isinf(sharpness_values))
            if np.sum(valid_sharpness_mask) == 0:
                print(f"    WARNING: no valid sharpness values for tau={tau}, skipping")
                continue
            best_k_idx = np.argmax(sharpness_values)
            best_k = best_k_idx + valid_k_start
            max_corr = valid_curve[best_k_idx]
            sharpness = sharpness_values[best_k_idx]
        else:
            best_k = valid_k_start if len(corr_curve) > valid_k_start else 2
            max_corr = 0.0
            sharpness = 0.0
        
        # we compute beta_hat for this tau using FEPLS on TRAIN set
        Y_sorted_train = np.sort(Y_fepls_train[0])[::-1]
        y_n = Y_sorted_train[best_k] if best_k < len(Y_sorted_train) else Y_sorted_train[0]
        y_matrix = y_n * np.ones_like(Y_fepls_train)
        
        try:
            E0 = fepls(X_fepls_train, Y_fepls_train, y_matrix, tau)
            beta_hat = E0[0, :]
        except Exception as e:
            print(f"    ERROR computing beta_hat for tau={tau}: {e}")
            continue
        
        # we evaluate on TEST set: compute test correlation and projections
        test_proj = np.dot(X_test, beta_hat) / d_points
        # we check for NaN/Inf before computing correlation
        if (len(test_proj) > 1 and np.std(test_proj) > 0 and 
            not np.any(np.isnan(test_proj)) and not np.any(np.isinf(test_proj)) and
            not np.any(np.isnan(Y_test)) and not np.any(np.isinf(Y_test)) and
            np.std(Y_test) > 0):
            test_corr = np.corrcoef(test_proj, Y_test)[0, 1]
            if np.isnan(test_corr) or np.isinf(test_corr):
                test_corr = 0.0
        else:
            test_corr = 0.0
        
        # we verify hypothesis
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
            'beta_hat': beta_hat,
            'best_k': best_k,
            'max_correlation_train': max_corr,  # train correlation
            'test_correlation': test_corr,  # test correlation
            'sharpness': sharpness,
            'hypothesis_value': hypothesis_value,
            'hypothesis_valid': hypothesis_valid,
        })
        
        # we create plot for this tau (only if valid, or create all if you want)
        if hypothesis_valid:  # we only plot valid tau to save space
            plot_path = os.path.join(SAVE_DIR, f"{pair_name}_tau_{tau:.1f}.png")
            # we plot using TRAIN data for visualization
            plot_single_tau_analysis(
                X_fepls_train, Y_fepls_train, beta_hat, tau, best_k,
                gamma_hat, kappa_hat, hypothesis_value, hypothesis_valid,
                pair_name, plot_path
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
    
    print(f"\nAnalyzing {len(pairs_to_analyze)} pairs with tau grid: {TAU_GRID}")
    
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
        
        print(f"\n{'='*80}")
        print(f"Analyzing pair: {pair_name} (n={n})")
        print(f"{'='*80}")
        
        try:
            analyze_pair_with_multiple_tau(X_array, Y_array, pair_name, TAU_GRID, train_ratio=0.8)
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

