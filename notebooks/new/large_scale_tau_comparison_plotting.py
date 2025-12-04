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
        fepls, bitcoin_concomittant_corr, get_hill_estimator,
        Exponential_QQ_Plot_1D, plot_quantile_conditional_on_sample_new
    )
except ImportError:
    # we try alternative import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("adapted", os.path.join(current_dir, "adapted.py"))
    adapted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapted)
    fepls = adapted.fepls
    bitcoin_concomittant_corr = adapted.bitcoin_concomittant_corr
    get_hill_estimator = adapted.get_hill_estimator
    Exponential_QQ_Plot_1D = adapted.Exponential_QQ_Plot_1D
    plot_quantile_conditional_on_sample_new = adapted.plot_quantile_conditional_on_sample_new

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
    
    # Plot 2: Best correlation vs tau
    ax2 = axes[0, 1]
    all_tau_vals = [r['tau'] for r in tau_results]
    all_max_corr = [r['max_correlation'] for r in tau_results]
    colors = ['green' if r['hypothesis_valid'] else 'red' for r in tau_results]
    ax2.scatter(all_tau_vals, all_max_corr, c=colors, s=100, alpha=0.6)
    ax2.set_xlabel('tau')
    ax2.set_ylabel('Max Correlation')
    ax2.set_title('Max Correlation vs tau', fontweight='bold')
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
# # Main analysis function

# %%
def analyze_pair_with_multiple_tau(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    pair_name: str,
    tau_grid: List[float],
) -> None:
    """we analyze a pair with multiple tau values, verify hypotheses, and create plots"""
    n, d = X_data.shape
    
    # we reshape for FEPLS functions (expects (N, n, d) format)
    X_fepls = np.expand_dims(X_data, axis=0)
    Y_fepls = np.expand_dims(Y_data, axis=0)
    
    n_samples = Y_fepls.shape[1]
    d_points = X_fepls.shape[2]
    m_threshold = int(n_samples / 5)
    
    # we estimate gamma and kappa once (they don't depend on tau)
    print(f"  Estimating gamma and kappa for {pair_name}...")
    gamma_hat = estimate_gamma(Y_data)
    if gamma_hat is None:
        print(f"  ERROR: Could not estimate gamma for {pair_name}")
        return
    
    # we need an initial beta_hat to estimate kappa (use tau=1.0 as default)
    beta_hat_init = compute_fepls_direction(X_data, Y_data, tau=1.0)
    if beta_hat_init is None:
        print(f"  ERROR: Could not compute initial beta_hat for {pair_name}")
        return
    
    kappa_hat = estimate_kappa(X_data, Y_data, beta_hat_init)
    if kappa_hat is None:
        print(f"  ERROR: Could not estimate kappa for {pair_name}")
        return
    
    print(f"  gamma_hat = {gamma_hat:.6f}, kappa_hat = {kappa_hat:.6f}")
    
    # we test each tau value
    tau_results = []
    valid_tau_count = 0
    
    for tau in tau_grid:
        print(f"  Testing tau = {tau}...")
        
        # we compute correlation curve for this tau
        corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau, m_threshold)
        
        # we find best k using sharpness (sharpest peak)
        valid_k_start = 5
        if len(corr_curve) > valid_k_start:
            valid_curve = corr_curve[valid_k_start:]
            # we calculate sharpness for all points in valid range
            sharpness_values = np.zeros(len(valid_curve))
            for i in range(1, len(valid_curve) - 1):
                # sharpness = 2*C[k] - C[k-1] - C[k+1] (local convexity)
                sharpness_values[i] = 2 * valid_curve[i] - valid_curve[i-1] - valid_curve[i+1]
            # we find the sharpest peak (highest sharpness)
            best_k_idx = np.argmax(sharpness_values)
            best_k = best_k_idx + valid_k_start
            max_corr = valid_curve[best_k_idx]
            sharpness = sharpness_values[best_k_idx]
        else:
            best_k = valid_k_start if len(corr_curve) > valid_k_start else 2
            max_corr = 0.0
            sharpness = 0.0
        
        # we compute beta_hat for this tau using FEPLS
        Y_sorted = np.sort(Y_fepls[0])[::-1]
        y_n = Y_sorted[best_k]
        y_matrix = y_n * np.ones_like(Y_fepls)
        
        try:
            E0 = fepls(X_fepls, Y_fepls, y_matrix, tau)
            beta_hat = E0[0, :]
        except Exception as e:
            print(f"    ERROR computing beta_hat for tau={tau}: {e}")
            continue
        
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
            'max_correlation': max_corr,
            'sharpness': sharpness,
            'hypothesis_value': hypothesis_value,
            'hypothesis_valid': hypothesis_valid,
        })
        
        # we create plot for this tau (only if valid, or create all if you want)
        if hypothesis_valid:  # we only plot valid tau to save space
            plot_path = os.path.join(SAVE_DIR, f"{pair_name}_tau_{tau:.1f}.png")
            plot_single_tau_analysis(
                X_fepls, Y_fepls, beta_hat, tau, best_k,
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
    
    # we define pairs to analyze (added 10 more + listed tickers from the prompt)
    pairs_to_analyze = [
        ('mol.hu.txt', 'otp.hu.txt'),
        ('otp.hu.txt', 'mol.hu.txt'),
        ('4ig.hu.txt', 'akko.hu.txt'),
        ('4ig.hu.txt', 'alteo.hu.txt'),
        ('4ig.hu.txt', 'amixa.hu.txt'),
        ('4ig.hu.txt', 'any.hu.txt'),
        ('4ig.hu.txt', 'appeninn.hu.txt'),
        ('4ig.hu.txt', 'astrasun.hu.txt'),
        ('4ig.hu.txt', 'autowallis.hu.txt'),
        ('4ig.hu.txt', 'bet.hu.txt'),
        ('4ig.hu.txt', 'bgreit.hu.txt'),
        ('akko.hu.txt', 'alteo.hu.txt'),
        ('akko.hu.txt', 'amixa.hu.txt'),
        ('akko.hu.txt', 'any.hu.txt'),
        ('akko.hu.txt', 'appeninn.hu.txt'),
        ('akko.hu.txt', 'astrasun.hu.txt'),
        ('akko.hu.txt', 'autowallis.hu.txt'),
        ('akko.hu.txt', 'bet.hu.txt'),
        ('akko.hu.txt', 'bgreit.hu.txt'),
        ('alteo.hu.txt', 'amixa.hu.txt'),
        ('alteo.hu.txt', 'any.hu.txt'),
        ('alteo.hu.txt', 'appeninn.hu.txt'),
        ('alteo.hu.txt', 'astrasun.hu.txt'),
        ('alteo.hu.txt', 'autowallis.hu.txt'),
        ('alteo.hu.txt', 'bet.hu.txt'),
        ('alteo.hu.txt', 'bgreit.hu.txt'),
        ('amixa.hu.txt', 'any.hu.txt'),
        ('amixa.hu.txt', 'appeninn.hu.txt'),
        ('amixa.hu.txt', 'astrasun.hu.txt'),
        ('amixa.hu.txt', 'autowallis.hu.txt'),
        # Add some random single-direction pairs to get near 30
        ('bif.hu.txt', 'biggeorge.hu.txt'),
        ('cdsys.hu.txt', 'chome.hu.txt'),
        ('cigpannonia.hu.txt', 'civita.hu.txt'),
        ('delta.hu.txt', 'deltakamatoz.hu.txt'),
        ('dmker.hu.txt', 'dunahouse.hu.txt'),
        ('enefi_els.hu.txt', 'enefi.hu.txt'),
        ('energyinvest.hu.txt', 'epduferr.hu.txt'),
        ('eproliusia.hu.txt', 'esense.hu.txt'),
        ('eu-solar.hu.txt', 'finext_b.hu.txt')
    ]
    
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
            analyze_pair_with_multiple_tau(X_array, Y_array, pair_name, TAU_GRID)
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

