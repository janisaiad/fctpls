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
import numpy as np
import pandas as pd
import numba
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
import os
import sys

# we add the current directory to import functions from adapted.py
# this works both in notebooks and scripts
try:
    # for scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # for notebooks
    current_dir = os.getcwd()

sys.path.append(current_dir)
from adapted import (
    load_stooq_file, create_functional_data, 
    get_hill_estimator, Exponential_QQ_Plot_1D
)

# Set random seed
np.random.seed(42)

# %% [markdown]
# # Vérification des Hypothèses Théoriques FEPLS
# 
# Ce script teste les hypothèses du papier FEPLS sur les données réelles.

# %%
# we load data (same as adapted.py)
DATA_DIR = "../../data/stooq/hungary/5_hu_txt/data/5 min/hu/bse stocks/"

targets = [
    'otp.hu.txt', 'mol.hu.txt', 'richter.hu.txt'
]

data_store = {}
for t in targets:
    path = os.path.join(DATA_DIR, t)
    if os.path.exists(path):
        print(f"Loading {t}...")
        data_store[t] = load_stooq_file(path)

# we create functional data
if 'otp.hu.txt' in data_store:
    _, master_grid = create_functional_data(data_store, 'otp.hu.txt')
    print(f"Master time grid: {len(master_grid)} points per day.")

func_data = {}
for t in data_store.keys():
    mat, _ = create_functional_data(data_store, t, time_grid=master_grid)
    if mat is not None:
        log_prices = np.log(mat.values)
        diff_curves = np.diff(log_prices, axis=1)
        func_data[t] = {
            'dates': mat.index,
            'curves': diff_curves,
            'max_return': np.max(diff_curves, axis=1)
        }

# %% [markdown]
# # 1. Test: Appartenance au Domaine d'Attraction de Fréchet (Queues Lourdes)

# %%
@numba.njit(parallel=False)
def pickands_estimator(ordered_data, k_max):
    """we estimate Pickands estimator for tail index"""
    n = len(ordered_data)
    estimates = np.zeros(k_max)
    for k in range(1, min(k_max, n//4)):
        if k < n//4:
            m = 2 * k
            if m < n:
                estimates[k] = (np.log(ordered_data[n-k-1] - ordered_data[n-2*k-1]) - 
                               np.log(ordered_data[n-2*k-1] - ordered_data[n-4*k-1])) / np.log(2)
    return estimates

def test_frechet_domain(Y_data, ticker_name, save_dir=None):
    """we test if Y belongs to Fréchet domain of attraction"""
    try:
        Y_sorted = np.sort(Y_data)[::-1]  # descending order
        n = len(Y_data)
        
        if n < 20:
            print(f"Warning: Not enough data for {ticker_name} (n={n})")
            return None
        
        k_max = min(n//5, 200)
        k_range = np.arange(5, k_max)
        
        if len(k_range) == 0:
            print(f"Warning: Empty k_range for {ticker_name}")
            return None
        
        # we compute Hill estimator
        hill_est = get_hill_estimator(Y_sorted)
        hill_est = hill_est[:k_max-1]
        
        # we compute Pickands estimator
        pickands_est = pickands_estimator(Y_sorted, k_max)
    except Exception as e:
        print(f"Error in test_frechet_domain for {ticker_name}: {e}")
        return None
    
    # we create plotly figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hill Plot: Estimation de gamma', 
                       'Pickands Plot: Vérification de la stabilité',
                       'Exponential QQ Plot', 
                       'Survival Function (Log-Log)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Hill Plot
    # we try to find a stable region for gamma estimation
    if len(hill_est) > 20:
        stable_region = hill_est[20:min(50, len(hill_est))]
    elif len(hill_est) > 10:
        stable_region = hill_est[10:]
    else:
        stable_region = hill_est[5:] if len(hill_est) > 5 else hill_est
    
    # we filter positive values and compute mean
    positive_vals = stable_region[stable_region > 0]
    if len(positive_vals) > 0:
        mean_gamma = np.mean(positive_vals)
    else:
        # we use median if no positive values
        mean_gamma = np.median(hill_est[hill_est > 0]) if np.any(hill_est > 0) else None
    
    fig.add_trace(
        go.Scatter(x=k_range, y=hill_est[4:k_max-1], mode='lines', name='Hill Estimator',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                 annotation_text="Borne supérieure: gamma=1", row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", 
                 annotation_text="Référence: gamma=0.5", row=1, col=1)
    if mean_gamma is not None and mean_gamma > 0:
        fig.add_annotation(
            text=f"Mean gamma (k=20-50): {mean_gamma:.3f}",
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, showarrow=False,
            bgcolor="wheat", bordercolor="black",
            row=1, col=1
        )
    
    # Plot 2: Pickands Plot
    valid_pickands = pickands_est[pickands_est > 0]
    valid_k = np.arange(len(pickands_est))[pickands_est > 0]
    if len(valid_pickands) > 5:
        fig.add_trace(
            go.Scatter(x=valid_k[5:], y=valid_pickands[5:], mode='lines', name='Pickands Estimator',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=1, col=2)
    
    # Plot 3: Exponential QQ Plot
    k_qq = min(50, n//10)
    qq_data = Exponential_QQ_Plot_1D(np.expand_dims(Y_data, axis=0), k_qq)
    if len(qq_data) > 1:
        slope, intercept, r_val, _, _ = linregress(qq_data[:,0], qq_data[:,1])
        fig.add_trace(
            go.Scatter(x=qq_data[:,0], y=qq_data[:,1], mode='markers', name='Data',
                      marker=dict(size=5, opacity=0.6)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=qq_data[:,0], y=intercept + slope*qq_data[:,0], mode='lines', 
                      name=f'Linear fit (R²={r_val**2:.3f}, slope={slope:.3f})',
                      line=dict(color='red', width=2)),
            row=2, col=1
        )
        fig.add_annotation(
            text=f"Slope ≈ gamma: {slope:.3f}",
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, showarrow=False,
            bgcolor="lightblue", bordercolor="black",
            row=2, col=1
        )
    
    # Plot 4: Survival Function (log-log plot)
    Y_sorted_asc = np.sort(Y_data)
    y_thresholds = Y_sorted_asc[::-1][:k_max]
    empirical_survival = np.arange(1, len(y_thresholds)+1) / n
    mask = (y_thresholds > 0) & (empirical_survival > 0)
    if np.sum(mask) > 10:
        fig.add_trace(
            go.Scatter(x=y_thresholds[mask], y=empirical_survival[mask], mode='lines', 
                      name='Empirical F_bar(y)', line=dict(color='blue', width=2)),
            row=2, col=2
        )
        if mean_gamma and mean_gamma > 0:
            y_fit = y_thresholds[mask]
            log_y = np.log(y_fit[y_fit > 0])
            log_surv = np.log(empirical_survival[mask][y_fit > 0])
            if len(log_y) > 5:
                slope_surv, _, _, _, _ = linregress(log_y, log_surv)
                y_theory = np.exp(slope_surv * np.log(y_fit) + log_surv[0] - slope_surv * log_y[0])
                fig.add_trace(
                    go.Scatter(x=y_fit, y=y_theory, mode='lines', 
                              name=f'Power law fit (alpha={-slope_surv:.3f})',
                              line=dict(color='red', width=2, dash='dash')),
                    row=2, col=2
                )
    
    # we update axes
    fig.update_xaxes(title_text="k (number of extremes)", row=1, col=1)
    fig.update_yaxes(title_text="gamma_hat(k)", row=1, col=1)
    fig.update_xaxes(title_text="k", row=1, col=2)
    fig.update_yaxes(title_text="gamma_hat_Pickands(k)", row=1, col=2)
    fig.update_xaxes(title_text="log((k+1)/i)", row=2, col=1)
    fig.update_yaxes(title_text="log(Y_{n-i+1,n}/Y_{n-k,n})", row=2, col=1)
    fig.update_xaxes(title_text="y (log scale)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="F_bar(y) (log scale)", type="log", row=2, col=2)
    
    fig.update_layout(
        title_text=f"Test 1: Domaine d'Attraction de Fréchet - {ticker_name}",
        height=900,
        showlegend=True
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'hyp1_frechet_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")
    
    # we print gamma estimation result
    if mean_gamma is not None and mean_gamma > 0:
        print(f"  → gamma estimé: {mean_gamma:.4f} (doit être entre 0 et 1)")
    else:
        print(f"  ⚠ Impossible d'estimer gamma pour {ticker_name}")
    
    return mean_gamma

# %% [markdown]
# # 2. Test: Régime des Statistiques d'Ordre Intermédiaires

# %%
def test_intermediate_regime(Y_data, ticker_name, save_dir=None):
    """we test intermediate order statistics regime"""
    try:
        n = len(Y_data)
        if n < 10:
            print(f"Warning: Not enough data for {ticker_name} (n={n})")
            return 10
        
        k_max = min(n//2, 500)
        k_range = np.arange(5, k_max)
        k_over_n = k_range / n
    except Exception as e:
        print(f"Error in test_intermediate_regime for {ticker_name}: {e}")
        return 10
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Condition: k → ∞ (avec k < n)', 
                       'Condition: k/n → 0 (régime intermédiaire)')
    )
    
    # Plot 1: k vs n
    valid_k_max = int(0.2 * n)
    recommended_k = int(0.1 * n)
    
    fig.add_trace(
        go.Scatter(x=k_range, y=k_range, mode='lines', name='k',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_hline(y=n, line_dash="dash", line_color="red", 
                 annotation_text=f"n={n} (total sample)", row=1, col=1)
    fig.add_vline(x=valid_k_max, line_dash="dot", line_color="green", 
                 annotation_text=f"k < 0.2n = {valid_k_max}", row=1, col=1)
    fig.add_annotation(
        text=f"Valid range: k ∈ [5, {valid_k_max}]",
        xref="x domain", yref="y domain",
        x=0.05, y=0.95, showarrow=False,
        bgcolor="wheat", bordercolor="black",
        row=1, col=1
    )
    
    # Plot 2: k/n ratio
    fig.add_trace(
        go.Scatter(x=k_range, y=k_over_n, mode='lines', name='k/n',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", 
                 annotation_text="k/n = 0.2 (upper bound)", row=1, col=2)
    fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                 opacity=0.7, row=1, col=2)
    fig.add_vline(x=recommended_k, line_dash="dot", line_color="purple", 
                 annotation_text=f"Recommended k={recommended_k}", row=1, col=2)
    fig.add_annotation(
        text=f"Recommended: k ≈ {recommended_k} (k/n = {recommended_k/n:.3f})",
        xref="x domain", yref="y domain",
        x=0.05, y=0.95, showarrow=False,
        bgcolor="lightgreen", bordercolor="black",
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="k", row=1, col=2)
    fig.update_yaxes(title_text="k/n", row=1, col=2)
    
    fig.update_layout(
        title_text=f"Test 2: Régime Intermédiaire - {ticker_name}",
        height=500,
        showlegend=True
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'hyp2_intermediate_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")
    
    return recommended_k

# %% [markdown]
# # 3. Test: Condition de Biais Classique

# %%
def estimate_auxiliary_function_A(Y_sorted, k_values):
    """we estimate auxiliary function A(t) for second-order regular variation"""
    n = len(Y_sorted)
    A_estimates = np.zeros(len(k_values))
    
    try:
        for idx, k in enumerate(k_values):
            if k < n//4 and k > 10:
                t = n / k
                U_t = Y_sorted[n - k - 1] if n - k - 1 >= 0 else Y_sorted[0]
                U_2t = Y_sorted[n - k//2 - 1] if n - k//2 - 1 >= 0 else Y_sorted[0]
                
                if U_t > 0 and U_2t > 0:
                    if k > 20:
                        # we compute gamma estimate
                        log_diffs = []
                        for i in range(1, min(20, k)):
                            if n - i - 1 >= 0 and n - k - 1 >= 0:
                                log_diffs.append(np.log(Y_sorted[n-i-1]) - np.log(Y_sorted[n-k-1]))
                        if len(log_diffs) > 0:
                            gamma_est = np.mean(log_diffs)
                            if gamma_est > 0:
                                ratio = U_2t / U_t
                                theoretical_ratio = 2.0 ** gamma_est
                                H_rho_2 = np.log(2.0)
                                if H_rho_2 > 0 and theoretical_ratio > 0:
                                    A_est = (ratio - theoretical_ratio) / (theoretical_ratio * H_rho_2)
                                    A_estimates[idx] = A_est
    except Exception as e:
        print(f"Warning: Error in estimate_auxiliary_function_A: {e}")
        return np.zeros(len(k_values))
    
    return A_estimates

def test_bias_condition(Y_data, ticker_name, recommended_k, save_dir=None):
    """we test bias condition sqrt(k)*A(n/k) = O(1)"""
    try:
        Y_sorted = np.sort(Y_data)[::-1]
        n = len(Y_data)
        # we ensure k_range is not empty
        k_max = max(recommended_k*2, 30, n//4)
        k_range = np.arange(20, min(k_max, n//4))
        
        if len(k_range) == 0:
            # we use a default range if recommended_k is too small
            k_range = np.arange(20, min(50, n//4))
            if len(k_range) == 0:
                print(f"Warning: Empty k_range for {ticker_name} (n={n})")
                return None
        
        A_estimates = estimate_auxiliary_function_A(Y_sorted, k_range)
        sqrt_k = np.sqrt(k_range)
        bias_term = sqrt_k * np.abs(A_estimates)
        
        valid_mask = (A_estimates != 0) & np.isfinite(A_estimates) & np.isfinite(bias_term)
    except Exception as e:
        print(f"Error in test_bias_condition for {ticker_name}: {e}")
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Fonction Auxiliaire A(n/k)', 
                       'Condition: sqrt(k)*A(n/k) = O(1)')
    )
    
    if np.sum(valid_mask) > 5:
        # Plot 1: A(n/k)
        mean_A = np.mean(np.abs(A_estimates[valid_mask]))
        fig.add_trace(
            go.Scatter(x=k_range[valid_mask], y=A_estimates[valid_mask], mode='lines', 
                      name='A_hat(n/k)', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_color="black", opacity=0.3, row=1, col=1)
        fig.add_annotation(
            text=f"Mean |A|: {mean_A:.4f}",
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, showarrow=False,
            bgcolor="wheat", bordercolor="black",
            row=1, col=1
        )
        
        # Plot 2: sqrt(k)*A(n/k)
        max_bias = np.max(bias_term[valid_mask])
        condition_satisfied = max_bias < 5.0
        color = "lightgreen" if condition_satisfied else "lightcoral"
        
        fig.add_trace(
            go.Scatter(x=k_range[valid_mask], y=bias_term[valid_mask], mode='lines', 
                      name='sqrt(k)*|A_hat(n/k)|', line=dict(color='red', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", 
                     annotation_text="Bound: O(1) = 1", row=1, col=2)
        fig.add_hline(y=2.0, line_dash="dash", line_color="orange", 
                     opacity=0.7, row=1, col=2)
        fig.add_annotation(
            text=f"Max value: {max_bias:.3f}<br>Condition: {'✓ Satisfied' if condition_satisfied else '✗ Violated'}",
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, showarrow=False,
            bgcolor=color, bordercolor="black",
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_yaxes(title_text="A_hat(n/k)", row=1, col=1)
    fig.update_xaxes(title_text="k", row=1, col=2)
    fig.update_yaxes(title_text="sqrt(k)*|A_hat(n/k)|", row=1, col=2)
    
    fig.update_layout(
        title_text=f"Test 3: Condition de Biais - {ticker_name}",
        height=500,
        showlegend=True
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'hyp3_bias_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")
    
    return condition_satisfied if np.sum(valid_mask) > 5 else None

# %% [markdown]
# # 4. Test: Dominance de la Queue du Signal sur le Bruit

# %%
def estimate_kappa_from_model(X_data, Y_data, beta_hat, gamma_est):
    """we estimate kappa by fitting the inverse model X = g(Y)*beta + epsilon"""
    d = X_data.shape[1]
    proj_X = np.dot(X_data, beta_hat) / d
    
    sort_idx = np.argsort(Y_data)[::-1]
    Y_sorted = Y_data[sort_idx]
    proj_X_sorted = proj_X[sort_idx]
    
    k_tail = min(50, len(Y_data) // 5)
    Y_tail = Y_sorted[:k_tail]
    proj_X_tail = np.abs(proj_X_sorted[:k_tail])
    
    mask = (Y_tail > 0) & (proj_X_tail > 0)
    if np.sum(mask) > 10:
        log_Y = np.log(Y_tail[mask])
        log_proj = np.log(proj_X_tail[mask])
        slope, intercept, r_val, _, _ = linregress(log_Y, log_proj)
        return max(0.1, slope), r_val**2
    
    return None, None

def test_signal_dominance(X_data, Y_data, beta_hat, gamma_est, ticker_name, save_dir=None):
    """we test condition q*kappa*gamma > 1"""
    try:
        # we estimate kappa independently of gamma (it doesn't need gamma)
        kappa_est, r2 = estimate_kappa_from_model(X_data, Y_data, beta_hat, gamma_est)
        
        if kappa_est is None:
            print(f"Could not estimate kappa for {ticker_name}")
            return None, None
        
        # we only need gamma for the condition test, not for kappa estimation
        if gamma_est is None or gamma_est <= 0:
            print(f"Warning: Invalid gamma_est for {ticker_name}, cannot test q*kappa*gamma > 1")
            # we still return kappa_est even if we can't test the condition
            return kappa_est, None
    except Exception as e:
        print(f"Error in test_signal_dominance for {ticker_name}: {e}")
        return None, None
    
    q_values = np.array([2.5, 3.0, 4.0, 5.0])
    q_kappa_gamma = q_values * kappa_est * gamma_est
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Estimation de kappa: g(Y) ~ Y^kappa', 
                       'Condition: q*kappa*gamma > 1')
    )
    
    # Plot 1: Estimation of kappa
    d = X_data.shape[1]
    proj_X = np.dot(X_data, beta_hat) / d
    sort_idx = np.argsort(Y_data)[::-1]
    Y_sorted = Y_data[sort_idx]
    proj_X_sorted = proj_X[sort_idx]
    k_tail = min(50, len(Y_data) // 5)
    Y_tail = Y_sorted[:k_tail]
    proj_X_tail = np.abs(proj_X_sorted[:k_tail])
    mask = (Y_tail > 0) & (proj_X_tail > 0)
    
    if np.sum(mask) > 10:
        log_Y = np.log(Y_tail[mask])
        log_proj = np.log(proj_X_tail[mask])
        slope, intercept, _, _, _ = linregress(log_Y, log_proj)
        
        fig.add_trace(
            go.Scatter(x=log_Y, y=log_proj, mode='markers', name='Tail data',
                      marker=dict(size=5, opacity=0.6)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=log_Y, y=intercept + slope*log_Y, mode='lines', 
                      name=f'Fit: kappa = {slope:.3f} (R²={r2:.3f})',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
    
    # Plot 2: Condition q*kappa*gamma > 1
    colors_bar = ['red' if x < 1 else 'green' for x in q_kappa_gamma]
    fig.add_trace(
        go.Bar(x=[f'q={q}' for q in q_values], y=q_kappa_gamma, 
              marker_color=colors_bar, name='q*kappa*gamma'),
        row=1, col=2
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", 
                 annotation_text="Bound: q*kappa*gamma = 1", row=1, col=2)
    
    satisfied = np.sum(q_kappa_gamma > 1)
    fig.add_annotation(
        text=f"gamma = {gamma_est:.3f}<br>kappa = {kappa_est:.3f}<br>Satisfied for {satisfied}/{len(q_values)} values of q",
        xref="x domain", yref="y domain",
        x=0.05, y=0.95, showarrow=False,
        bgcolor="lightblue", bordercolor="black",
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="log(Y)", row=1, col=1)
    fig.update_yaxes(title_text="log(|<X, beta>|)", row=1, col=1)
    fig.update_xaxes(title_text="q", row=1, col=2)
    fig.update_yaxes(title_text="q*kappa*gamma", row=1, col=2)
    
    fig.update_layout(
        title_text=f"Test 4: Dominance Signal/Bruit - {ticker_name}",
        height=500,
        showlegend=True
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'hyp4_dominance_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")
    
    # we print kappa estimation result
    if kappa_est is not None:
        print(f"  → kappa estimé: {kappa_est:.4f} (doit être > 0)")
        if q_kappa_gamma is not None:
            satisfied_count = np.sum(q_kappa_gamma > 1)
            print(f"  → Condition q*kappa*gamma > 1: satisfaite pour {satisfied_count}/{len(q_values)} valeurs de q")
    else:
        print(f"  ⚠ Impossible d'estimer kappa pour {ticker_name}")
    
    return kappa_est, q_kappa_gamma

# %% [markdown]
# # 5. Test: Préservation de la Régularité en Queue (Variation Régulière)

# %%
def test_regular_variation(Y_data, X_data, beta_hat, tau_values, ticker_name, save_dir=None):
    """we test regular variation of g and phi"""
    try:
        d = X_data.shape[1]
        proj_X = np.dot(X_data, beta_hat) / d
        sort_idx = np.argsort(Y_data)[::-1]
        Y_sorted = Y_data[sort_idx]
        proj_X_sorted = proj_X[sort_idx]
        
        k_tail = min(100, len(Y_data) // 3)
        Y_tail = Y_sorted[:k_tail]
        proj_X_tail = np.abs(proj_X_sorted[:k_tail])
        mask = (Y_tail > 0) & (proj_X_tail > 0)
        
        if np.sum(mask) < 10:
            print(f"Warning: Not enough valid data for {ticker_name}")
            return None
    except Exception as e:
        print(f"Error in test_regular_variation for {ticker_name}: {e}")
        return None
    
    log_Y = np.log(Y_tail[mask])
    log_proj = np.log(proj_X_tail[mask])
    kappa_est, _ = linregress(log_Y, log_proj)[:2]
    slope, intercept, r_val, _, _ = linregress(log_Y, log_proj)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('g ∈ RV_kappa: Vérification g(Y) ~ Y^kappa', 
                       'phi ∈ RV_tau: phi(Y) = Y^tau',
                       'Composition: g ∘ phi (should be RV_{kappa+tau})',
                       'Summary')
    )
    
    # Plot 1: g(Y) regular variation
    fig.add_trace(
        go.Scatter(x=log_Y, y=log_proj, mode='markers', name='Data',
                  marker=dict(size=4, opacity=0.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=log_Y, y=intercept + slope*log_Y, mode='lines', 
                  name=f'g(Y) ~ Y^{{{slope:.3f}}} (R²={r_val**2:.3f})',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Plot 2: phi(Y) for different tau
    for tau in tau_values:
        phi_Y = Y_tail[mask] ** tau
        log_phi = np.log(phi_Y[phi_Y > 0])
        log_Y_phi = log_Y[phi_Y > 0]
        if len(log_phi) > 5:
            fig.add_trace(
                go.Scatter(x=log_Y_phi, y=log_phi, mode='markers', 
                          name=f'Données observées (tau={tau})', marker=dict(size=3, opacity=0.4)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=log_Y_phi, y=tau * log_Y_phi, mode='lines', 
                          name=f'Théorique: log(phi) = {tau}*log(Y)',
                          line=dict(dash='dash', width=1.5),
                          opacity=0.7,
                          showlegend=True),
                row=1, col=2
            )
    
    # Plot 3: Composition g(phi(Y))
    tau_test = 1.0
    phi_Y_test = Y_tail[mask] ** tau_test
    g_phi_Y = phi_Y_test ** kappa_est
    log_g_phi = np.log(g_phi_Y[g_phi_Y > 0])
    log_Y_comp = log_Y[g_phi_Y > 0]
    if len(log_g_phi) > 5:
        theoretical_slope = kappa_est * tau_test
        fig.add_trace(
            go.Scatter(x=log_Y_comp, y=log_g_phi, mode='markers', 
                      name='Données observées: g(phi(Y))',
                      marker=dict(size=4, opacity=0.5)),
            row=2, col=1
        )
        # we compute actual slope from data for comparison
        if len(log_Y_comp) > 5:
            actual_slope, _, _, _, _ = linregress(log_Y_comp, log_g_phi)
            fig.add_trace(
                go.Scatter(x=log_Y_comp, y=actual_slope * log_Y_comp, mode='lines', 
                          name=f'Fit observé: log(g∘phi) = {actual_slope:.3f}*log(Y)',
                          line=dict(color='blue', width=2)),
                row=2, col=1
            )
        # we compute actual slope from data for comparison
        if len(log_Y_comp) > 5:
            actual_slope, _, _, _, _ = linregress(log_Y_comp, log_g_phi)
            fig.add_trace(
                go.Scatter(x=log_Y_comp, y=actual_slope * log_Y_comp, mode='lines', 
                          name=f'Fit observé sur données: log(g∘phi) = {actual_slope:.3f}*log(Y)',
                          line=dict(color='blue', width=2)),
                row=2, col=1
            )
        fig.add_trace(
            go.Scatter(x=log_Y_comp, y=theoretical_slope * log_Y_comp, mode='lines', 
                      name=f'Théorique attendu: log(g∘phi) = {theoretical_slope:.3f}*log(Y) (kappa+tau={kappa_est+tau_test:.3f})',
                      line=dict(color='red', width=2, dash='dash')),
            row=2, col=1
        )
    
    # Plot 4: Summary
    summary_text = f"""
    Summary - Regular Variation:
    
    Estimated kappa (from g): {kappa_est:.3f}
    
    Tested tau values: {tau_values}
    
    Composition g ∘ phi:
    - Expected index: kappa + tau
    - For tau=1: kappa + 1 = {kappa_est + 1:.3f}
    
    Condition: Functions should be regularly varying
    with indices kappa > 0 and tau ∈ R
    """
    fig.add_annotation(
        text=summary_text,
        xref="x domain", yref="y domain",
        x=0.5, y=0.5, showarrow=False,
        bgcolor="wheat", bordercolor="black",
        row=2, col=2
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=2)
    
    fig.update_xaxes(title_text="log(Y)", row=1, col=1)
    fig.update_yaxes(title_text="log(|g(Y)|)", row=1, col=1)
    fig.update_xaxes(title_text="log(Y)", row=1, col=2)
    fig.update_yaxes(title_text="log(phi(Y))", row=1, col=2)
    fig.update_xaxes(title_text="log(Y)", row=2, col=1)
    fig.update_yaxes(title_text="log(g(phi(Y)))", row=2, col=1)
    
    fig.update_layout(
        title_text=f"Test 5: Variation Régulière - {ticker_name}",
        height=900,
        showlegend=True
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'hyp5_regular_var_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")
    
    return kappa_est

# %% [markdown]
# # 6. Test: Existence des Tail-Moments

# %%
def test_tail_moments_existence(gamma_est, kappa_est, tau_values, ticker_name, save_dir=None):
    """we test condition 2*(kappa+tau)*gamma < 1"""
    try:
        if gamma_est is None or kappa_est is None:
            print(f"Warning: Missing gamma or kappa for {ticker_name}")
            return None, None
        
        condition_values = 2 * (kappa_est + np.array(tau_values)) * gamma_est
    except Exception as e:
        print(f"Error in test_tail_moments_existence for {ticker_name}: {e}")
        return None, None
    
    fig = go.Figure()
    
    colors_bar = ['green' if x < 1 else 'red' for x in condition_values]
    fig.add_trace(
        go.Bar(x=[f'tau={tau}' for tau in tau_values], y=condition_values, 
              marker_color=colors_bar, name='2*(kappa+tau)*gamma')
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", 
                 annotation_text="Bound: 2*(kappa+tau)*gamma = 1")
    
    satisfied = np.sum(condition_values < 1)
    summary_text = f"""
    gamma = {gamma_est:.3f}
    kappa = {kappa_est:.3f}
    
    Condition satisfied for {satisfied}/{len(tau_values)} values of tau
    
    Valid tau range: tau < 1/(2*gamma) - kappa = {0.5/gamma_est - kappa_est:.3f}
    """
    fig.add_annotation(
        text=summary_text,
        xref="x domain", yref="y domain",
        x=0.02, y=0.98, showarrow=False,
        bgcolor="lightblue", bordercolor="black",
        align="left"
    )
    
    fig.update_xaxes(title_text="tau")
    fig.update_yaxes(title_text="2*(kappa+tau)*gamma")
    fig.update_layout(
        title_text=f"Test 6: Existence des Tail-Moments - {ticker_name}",
        height=600,
        showlegend=False
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'hyp6_tail_moments_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")
    
    return condition_values, tau_values[condition_values < 1]

# %% [markdown]
# # 7. Conditional Quantile Plot with Scatter

# %%
def plot_conditional_quantile_with_scatter(X_data, Y_data, beta_hat, best_k, ticker_name, save_dir=None):
    """we plot conditional quantile with scatter of extreme and non-extreme points"""
    try:
        from adapted import plot_quantile_conditional_on_sample_new
        
        X_fepls = np.expand_dims(X_data, axis=0)
        Y_fepls = np.expand_dims(Y_data, axis=0)
        n_samples = Y_fepls.shape[1]
        d_points = X_fepls.shape[2]
        
        proj_vals = np.dot(X_data, beta_hat) / d_points
        
        h_univ = 0.2 * np.std(proj_vals)
        if h_univ < 1e-6:
            h_univ = 1e-6
        h_func = 0.2 * np.mean(np.std(X_data, axis=0))
        if h_func < 1e-6:
            h_func = 1e-6
        
        h_univ_vec = h_univ * np.ones(n_samples)
        h_func_vec = h_func * np.ones(n_samples)
        
        quantiles, s_grid = plot_quantile_conditional_on_sample_new(
            X_fepls, Y_fepls,
            dimred=beta_hat,
            x_func=beta_hat,
            alpha=0.95,
            h_univ_vector=h_univ_vec,
            h_func_vector=h_func_vec
        )
    except Exception as e:
        print(f"Error in plot_conditional_quantile_with_scatter for {ticker_name}: {e}")
        return
    
    Y_sorted_idx = np.argsort(Y_data)[::-1]
    extreme_threshold = Y_data[Y_sorted_idx[best_k]] if best_k < len(Y_data) else np.median(Y_data)
    is_extreme = Y_data >= extreme_threshold
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Conditional 95% Quantile with Scatter', 
                       'Distribution of Projections',
                       'Y vs Projection (Color-coded by Y)', 
                       'Conditional Quantiles at Different Levels')
    )
    
    # Plot 1: Conditional Quantile with Scatter
    fig.add_trace(
        go.Scatter(x=proj_vals[~is_extreme], y=Y_data[~is_extreme], mode='markers', 
                  name='Non-extreme', marker=dict(size=4, color='blue', opacity=0.4)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=proj_vals[is_extreme], y=Y_data[is_extreme], mode='markers', 
                  name='Extreme', marker=dict(size=5, color='red', opacity=0.7)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=s_grid, y=quantiles[:, 0], mode='lines', 
                  name='Univariate (95%)', line=dict(color='green', width=2, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=s_grid, y=quantiles[:, 1], mode='lines', 
                  name='Functional (95%)', line=dict(color='purple', width=2)),
        row=1, col=1
    )
    
    # Plot 2: Distribution of projections
    fig.add_trace(
        go.Histogram(x=proj_vals[~is_extreme], name='Non-extreme', opacity=0.5, 
                    marker_color='blue', histnorm='probability density'),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=proj_vals[is_extreme], name='Extreme', opacity=0.7, 
                    marker_color='red', histnorm='probability density'),
        row=1, col=2
    )
    
    # Plot 3: Y vs Projection (all points)
    fig.add_trace(
        go.Scatter(x=proj_vals, y=Y_data, mode='markers', 
                  marker=dict(size=4, color=Y_data, colorscale='Viridis', 
                            showscale=True, colorbar=dict(title="Y value")),
                  name='All points'),
        row=2, col=1
    )
    
    # Plot 4: Tail behavior comparison
    alpha_values = [0.90, 0.95, 0.99]
    colors_alpha = ['orange', 'green', 'red']
    for alpha, color in zip(alpha_values, colors_alpha):
        quantiles_alpha, _ = plot_quantile_conditional_on_sample_new(
            X_fepls, Y_fepls,
            dimred=beta_hat,
            x_func=beta_hat,
            alpha=alpha,
            h_univ_vector=h_univ_vec,
            h_func_vector=h_func_vec
        )
        fig.add_trace(
            go.Scatter(x=s_grid, y=quantiles_alpha[:, 1], mode='lines', 
                      name=f'Functional ({int(alpha*100)}%)',
                      line=dict(color=color, width=2, dash='dash')),
            row=2, col=2
        )
    fig.add_trace(
        go.Scatter(x=proj_vals[is_extreme], y=Y_data[is_extreme], mode='markers', 
                  name='Extreme points', marker=dict(size=4, color='red', opacity=0.6)),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="<X, beta_hat>", row=1, col=1)
    fig.update_yaxes(title_text="Y (Response)", row=1, col=1)
    fig.update_xaxes(title_text="<X, beta_hat>", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    fig.update_xaxes(title_text="<X, beta_hat>", row=2, col=1)
    fig.update_yaxes(title_text="Y (Response)", row=2, col=1)
    fig.update_xaxes(title_text="<X, beta_hat>", row=2, col=2)
    fig.update_yaxes(title_text="Y (Response)", row=2, col=2)
    
    fig.update_layout(
        title_text=f"Conditional Quantile Analysis - {ticker_name}",
        height=900,
        showlegend=True
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'conditional_quantile_scatter_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")

# %% [markdown]
# # 8. Additional Diagnostic Plots

# %%
def plot_additional_diagnostics(X_data, Y_data, beta_hat, gamma_est, kappa_est, ticker_name, save_dir=None):
    """we create additional diagnostic plots"""
    try:
        d = X_data.shape[1]
        proj_X = np.dot(X_data, beta_hat) / d
        n = len(Y_data)
    except Exception as e:
        print(f"Error in plot_additional_diagnostics for {ticker_name}: {e}")
        return
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('FEPLS Direction Function', 
                       'Projection vs Y (correlation)',
                       'Hill Estimator Stability',
                       'Distribution of Y',
                       'Survival Function (Log-Log)',
                       'Summary Statistics')
    )
    
    # Plot 1: Beta function
    time_grid = np.linspace(0, 1, len(beta_hat))
    fig.add_trace(
        go.Scatter(x=time_grid, y=beta_hat, mode='lines', name='beta_hat(t)',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Plot 2: Correlation between projection and Y
    corr_coef = np.corrcoef(proj_X, Y_data)[0, 1]
    z = np.polyfit(proj_X, Y_data, 1)
    p = np.poly1d(z)
    
    fig.add_trace(
        go.Scatter(x=proj_X, y=Y_data, mode='markers', name='Data',
                  marker=dict(size=4, opacity=0.5)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=proj_X, y=p(proj_X), mode='lines', 
                  name=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=2
    )
    
    # Plot 3: Tail index stability
    Y_sorted = np.sort(Y_data)[::-1]
    k_range = np.arange(10, min(100, n//5))
    hill_est = get_hill_estimator(Y_sorted)
    fig.add_trace(
        go.Scatter(x=k_range, y=hill_est[9:len(k_range)+9], mode='lines', 
                  name='Hill Estimator', line=dict(color='blue', width=2)),
        row=1, col=3
    )
    if gamma_est is not None:
        fig.add_hline(y=gamma_est, line_dash="dash", line_color="red", 
                     annotation_text=f"Estimated gamma={gamma_est:.3f}", row=1, col=3)
    
    # Plot 4: Distribution of Y
    fig.add_trace(
        go.Histogram(x=Y_data, name='Y distribution', opacity=0.7, 
                    marker_color='skyblue', histnorm='probability density'),
        row=2, col=1
    )
    fig.add_vline(x=np.median(Y_data), line_dash="dash", line_color="red", 
                 annotation_text=f"Median={np.median(Y_data):.3f}", row=2, col=1)
    fig.add_vline(x=np.mean(Y_data), line_dash="dash", line_color="green", 
                 annotation_text=f"Mean={np.mean(Y_data):.3f}", row=2, col=1)
    
    # Plot 5: Log-log plot of Y distribution
    Y_sorted_asc = np.sort(Y_data)
    y_thresholds = Y_sorted_asc[::-1][:min(200, n)]
    empirical_survival = np.arange(1, len(y_thresholds)+1) / n
    mask = (y_thresholds > 0) & (empirical_survival > 0)
    if np.sum(mask) > 10:
        fig.add_trace(
            go.Scatter(x=y_thresholds[mask], y=empirical_survival[mask], mode='lines', 
                      name='F_bar(y)', line=dict(color='blue', width=2)),
            row=2, col=2
        )
    
    # Plot 6: Summary statistics
    summary_text = f"""
    Summary Statistics:
    
    Sample size: n = {n}
    Dimension: d = {d}
    
    Y Statistics:
    - Mean: {np.mean(Y_data):.4f}
    - Median: {np.median(Y_data):.4f}
    - Std: {np.std(Y_data):.4f}
    - Min: {np.min(Y_data):.4f}
    - Max: {np.max(Y_data):.4f}
    
    Estimated Parameters:
    - gamma (tail index): {gamma_est:.4f if gamma_est is not None else 'N/A'}
    - kappa (link function): {kappa_est:.4f if kappa_est is not None else 'N/A'}
    
    Correlation:
    - rho(<X, beta_hat>, Y): {corr_coef:.4f}
    """
    fig.add_annotation(
        text=summary_text,
        xref="x domain", yref="y domain",
        x=0.5, y=0.5, showarrow=False,
        bgcolor="wheat", bordercolor="black",
        align="left",
        row=2, col=3
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=3)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=3)
    
    fig.update_xaxes(title_text="Time (normalized)", row=1, col=1)
    fig.update_yaxes(title_text="beta_hat(t)", row=1, col=1)
    fig.update_xaxes(title_text="<X, beta_hat>", row=1, col=2)
    fig.update_yaxes(title_text="Y", row=1, col=2)
    fig.update_xaxes(title_text="k", row=1, col=3)
    fig.update_yaxes(title_text="gamma_hat(k)", row=1, col=3)
    fig.update_xaxes(title_text="Y", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_xaxes(title_text="y (log scale)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="F_bar(y) (log scale)", type="log", row=2, col=2)
    
    fig.update_layout(
        title_text=f"Additional Diagnostics - {ticker_name}",
        height=900,
        showlegend=True
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'additional_diagnostics_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")

# %% [markdown]
# # 9. Test Complet de Toutes les Hypothèses

# %%
def test_all_hypotheses_comprehensive(Y_data, X_data, beta_hat, gamma_est, kappa_est, 
                                      recommended_k, tau_values, q_values, ticker_name, save_dir=None):
    """we test all hypotheses from the paper comprehensively"""
    
    results = {
        'ticker': ticker_name,
        'gamma': gamma_est,
        'kappa': kappa_est,
        'recommended_k': recommended_k,
        'n': len(Y_data),
        'hypotheses': {}
    }
    
    # we create comprehensive figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Hyp 1: 0 < gamma < 1 (Queue lourde intégrable)',
            'Hyp 2: kappa > 0 (Variation régulière de g)',
            'Hyp 3: 0 < 2(kappa+tau)*gamma < 1 (Existence tail-moments)',
            'Hyp 4: q*kappa*gamma > 1 (Dominance signal/bruit)',
            'Hyp 5: sqrt(k)*A(n/k) = O(1) (Condition de biais)',
            'Hyp 6: k → ∞ et k/n → 0 (Régime intermédiaire)'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Hypothesis 1: 0 < gamma < 1
    if gamma_est is not None:
        hyp1_satisfied = 0 < gamma_est < 1
        results['hypotheses']['H1_gamma_range'] = {
            'satisfied': hyp1_satisfied,
            'value': gamma_est,
            'condition': '0 < gamma < 1'
        }
        
        fig.add_trace(
            go.Bar(x=['gamma'], y=[gamma_est], 
                  marker_color='green' if hyp1_satisfied else 'red',
                  name='gamma'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_color="black", row=1, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Upper bound: gamma=1", row=1, col=1)
        fig.add_annotation(
            text=f"gamma = {gamma_est:.4f}<br>{'✓ Satisfied' if hyp1_satisfied else '✗ Violated'}",
            xref="x domain", yref="y domain",
            x=0.5, y=0.95, showarrow=False,
            bgcolor="lightgreen" if hyp1_satisfied else "lightcoral",
            row=1, col=1
        )
    else:
        results['hypotheses']['H1_gamma_range'] = {'satisfied': False, 'value': None}
    
    # Hypothesis 2: kappa > 0
    if kappa_est is not None:
        hyp2_satisfied = kappa_est > 0
        results['hypotheses']['H2_kappa_positive'] = {
            'satisfied': hyp2_satisfied,
            'value': kappa_est,
            'condition': 'kappa > 0'
        }
        
        fig.add_trace(
            go.Bar(x=['kappa'], y=[kappa_est], 
                  marker_color='green' if hyp2_satisfied else 'red',
                  name='kappa'),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                     annotation_text="Borne inférieure: kappa=0", row=1, col=2)
        fig.add_annotation(
            text=f"kappa = {kappa_est:.4f}<br>{'✓ Satisfied' if hyp2_satisfied else '✗ Violated'}",
            xref="x domain", yref="y domain",
            x=0.5, y=0.95, showarrow=False,
            bgcolor="lightgreen" if hyp2_satisfied else "lightcoral",
            row=1, col=2
        )
    else:
        results['hypotheses']['H2_kappa_positive'] = {'satisfied': False, 'value': None}
    
    # Hypothesis 3: 0 < 2(kappa+tau)*gamma < 1
    if gamma_est is not None and kappa_est is not None:
        condition_values = 2 * (kappa_est + np.array(tau_values)) * gamma_est
        hyp3_satisfied = np.all((condition_values > 0) & (condition_values < 1))
        results['hypotheses']['H3_tail_moments'] = {
            'satisfied': hyp3_satisfied,
            'values': condition_values.tolist(),
            'tau_values': tau_values,
            'condition': '0 < 2(kappa+tau)*gamma < 1'
        }
        
        colors_bar = ['green' if (0 < x < 1) else 'red' for x in condition_values]
        fig.add_trace(
            go.Bar(x=[f'tau={tau}' for tau in tau_values], y=condition_values, 
                  marker_color=colors_bar, name='2*(kappa+tau)*gamma'),
            row=2, col=1
        )
        fig.add_hline(y=0, line_color="black", row=2, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Borne supérieure: 1", row=2, col=1)
        satisfied_count = np.sum((condition_values > 0) & (condition_values < 1))
        fig.add_annotation(
            text=f"Satisfied for {satisfied_count}/{len(tau_values)} values of tau",
            xref="x domain", yref="y domain",
            x=0.5, y=0.95, showarrow=False,
            bgcolor="lightgreen" if hyp3_satisfied else "lightcoral",
            row=2, col=1
        )
    else:
        results['hypotheses']['H3_tail_moments'] = {'satisfied': False, 'values': None}
    
    # Hypothesis 4: q*kappa*gamma > 1
    if gamma_est is not None and kappa_est is not None:
        q_kappa_gamma = q_values * kappa_est * gamma_est
        hyp4_satisfied = np.all(q_kappa_gamma > 1)
        results['hypotheses']['H4_signal_dominance'] = {
            'satisfied': hyp4_satisfied,
            'values': q_kappa_gamma.tolist(),
            'q_values': q_values.tolist(),
            'condition': 'q*kappa*gamma > 1'
        }
        
        colors_bar = ['green' if x > 1 else 'red' for x in q_kappa_gamma]
        fig.add_trace(
            go.Bar(x=[f'q={q}' for q in q_values], y=q_kappa_gamma, 
                  marker_color=colors_bar, name='q*kappa*gamma'),
            row=2, col=2
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Borne inférieure: 1", row=2, col=2)
        satisfied_count = np.sum(q_kappa_gamma > 1)
        fig.add_annotation(
            text=f"Satisfied for {satisfied_count}/{len(q_values)} values of q",
            xref="x domain", yref="y domain",
            x=0.5, y=0.95, showarrow=False,
            bgcolor="lightgreen" if hyp4_satisfied else "lightcoral",
            row=2, col=2
        )
    else:
        results['hypotheses']['H4_signal_dominance'] = {'satisfied': False, 'values': None}
    
    # Hypothesis 5: sqrt(k)*A(n/k) = O(1)
    try:
        Y_sorted = np.sort(Y_data)[::-1]
        n = len(Y_data)
        k_range = np.arange(20, min(recommended_k*2, n//4))
        if len(k_range) > 0:
            A_estimates = estimate_auxiliary_function_A(Y_sorted, k_range)
            sqrt_k = np.sqrt(k_range)
            bias_term = sqrt_k * np.abs(A_estimates)
            valid_mask = (A_estimates != 0) & np.isfinite(A_estimates) & np.isfinite(bias_term)
            
            if np.sum(valid_mask) > 5:
                max_bias = np.max(bias_term[valid_mask])
                hyp5_satisfied = max_bias < 5.0  # we use relaxed bound
                results['hypotheses']['H5_bias_condition'] = {
                    'satisfied': hyp5_satisfied,
                    'max_value': float(max_bias),
                    'condition': 'sqrt(k)*A(n/k) = O(1)'
                }
                
                fig.add_trace(
                    go.Scatter(x=k_range[valid_mask], y=bias_term[valid_mask], 
                             mode='lines', name='sqrt(k)*|A(n/k)|',
                             line=dict(color='red', width=2)),
                    row=3, col=1
                )
                fig.add_hline(y=1, line_dash="dash", line_color="green", 
                             annotation_text="Borne théorique: O(1) = 1", row=3, col=1)
                fig.add_hline(y=5, line_dash="dash", line_color="orange", 
                             annotation_text="Borne relâchée: 5", row=3, col=1)
                fig.add_annotation(
                    text=f"Max value: {max_bias:.3f}<br>{'✓ Satisfied' if hyp5_satisfied else '✗ Violated'}",
                    xref="x domain", yref="y domain",
                    x=0.05, y=0.95, showarrow=False,
                    bgcolor="lightgreen" if hyp5_satisfied else "lightcoral",
                    row=3, col=1
                )
            else:
                results['hypotheses']['H5_bias_condition'] = {'satisfied': None, 'max_value': None}
        else:
            results['hypotheses']['H5_bias_condition'] = {'satisfied': None, 'max_value': None}
    except Exception as e:
        print(f"Warning: Could not test H5 for {ticker_name}: {e}")
        results['hypotheses']['H5_bias_condition'] = {'satisfied': None, 'error': str(e)}
    
    # Hypothesis 6: k → ∞ et k/n → 0
    n = len(Y_data)
    k_max = min(n//2, 500)
    k_range = np.arange(5, k_max)
    k_over_n = k_range / n
    valid_k_max = int(0.2 * n)
    hyp6_satisfied = (recommended_k < n) and (recommended_k / n < 0.2) and (recommended_k > 5)
    results['hypotheses']['H6_intermediate_regime'] = {
        'satisfied': hyp6_satisfied,
        'k': recommended_k,
        'k/n': recommended_k / n if n > 0 else 0,
        'condition': 'k → ∞ et k/n → 0'
    }
    
    fig.add_trace(
        go.Scatter(x=k_range, y=k_over_n, mode='lines', name='k/n',
                  line=dict(color='green', width=2)),
        row=3, col=2
    )
    fig.add_vline(x=recommended_k, line_dash="dot", line_color="purple", 
                 annotation_text=f"k={recommended_k}", row=3, col=2)
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", 
                 annotation_text="Borne supérieure: k/n = 0.2", row=3, col=2)
    fig.add_annotation(
        text=f"k = {recommended_k}, k/n = {recommended_k/n:.4f}<br>{'✓ Satisfied' if hyp6_satisfied else '✗ Violated'}",
        xref="x domain", yref="y domain",
        x=0.05, y=0.95, showarrow=False,
        bgcolor="lightgreen" if hyp6_satisfied else "lightcoral",
        row=3, col=2
    )
    
    # we update axes
    fig.update_xaxes(title_text="Parameter", row=1, col=1)
    fig.update_yaxes(title_text="gamma", row=1, col=1)
    fig.update_xaxes(title_text="Parameter", row=1, col=2)
    fig.update_yaxes(title_text="kappa", row=1, col=2)
    fig.update_xaxes(title_text="tau", row=2, col=1)
    fig.update_yaxes(title_text="2*(kappa+tau)*gamma", row=2, col=1)
    fig.update_xaxes(title_text="q", row=2, col=2)
    fig.update_yaxes(title_text="q*kappa*gamma", row=2, col=2)
    fig.update_xaxes(title_text="k", row=3, col=1)
    fig.update_yaxes(title_text="sqrt(k)*|A(n/k)|", row=3, col=1)
    fig.update_xaxes(title_text="k", row=3, col=2)
    fig.update_yaxes(title_text="k/n", row=3, col=2)
    
    fig.update_layout(
        title_text=f"Test Complet de Toutes les Hypothèses - {ticker_name}",
        height=1200,
        showlegend=True
    )
    
    try:
        fig.show()
        if save_dir:
            fig.write_html(os.path.join(save_dir, f'all_hypotheses_{ticker_name}.html'))
    except Exception as e:
        print(f"Warning: Could not display/save figure for {ticker_name}: {e}")
    
    # we print summary in a clear format
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ COMPLET DES HYPOTHÈSES POUR {ticker_name}")
    print(f"{'='*80}")
    print(f"Paramètres estimés:")
    print(f"  - gamma (indice de queue): {gamma_est:.4f if gamma_est is not None else 'NON ESTIMÉ'}")
    print(f"  - kappa (fonction de lien): {kappa_est:.4f if kappa_est is not None else 'NON ESTIMÉ'}")
    print(f"  - k recommandé: {recommended_k}")
    print(f"\nStatut des hypothèses:")
    
    for hyp_name, hyp_data in results['hypotheses'].items():
        if 'satisfied' in hyp_data:
            if hyp_data['satisfied'] is True:
                status = "✓ SATISFAIT"
                color_indicator = "[OK]"
            elif hyp_data['satisfied'] is False:
                status = "✗ VIOLÉ"
                color_indicator = "[ERREUR]"
            else:
                status = "? NON TESTÉ"
                color_indicator = "[SKIP]"
            
            print(f"\n{color_indicator} {hyp_name}: {hyp_data.get('condition', 'N/A')}")
            print(f"   Statut: {status}")
            
            if 'value' in hyp_data and hyp_data['value'] is not None:
                print(f"   Valeur estimée: {hyp_data['value']:.4f}")
            
            if 'values' in hyp_data and hyp_data['values'] is not None:
                print(f"   Valeurs: {[f'{v:.4f}' for v in hyp_data['values']]}")
                if 'tau_values' in hyp_data:
                    print(f"   Pour tau = {hyp_data['tau_values']}")
    
    print(f"\n{'='*80}\n")
    
    return results

# %% [markdown]
# # Exécution Complète des Tests

# %%
# we create output directory
SAVE_DIR = "../../results/hypothesis_verification/"
os.makedirs(SAVE_DIR, exist_ok=True)

# we define pairs to test
pairs_to_test = [
    ('mol.hu.txt', 'otp.hu.txt'),
    ('otp.hu.txt', 'mol.hu.txt'),
]

tau_values = [-2, -1, 0, 1, 2]

results_summary = []

for ticker_X, ticker_Y in pairs_to_test:
    if ticker_X not in func_data or ticker_Y not in func_data:
        continue
    
    print(f"\n{'='*80}")
    print(f"Testing: {ticker_X[:-7]} (X) -> {ticker_Y[:-7]} (Y)")
    print(f"{'='*80}\n")
    
    # we align data
    common_dates = func_data[ticker_X]['dates'].intersection(func_data[ticker_Y]['dates'])
    idx_X = func_data[ticker_X]['dates'].isin(common_dates)
    idx_Y = func_data[ticker_Y]['dates'].isin(common_dates)
    
    X_data = func_data[ticker_X]['curves'][idx_X]
    Y_data = func_data[ticker_Y]['max_return'][idx_Y]
    
    # we compute FEPLS beta_hat (simplified)
    from adapted import fepls
    X_fepls = np.expand_dims(X_data, axis=0)
    Y_fepls = np.expand_dims(Y_data, axis=0)
    n_samples = Y_fepls.shape[1]
    Y_sorted = np.sort(Y_fepls[0])[::-1]
    best_k = min(50, n_samples // 10)
    y_n = Y_sorted[best_k]
    y_matrix = y_n * np.ones_like(Y_fepls)
    E0 = fepls(X_fepls, Y_fepls, y_matrix, tau=1.0)
    beta_hat = E0[0,:]
    
    pair_name = f"{ticker_X[:-7]}_{ticker_Y[:-7]}"
    
    # Test 1: Fréchet Domain
    print("Test 1: Domaine d'Attraction de Fréchet...")
    try:
        gamma_est = test_frechet_domain(Y_data, pair_name, SAVE_DIR)
    except Exception as e:
        print(f"  ⚠ Skipped Test 1 due to error: {e}")
        gamma_est = None
    
    # Test 2: Intermediate Regime
    print("Test 2: Régime Intermédiaire...")
    try:
        recommended_k = test_intermediate_regime(Y_data, pair_name, SAVE_DIR)
    except Exception as e:
        print(f"  ⚠ Skipped Test 2 due to error: {e}")
        recommended_k = 30
    
    # Test 3: Bias Condition
    print("Test 3: Condition de Biais...")
    try:
        bias_ok = test_bias_condition(Y_data, pair_name, recommended_k, SAVE_DIR)
    except Exception as e:
        print(f"  ⚠ Skipped Test 3 due to error: {e}")
        bias_ok = None
    
    # Test 4: Signal Dominance
    print("Test 4: Dominance Signal/Bruit...")
    try:
        kappa_est, q_kappa_gamma = test_signal_dominance(X_data, Y_data, beta_hat, gamma_est, pair_name, SAVE_DIR)
    except Exception as e:
        print(f"  ⚠ Skipped Test 4 due to error: {e}")
        kappa_est, q_kappa_gamma = None, None
    
    # Test 5: Regular Variation
    print("Test 5: Variation Régulière...")
    try:
        kappa_est2 = test_regular_variation(Y_data, X_data, beta_hat, tau_values, pair_name, SAVE_DIR)
        if kappa_est2 is not None:
            kappa_est = kappa_est2
    except Exception as e:
        print(f"  ⚠ Skipped Test 5 due to error: {e}")
    
    # Test 6: Tail Moments
    if gamma_est is not None and kappa_est is not None:
        print("Test 6: Existence des Tail-Moments...")
        try:
            condition_vals, valid_tau = test_tail_moments_existence(gamma_est, kappa_est, tau_values, pair_name, SAVE_DIR)
        except Exception as e:
            print(f"  ⚠ Skipped Test 6 due to error: {e}")
            condition_vals, valid_tau = None, None
        
        # Additional Plot 7: Conditional Quantile with Scatter
        print("Plot 7: Conditional Quantile with Scatter...")
        try:
            plot_conditional_quantile_with_scatter(X_data, Y_data, beta_hat, best_k, pair_name, SAVE_DIR)
        except Exception as e:
            print(f"  ⚠ Skipped Plot 7 due to error: {e}")
        
        # Additional Plot 8: Additional Diagnostics
        print("Plot 8: Additional Diagnostics...")
        try:
            plot_additional_diagnostics(X_data, Y_data, beta_hat, gamma_est, kappa_est, pair_name, SAVE_DIR)
        except Exception as e:
            print(f"  ⚠ Skipped Plot 8 due to error: {e}")
        
        # Test 9: Comprehensive Test of All Hypotheses
        print("Test 9: Test Complet de Toutes les Hypothèses...")
        try:
            q_values = np.array([2.5, 3.0, 4.0, 5.0])
            all_hyp_results = test_all_hypotheses_comprehensive(
                Y_data, X_data, beta_hat, gamma_est, kappa_est,
                recommended_k, tau_values, q_values, pair_name, SAVE_DIR
            )
        except Exception as e:
            print(f"  ⚠ Skipped Test 9 due to error: {e}")
            all_hyp_results = None
    else:
        print("  ⚠ Skipped Tests 6-9: Missing gamma or kappa estimates")
        condition_vals, valid_tau = None, None
        all_hyp_results = None
        
        # we store results
        results_summary.append({
            'Pair': pair_name,
            'gamma': gamma_est,
            'kappa': kappa_est,
            'recommended_k': recommended_k,
            'bias_condition': bias_ok,
            'q_kappa_gamma_min': np.min(q_kappa_gamma) if q_kappa_gamma is not None else None,
            'valid_tau': list(valid_tau) if valid_tau is not None else [],
            'n_samples': n_samples
        })

# we save summary
if results_summary:
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv(os.path.join(SAVE_DIR, 'hypothesis_verification_summary.csv'), index=False)
    print(f"\n{'='*80}")
    print("Summary saved to:", os.path.join(SAVE_DIR, 'hypothesis_verification_summary.csv'))
    print(f"{'='*80}\n")
    print(df_summary.to_string())

print("\nAll hypothesis tests completed!")
