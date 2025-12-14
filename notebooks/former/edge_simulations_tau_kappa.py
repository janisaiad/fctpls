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

# %% [markdown]
# # Edge-of-Inequality Simulations: Optimal k and Variance Analysis
# 
# This script generates Monte Carlo samples for fixed q and gamma, varying rho,
# then processes them for different tau and kappa values to study optimal k and variance.
# 
# Strategy: Generate (Y, Z) once per rho (expensive), then generate X on-the-fly for each kappa.

# %%
###################### Packages
import numpy as np
import numpy.random as npr
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import warnings
import numba
warnings.filterwarnings('ignore')

# we configure matplotlib settings
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'normal'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits'] = (-6, 6)
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

# %%
###################### Function Implementations

@numba.njit(parallel=True, fastmath=False)
def Burr_quantile_function(x, theta, rho):
    """Burr quantile function."""
    return ((1-x)**(-1/theta)-1)**(1/rho)

@numba.njit(parallel=True, fastmath=False)
def beta_func(d):
    """Beta function for index vector."""
    grid = np.linspace(0, 1, d)
    norm = np.sqrt(np.sum(np.sin(2*np.pi*grid)**2)/d)
    return np.sin(2*np.pi*grid)/norm

@numba.njit(parallel=True, fastmath=False)
def sigma(u, c, snr):
    """Sigma function for noise variance."""
    return (u**c)/snr

@numba.njit(parallel=True, fastmath=False)
def noise_mean(d, mu):
    """Noise mean function."""
    grid = np.linspace(0, 1, d)
    return mu*grid

@numba.njit(parallel=True, fastmath=False)
def sort_2d_array(x):
    """Sort each row of a 2D array."""
    n, m = np.shape(x)
    for row in numba.prange(n):
        x[row] = np.sort(x[row])
    return x

@numba.njit(parallel=True, fastmath=False)
def coeurjolly_cholesky_fbm_var(Y, Z, H, c, snr, mu):
    """Generate fBm noise that depends on Y and kappa (c)."""
    N = Y.shape[0]
    n = Y.shape[1]
    d = Z.shape[2] + 1
    out = np.zeros((N, n, d))
    H2 = 2 * H
    for p in numba.prange(N):
        for q in numba.prange(n):
            matcov = np.zeros((d-1, d-1))
            for i in numba.prange(d-1):
                for j in numba.prange(i, d-1):
                    r = (sigma(Y[p, q], c, snr)**2) * (1/2) * (abs(i+1)**H2 + abs(j+1)**H2 - abs(j - i)**H2)
                    r = r/(d**H2)
                    matcov[i, j] = r
                    matcov[j, i] = matcov[i, j]
            L = np.linalg.cholesky(matcov)
            fBm = np.dot(L, Z[p, q, :])
            out[p, q, :] = np.asarray([0] + list(fBm)) + noise_mean(d, mu)
    return out

def fepls(X, Y, y_matrix, tau):
    """FEPLS estimator."""
    N = X.shape[0]
    n = X.shape[1]
    d = X.shape[2]
    out = np.zeros((N, d))
    for j in range(d):
        aux = np.multiply(X[:, :, j], Y**tau)
        out2 = np.multiply(aux, np.greater_equal(Y, y_matrix))
        out[:, j] = np.sum(out2, axis=1)/n
    norms = np.sqrt(np.sum(out**2, axis=1)/d)
    out2 = out * (norms.reshape((norms.size, 1)))**(-1)
    return out2

@numba.njit(parallel=True, fastmath=False)
def fepls_numba(X, Y, y_matrix, tau):
    """FEPLS estimator (numba version)."""
    N = X.shape[0]
    n = X.shape[1]
    d = X.shape[2]
    out = np.zeros((N, d))
    for j in numba.prange(d):
        aux = np.multiply(X[:, :, j], Y**tau)
        out2 = np.multiply(aux, np.greater_equal(Y, y_matrix))
        out[:, j] = np.sum(out2, axis=1)/n
    norms = np.sqrt(np.sum(out**2, axis=1)/d)
    out2 = out * (norms.reshape((norms.size, 1)))**(-1)
    return out2

@numba.njit(parallel=True, fastmath=False)
def concomittant_corr(X, Y, Y_sort_index, tau, m):
    """Compute correlation for different k values."""
    N = X.shape[0]
    n = X.shape[1]
    d = X.shape[2]
    out = np.zeros((N, m))
    YY = np.copy(Y)
    Y_sort = sort_2d_array(YY)
    for k in numba.prange(m):
        y_array = np.zeros((N, n, k+1))
        aux = np.zeros((N, k+1))
        aux2 = np.zeros((N, k+1))
        aux3 = Y_sort[:, n-k-1:]
        aux3_sum = np.sum(aux3, axis=1)
        for i in numba.prange(k):
            y_array[:, 0, i] = Y_sort[:, n-i-1]
            for j_2 in numba.prange(N):
                y_array[j_2, :, i] = y_array[j_2, 0, i]
            hat_beta = fepls_numba(X, Y, y_array[:, :, i], tau)
            for j_1 in numba.prange(N):
                i_c = Y_sort_index[j_1, i]
                aux[j_1, i] = (1/d)*np.sum(np.multiply(hat_beta[j_1, :], X[j_1, i_c, :]))
                aux2[j_1, i] = np.multiply(aux[j_1, i], Y_sort[j_1, n-i-1])
                out[j_1, k] = np.corrcoef(aux3[j_1, :], aux[j_1, :])[0, 1]
    return out

@numba.njit(parallel=True, fastmath=False)
def threshold_index(X, Y, Y_sort_index, tau, m, start):
    """Compute threshold index."""
    N = X.shape[0]
    n = X.shape[1]
    out = np.zeros((N,))
    aux = concomittant_corr(X, Y, Y_sort_index, tau, m)[:, start:]
    return start + np.argmax(aux, axis=1)

@numba.njit(parallel=True, fastmath=False)
def threshold(X, Y, Y_sort_index, tau, m, start):
    """Compute threshold matrix."""
    N = X.shape[0]
    n = X.shape[1]
    y_matrix_out = np.zeros((N, n))
    YY = np.copy(Y)
    Y_sort = sort_2d_array(YY)
    index = threshold_index(X, Y, Y_sort_index, tau, m, start)
    for i in numba.prange(N):
        y_matrix_out[i, :] = Y_sort[i, n-index[i]-1]*np.ones((n,))
    return y_matrix_out

# %%
###################### Configuration
# Fixed parameters
Q = 2.1  # fixed q (noise integrability order) - using 2.1 to satisfy q>2 constraint (user mentioned q=2, but theory requires q>2)
GAMMA = 0.99  # fixed gamma (tail index) - using 0.99 to satisfy gamma<1 constraint (user mentioned gamma=1, but theory requires gamma<1)
N_MC = 1000  # number of Monte Carlo replications (increased for "a lot" of simulations)
N_SAMPLES = 500  # sample size n
D = 101  # dimension d
SNR = 10.0  # signal-to-noise ratio
H = 1/3  # Hurst parameter
MU = 200.0  # noise mean
START = 4  # minimum k for threshold selection

# Varying parameters
RHO_VALUES = [-2.0, -1.0, -0.5]  # different rho values to test
KAPPA_VALUES = np.linspace(0.5, 3.0, 11)  # kappa (c) values to test
# we generate tau values near the edges of the inequality 0 < 2*(kappa+tau)*gamma < 1
# for each kappa, we compute valid tau range and sample near boundaries
TAU_GRID_SIZE = 20  # number of tau values to test per kappa

# Output directory
BASE_DIR = Path("data/simuls")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# %%
###################### Helper Functions

def check_cond_edge(gamma, c, tau, q):
    """Check if parameters satisfy the edge conditions."""
    if tau > 0:
        return (gamma < 1 and gamma > 0 and q > 2 and c > 0 and 
                0 < 2*(c+tau)*gamma < 1 and q*(1-2*tau*gamma) > 2 and 
                q*c*gamma > 1 and tau < 1/(2*gamma))
    else:
        return (gamma < 1 and gamma > 0 and q > 2 and c > 0 and 
                0 < 2*(c+tau)*gamma < 1 and q*c*gamma > 1)

def get_data_path(rho):
    """Get the path for saving/loading data for a given rho."""
    rho_str = f"{rho:.2f}".replace('.', 'p').replace('-', 'm')
    rho_dir = BASE_DIR / f"rho{rho_str}"
    rho_dir.mkdir(parents=True, exist_ok=True)
    return rho_dir / "data.npz"

def generate_data_once(rho, random_seed=0):
    """Generate (Y, Z) data once for a given rho. This is expensive, so we save it."""
    data_path = get_data_path(rho)
    
    # we check if data already exists
    if data_path.exists():
        print(f"Data for rho={rho} already exists at {data_path}, skipping generation.")
        return data_path
    
    print(f"Generating data for rho={rho}...")
    tic = time.time()
    
    npr.seed(random_seed)  # we fix seed for reproducibility
    
    # we generate Y from Burr distribution
    # note: following the original code pattern, Burr_quantile_function is called with gamma directly
    Y = Burr_quantile_function(npr.uniform(0, 1, size=(N_MC, N_SAMPLES)), GAMMA, rho)
    Y_sort_index = np.argsort(Y, axis=1)
    
    # we generate Z for fBm (noise generation depends on kappa, so we save Z and generate X later)
    Z = npr.normal(0, 1, size=(N_MC, N_SAMPLES, D - 1))
    
    # we save Y, Z, and metadata
    np.savez_compressed(
        data_path,
        Y=Y,
        Y_sort_index=Y_sort_index,
        Z=Z,
        rho=rho,
        gamma=GAMMA,
        q=Q,
        n_mc=N_MC,
        n_samples=N_SAMPLES,
        d=D,
        random_seed=random_seed
    )
    
    print(f"Data saved to {data_path} (time: {time.time()-tic:.2f}s)")
    return data_path

# %%
###################### Generate Data for All Rho Values
print("\n" + "=" * 80)
print("STEP 1: Generating data for all rho values")
print("=" * 80)
print(f"[STEP 1] Number of rho values to process: {len(RHO_VALUES)}")
print(f"[STEP 1] Rho values: {RHO_VALUES}")

data_paths = {}
for i, rho in enumerate(RHO_VALUES):
    print(f"\n[STEP 1] Processing rho {i+1}/{len(RHO_VALUES)}: rho={rho}")
    data_paths[rho] = generate_data_once(rho, random_seed=0)
    print(f"[STEP 1] Completed rho={rho}")

print(f"\n[STEP 1] All data generation completed!")
print(f"[STEP 1] Generated data paths: {list(data_paths.keys())}")

# %%
###################### Process Data: Compute Optimal k and Variance for Each (tau, kappa)

def get_tau_range_for_kappa(kappa, eps=1e-3):
    """
    Compute valid tau range for given kappa based on inequality 0 < 2*(kappa+tau)*gamma < 1.
    Returns (tau_min, tau_max) with small epsilon margin.
    """
    tau_lower = -kappa + eps  # from 2*(kappa+tau)*gamma > 0
    tau_upper = (1.0 / (2.0 * GAMMA)) - kappa - eps  # from 2*(kappa+tau)*gamma < 1
    return tau_lower, tau_upper

def compute_optimal_k_and_variance(rho, kappa, tau, k_range=None, verbose=False):
    """
    Load pre-generated data, generate X for given kappa, then compute optimal k and variance.
    
    Returns:
        optimal_k: optimal threshold k
        variance: variance of <hat_beta, beta> at optimal k
        mean_alignment: mean of <hat_beta, beta> at optimal k
    """
    if verbose:
        print(f"    [compute_optimal_k_and_variance] rho={rho}, kappa={kappa:.4f}, tau={tau:.4f}")
    
    data_path = get_data_path(rho)
    if not data_path.exists():
        if verbose:
            print(f"    [compute_optimal_k_and_variance] ERROR: Data not found at {data_path}")
        raise FileNotFoundError(f"Data not found at {data_path}. Run generation first.")
    
    # we load pre-generated data
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Loading data from {data_path}...")
    data = np.load(data_path)
    Y = data['Y']
    Y_sort_index = data['Y_sort_index']
    Z = data['Z']
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Data loaded: Y.shape={Y.shape}, Z.shape={Z.shape}")
    
    # we check if kappa is valid (q*kappa*gamma > 1)
    q_kappa_gamma = Q * kappa * GAMMA
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Checking condition: q*kappa*gamma = {q_kappa_gamma:.4f} > 1? {q_kappa_gamma > 1}")
    if q_kappa_gamma <= 1:
        if verbose:
            print(f"    [compute_optimal_k_and_variance] Invalid: q*kappa*gamma = {q_kappa_gamma:.4f} <= 1")
        return None, None, None  # invalid configuration
    
    # we check if tau is valid (0 < 2*(kappa+tau)*gamma < 1)
    two_kappa_tau_gamma = 2*(kappa+tau)*GAMMA
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Checking condition: 0 < 2*(kappa+tau)*gamma = {two_kappa_tau_gamma:.4f} < 1? {0 < two_kappa_tau_gamma < 1}")
    if not (0 < two_kappa_tau_gamma < 1):
        if verbose:
            print(f"    [compute_optimal_k_and_variance] Invalid: 2*(kappa+tau)*gamma = {two_kappa_tau_gamma:.4f} not in (0, 1)")
        return None, None, None  # invalid configuration
    
    # we generate X for this specific kappa
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Generating X for kappa={kappa:.4f}...")
    aux = np.multiply.outer(Y ** kappa, beta_func(D))  # g(Y)*beta
    if verbose:
        print(f"    [compute_optimal_k_and_variance] aux computed: shape={aux.shape}")
    eps = coeurjolly_cholesky_fbm_var(Y, Z, H, kappa, SNR, MU)  # noise depends on kappa
    if verbose:
        print(f"    [compute_optimal_k_and_variance] eps computed: shape={eps.shape}, mean={eps.mean():.4f}, std={eps.std():.4f}")
    X = aux + eps
    if verbose:
        print(f"    [compute_optimal_k_and_variance] X computed: shape={X.shape}")
    
    # we set k_range if not provided
    if k_range is None:
        k_range = np.arange(START, int(N_SAMPLES/5) + 1, 1)
    if verbose:
        print(f"    [compute_optimal_k_and_variance] k_range: {k_range.min()} to {k_range.max()}, length={len(k_range)}")
    
    # we compute correlation for different k values to find optimal
    m = min(100, N_SAMPLES - START)
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Computing correlations for m={m}...")
    correlations = concomittant_corr(X, Y, Y_sort_index, tau, m)
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Correlations computed: shape={correlations.shape}")
    
    if correlations.shape[1] > START:
        correlations = correlations[:, START:]
        # we find optimal k (maximize correlation)
        optimal_k_indices = np.argmax(correlations, axis=1) + START
        optimal_k = int(np.median(optimal_k_indices))  # we use median over MC replications
        if verbose:
            print(f"    [compute_optimal_k_and_variance] Optimal k indices: min={optimal_k_indices.min()}, max={optimal_k_indices.max()}, median={optimal_k}")
    else:
        optimal_k = START + 10  # we use a default value if correlation computation fails
        if verbose:
            print(f"    [compute_optimal_k_and_variance] Warning: Using default optimal_k={optimal_k}")
    
    # we compute FEPLS estimator at optimal k using threshold function
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Computing threshold matrix...")
    y_matrix = threshold(X, Y, Y_sort_index, tau, 100, START)
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Computing FEPLS estimator...")
    E = fepls(X, Y, y_matrix, tau)
    if verbose:
        print(f"    [compute_optimal_k_and_variance] FEPLS estimator computed: shape={E.shape}")
    
    # we compute alignment with true beta
    beta_true = beta_func(D)
    hatbeta_dot_beta_vals = (1.0 / D) * np.sum(E * beta_true, axis=1)
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Alignment computed: mean={hatbeta_dot_beta_vals.mean():.4f}, std={hatbeta_dot_beta_vals.std():.4f}")
    
    # we compute variance and mean
    variance = float(np.var(hatbeta_dot_beta_vals))
    mean_alignment = float(np.mean(hatbeta_dot_beta_vals))
    if verbose:
        print(f"    [compute_optimal_k_and_variance] Results: optimal_k={optimal_k}, variance={variance:.6f}, mean_alignment={mean_alignment:.4f}")
    
    return optimal_k, variance, mean_alignment

# %%
###################### Process All Combinations
print("\n" + "=" * 80)
print("STEP 2: Processing data for all (tau, kappa) combinations")
print("=" * 80)

results = {}  # we store results as results[rho][kappa][tau] = {'optimal_k': ..., 'variance': ..., 'mean': ...}
all_tau_values = set()  # we collect all tau values used across kappas

# we first determine tau values for each kappa (near edges)
print(f"[STEP 2] Determining tau values for each kappa...")
print(f"[STEP 2] TAU_GRID_SIZE = {TAU_GRID_SIZE}")
tau_by_kappa = {}
for i, kappa in enumerate(KAPPA_VALUES):
    print(f"[STEP 2] Processing kappa {i+1}/{len(KAPPA_VALUES)}: kappa={kappa:.4f}")
    tau_lower, tau_upper = get_tau_range_for_kappa(kappa, eps=1e-2)
    print(f"  [STEP 2]   tau range for kappa={kappa:.4f}: [{tau_lower:.4f}, {tau_upper:.4f}]")
    if tau_lower < tau_upper:
        # we sample tau values, with more points near the boundaries
        tau_edge_lower = np.linspace(tau_lower, tau_lower + 0.1, TAU_GRID_SIZE // 3)
        tau_middle = np.linspace(tau_lower + 0.1, tau_upper - 0.1, TAU_GRID_SIZE // 3)
        tau_edge_upper = np.linspace(tau_upper - 0.1, tau_upper, TAU_GRID_SIZE // 3)
        tau_by_kappa[kappa] = np.concatenate([tau_edge_lower, tau_middle, tau_edge_upper])
        all_tau_values.update(tau_by_kappa[kappa])
        print(f"  [STEP 2]   Generated {len(tau_by_kappa[kappa])} tau values for kappa={kappa:.4f}")
    else:
        tau_by_kappa[kappa] = np.array([])
        print(f"  [STEP 2]   WARNING: No valid tau range for kappa={kappa:.4f}")

TAU_VALUES = np.array(sorted(all_tau_values))
print(f"[STEP 2] Total unique tau values to test: {len(TAU_VALUES)}")
print(f"[STEP 2] Tau value range: [{TAU_VALUES.min():.4f}, {TAU_VALUES.max():.4f}]")

for rho_idx, rho in enumerate(RHO_VALUES):
    print(f"\n{'='*80}")
    print(f"[STEP 2] Processing rho {rho_idx+1}/{len(RHO_VALUES)}: rho = {rho}")
    print(f"{'='*80}")
    results[rho] = {}
    
    valid_kappas = [k for k in KAPPA_VALUES if k in tau_by_kappa and len(tau_by_kappa[k]) > 0]
    print(f"[STEP 2] Valid kappas for rho={rho}: {len(valid_kappas)}/{len(KAPPA_VALUES)}")
    
    for kappa_idx, kappa in enumerate(valid_kappas):
        print(f"\n[STEP 2] Processing kappa {kappa_idx+1}/{len(valid_kappas)}: kappa = {kappa:.4f}")
        results[rho][kappa] = {}
        valid_count = 0
        error_count = 0
        
        tau_list = tau_by_kappa[kappa]
        print(f"  [STEP 2] Testing {len(tau_list)} tau values for kappa={kappa:.4f}")
        
        for tau_idx, tau in enumerate(tau_list):
            if (tau_idx + 1) % 5 == 0 or tau_idx == 0 or tau_idx == len(tau_list) - 1:
                print(f"    [STEP 2] Processing tau {tau_idx+1}/{len(tau_list)}: tau={tau:.4f}")
            try:
                optimal_k, variance, mean_alignment = compute_optimal_k_and_variance(rho, kappa, tau, verbose=False)
                if optimal_k is not None:
                    results[rho][kappa][tau] = {
                        'optimal_k': optimal_k,
                        'variance': variance,
                        'mean_alignment': mean_alignment
                    }
                    valid_count += 1
                else:
                    if (tau_idx + 1) % 5 == 0:
                        print(f"      [STEP 2] Invalid configuration (conditions not satisfied)")
            except Exception as e:
                error_count += 1
                if (tau_idx + 1) % 5 == 0:
                    print(f"      [STEP 2] ERROR: {str(e)}")
                continue
        
        print(f"  [STEP 2] Completed kappa={kappa:.4f}: {valid_count}/{len(tau_list)} valid, {error_count} errors")
    
    total_configs = sum(len(results[rho][k]) for k in results[rho].keys())
    print(f"\n[STEP 2] Completed rho={rho}: {total_configs} total valid configurations")

# %%
###################### Save Results
print("\n" + "=" * 80)
print("STEP 3: Saving results")
print("=" * 80)

results_path = BASE_DIR / "results_tau_kappa.npz"
print(f"[STEP 3] Results will be saved to: {results_path}")

# we convert results to arrays for saving
print(f"[STEP 3] Converting results to dictionary format...")
results_dict = {}
total_entries = 0
for rho in RHO_VALUES:
    if rho not in results:
        print(f"[STEP 3] Skipping rho={rho} (no results)")
        continue
    print(f"[STEP 3] Processing rho={rho}...")
    for kappa in KAPPA_VALUES:
        if kappa not in results[rho]:
            continue
        for tau in results[rho][kappa].keys():
            rho_str = f"{rho:.2f}".replace('.', 'p').replace('-', 'm')
            kappa_str = f"{kappa:.2f}".replace('.', 'p').replace('-', 'm')
            tau_str = f"{tau:.4f}".replace('.', 'p').replace('-', 'm')
            key = f"rho{rho_str}_kappa{kappa_str}_tau{tau_str}"
            results_dict[f"{key}_optimal_k"] = results[rho][kappa][tau]['optimal_k']
            results_dict[f"{key}_variance"] = results[rho][kappa][tau]['variance']
            results_dict[f"{key}_mean"] = results[rho][kappa][tau]['mean_alignment']
            total_entries += 1
    print(f"[STEP 3] Processed {total_entries} entries for rho={rho}")

print(f"[STEP 3] Total entries to save: {total_entries}")

# we also save parameter grids
print(f"[STEP 3] Adding parameter grids to results...")
results_dict['rho_values'] = np.array(RHO_VALUES)
results_dict['kappa_values'] = np.array(KAPPA_VALUES)
results_dict['tau_values'] = TAU_VALUES  # this is a set converted to array, may have variable length
results_dict['gamma'] = GAMMA
results_dict['q'] = Q
print(f"[STEP 3] Parameter grids added: rho_values={RHO_VALUES}, kappa_values shape={KAPPA_VALUES.shape}, tau_values shape={TAU_VALUES.shape}")

print(f"[STEP 3] Saving to {results_path}...")
np.savez_compressed(results_path, **results_dict)
print(f"[STEP 3] Results saved successfully to {results_path}")
file_size = results_path.stat().st_size / (1024 * 1024)  # size in MB
print(f"[STEP 3] File size: {file_size:.2f} MB")

# %%
###################### Create Plots: Optimal k vs tau for different kappa
print("\n" + "=" * 80)
print("STEP 4: Creating plots")
print("=" * 80)

for rho_idx, rho in enumerate(RHO_VALUES):
    print(f"\n[STEP 4] Plotting for rho {rho_idx+1}/{len(RHO_VALUES)}: rho = {rho}")
    
    # we prepare data for plotting
    if rho not in results:
        print(f"  [STEP 4] No results for rho={rho}, skipping plots.")
        continue
    
    valid_kappas = [k for k in KAPPA_VALUES if k in results[rho] and len(results[rho][k]) > 0]
    print(f"  [STEP 4] Found {len(valid_kappas)} valid kappas for plotting")
    
    if len(valid_kappas) == 0:
        print(f"  [STEP 4] No valid data for rho={rho}, skipping plots.")
        continue
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # we plot optimal k vs tau for different kappa
    ax1 = axes[0]
    for kappa in valid_kappas[:5]:  # we plot first 5 kappa values to avoid clutter
        tau_list = sorted([tau for tau in results[rho][kappa].keys()])
        optimal_k_list = [results[rho][kappa][tau]['optimal_k'] for tau in tau_list]
        if len(tau_list) > 0:
            ax1.plot(tau_list, optimal_k_list, marker='o', label=f'κ={kappa:.2f}')
    
    ax1.set_xlabel('τ (tau)')
    ax1.set_ylabel('Optimal k')
    ax1.set_title(f'Optimal k vs τ for different κ (rho={rho:.2f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # we plot variance vs tau for different kappa
    ax2 = axes[1]
    for kappa in valid_kappas[:5]:
        tau_list = sorted([tau for tau in results[rho][kappa].keys()])
        variance_list = [results[rho][kappa][tau]['variance'] for tau in tau_list]
        if len(tau_list) > 0:
            ax2.plot(tau_list, variance_list, marker='s', label=f'κ={kappa:.2f}')
    
    ax2.set_xlabel('τ (tau)')
    ax2.set_ylabel('Variance of <β̂, β>')
    ax2.set_title(f'Variance vs τ for different κ (rho={rho:.2f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # we use log scale for variance
    
    plt.tight_layout()
    rho_str = f"{rho:.2f}".replace('.', 'p').replace('-', 'm')
    plot_path = BASE_DIR / f"plots_rho{rho_str}.png"
    print(f"  [STEP 4] Saving plot to {plot_path}...")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  [STEP 4] Plot saved successfully to {plot_path}")
    plt.close()

# %%
###################### Create 2D Heatmaps: Optimal k and Variance as function of (tau, kappa)
for rho_idx, rho in enumerate(RHO_VALUES):
    print(f"\n[STEP 4] Creating heatmaps for rho {rho_idx+1}/{len(RHO_VALUES)}: rho = {rho}")
    
    if rho not in results or len(results[rho]) == 0:
        print(f"  [STEP 4] No valid data for rho={rho}, skipping heatmaps.")
        continue
    
    print(f"  [STEP 4] Preparing heatmap data...")
    
    # we collect all tau values for this rho
    all_taus_rho = set()
    for kappa in results[rho].keys():
        all_taus_rho.update(results[rho][kappa].keys())
    all_taus_rho = sorted(all_taus_rho)
    
    # we create matrices for heatmaps
    optimal_k_matrix = np.full((len(all_taus_rho), len(KAPPA_VALUES)), np.nan)
    variance_matrix = np.full((len(all_taus_rho), len(KAPPA_VALUES)), np.nan)
    
    for i, tau in enumerate(all_taus_rho):
        for j, kappa in enumerate(KAPPA_VALUES):
            if kappa in results[rho] and tau in results[rho][kappa]:
                optimal_k_matrix[i, j] = results[rho][kappa][tau]['optimal_k']
                variance_matrix[i, j] = results[rho][kappa][tau]['variance']
    
    # we create heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # we plot optimal k heatmap
    im1 = axes[0].imshow(optimal_k_matrix, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    axes[0].set_xlabel('κ (kappa)')
    axes[0].set_ylabel('τ (tau)')
    axes[0].set_title(f'Optimal k as function of (τ, κ) - rho={rho:.2f}')
    axes[0].set_xticks(np.arange(len(KAPPA_VALUES))[::2])
    axes[0].set_xticklabels([f'{k:.2f}' for k in KAPPA_VALUES[::2]])
    axes[0].set_yticks(np.arange(len(all_taus_rho))[::max(1, len(all_taus_rho)//10)])
    axes[0].set_yticklabels([f'{t:.2f}' for t in all_taus_rho[::max(1, len(all_taus_rho)//10)]])
    plt.colorbar(im1, ax=axes[0], label='Optimal k')
    
    # we plot variance heatmap
    variance_log = np.log10(variance_matrix + 1e-10)
    im2 = axes[1].imshow(variance_log, aspect='auto', origin='lower', cmap='plasma', interpolation='nearest')
    axes[1].set_xlabel('κ (kappa)')
    axes[1].set_ylabel('τ (tau)')
    axes[1].set_title(f'log10(Variance) as function of (τ, κ) - rho={rho:.2f}')
    axes[1].set_xticks(np.arange(len(KAPPA_VALUES))[::2])
    axes[1].set_xticklabels([f'{k:.2f}' for k in KAPPA_VALUES[::2]])
    axes[1].set_yticks(np.arange(len(all_taus_rho))[::max(1, len(all_taus_rho)//10)])
    axes[1].set_yticklabels([f'{t:.2f}' for t in all_taus_rho[::max(1, len(all_taus_rho)//10)]])
    plt.colorbar(im2, ax=axes[1], label='log10(Variance)')
    
    plt.tight_layout()
    rho_str = f"{rho:.2f}".replace('.', 'p').replace('-', 'm')
    heatmap_path = BASE_DIR / f"heatmaps_rho{rho_str}.png"
    print(f"  [STEP 4] Saving heatmap to {heatmap_path}...")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"  [STEP 4] Heatmap saved successfully to {heatmap_path}")
    plt.close()

# %%
###################### Summary Statistics
print("\n" + "=" * 80)
print("STEP 5: Summary Statistics")
print("=" * 80)

for rho_idx, rho in enumerate(RHO_VALUES):
    if rho not in results or len(results[rho]) == 0:
        print(f"\n[STEP 5] No results for rho={rho}, skipping summary.")
        continue
    print(f"\n{'='*80}")
    print(f"[STEP 5] Summary for rho {rho_idx+1}/{len(RHO_VALUES)}: rho = {rho}")
    print(f"{'='*80}")
    
    total_configs = 0
    for kappa in results[rho].keys():
        total_configs += len(results[rho][kappa])
    print(f"[STEP 5] Total valid (tau, kappa) configurations: {total_configs}")
    print(f"[STEP 5] Number of unique kappas: {len(results[rho].keys())}")
    
    # we compute average optimal k and variance across all configurations
    print(f"[STEP 5] Computing summary statistics...")
    all_optimal_k = []
    all_variance = []
    all_mean_alignment = []
    for kappa in results[rho].keys():
        for tau in results[rho][kappa].keys():
            all_optimal_k.append(results[rho][kappa][tau]['optimal_k'])
            all_variance.append(results[rho][kappa][tau]['variance'])
            all_mean_alignment.append(results[rho][kappa][tau]['mean_alignment'])
    
    if len(all_optimal_k) > 0:
        print(f"[STEP 5] Optimal k statistics:")
        print(f"  Mean: {np.mean(all_optimal_k):.2f}")
        print(f"  Std: {np.std(all_optimal_k):.2f}")
        print(f"  Min: {np.min(all_optimal_k)}")
        print(f"  Max: {np.max(all_optimal_k)}")
        print(f"[STEP 5] Variance statistics (log10):")
        print(f"  Mean: {np.mean(np.log10(all_variance)):.4f}")
        print(f"  Std: {np.std(np.log10(all_variance)):.4f}")
        print(f"  Min: {np.min(np.log10(all_variance)):.4f}")
        print(f"  Max: {np.max(np.log10(all_variance)):.4f}")
        print(f"[STEP 5] Mean alignment statistics:")
        print(f"  Mean: {np.mean(all_mean_alignment):.4f}")
        print(f"  Std: {np.std(all_mean_alignment):.4f}")
        print(f"  Min: {np.min(all_mean_alignment):.4f}")
        print(f"  Max: {np.max(all_mean_alignment):.4f}")
    else:
        print(f"[STEP 5] WARNING: No valid configurations found!")

print("\n" + "=" * 80)
print("=" * 80)
print("ALL SIMULATIONS AND PLOTS COMPLETED!")
print("=" * 80)
print("=" * 80)
print(f"Summary:")
print(f"  - Processed {len(RHO_VALUES)} rho values: {RHO_VALUES}")
print(f"  - Processed {len(KAPPA_VALUES)} kappa values")
print(f"  - Generated {len(TAU_VALUES)} unique tau values")
print(f"  - Monte Carlo replications: {N_MC}")
print(f"  - Sample size: {N_SAMPLES}")
print(f"  - Dimension: {D}")
print(f"  - Fixed parameters: gamma={GAMMA}, q={Q}")
print(f"  - Results saved to: {BASE_DIR}")
print("=" * 80)
