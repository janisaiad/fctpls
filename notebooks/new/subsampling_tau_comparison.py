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
Subsampling-based tau comparison with hypothesis verification and plotting
Uses subsampled mid prices to create functional data and applies FEPLS analysis
Tests multiple tau values, subsampling intervals, dimensions d, and prediction horizons k
"""

# %%
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import List, Tuple, Dict, Optional
import polars as pl
from tqdm import tqdm
from datetime import datetime

# we set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('subsampling_tau_comparison.log'),
        logging.StreamHandler()
    ]
)

# we add current directory to path to import from adapted.py
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
sys.path.append(current_dir)

# we add utils directory to path for fepls_nd (utils is at project root, not in notebooks)
project_root = os.path.normpath(os.path.join(current_dir, '..', '..'))
utils_dir = os.path.join(project_root, 'utils')
sys.path.append(utils_dir)

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

# we import 2D FEPLS functions from utils
fepls_nd_path = os.path.join(utils_dir, "fepls_nd.py")
if not os.path.exists(fepls_nd_path):
    # we try alternative path calculation
    project_root_alt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_dir = os.path.join(project_root_alt, 'utils')
    fepls_nd_path = os.path.join(utils_dir, "fepls_nd.py")

try:
    from fepls_nd import fepls_nd, projection_nd
except ImportError:
    import importlib.util
    if os.path.exists(fepls_nd_path):
        spec = importlib.util.spec_from_file_location("fepls_nd", fepls_nd_path)
        fepls_nd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fepls_nd_module)
        fepls_nd = fepls_nd_module.fepls_nd
        projection_nd = fepls_nd_module.projection_nd
    else:
        raise ImportError(f"Could not find fepls_nd.py at {fepls_nd_path}")

# we add helper function to reshape 1D to 2D
def reshape_1d_to_2d(X_1d: np.ndarray, d1: Optional[int] = None, d2: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
    """
    we reshape 1D functional data to 2D
    X_1d: shape (N, n, d) or (n, d)
    returns: (X_2d, d1, d2) where X_2d has shape (N, n, d1, d2) or (n, d1, d2)
    """
    if X_1d.ndim == 2:
        n, d = X_1d.shape
        if d1 is None or d2 is None:
            sqrt_d = int(np.sqrt(d))
            for d1_candidate in range(sqrt_d, 0, -1):
                if d % d1_candidate == 0:
                    d1 = d1_candidate
                    d2 = d // d1_candidate
                    break
            if d1 is None:
                d1 = sqrt_d
                d2 = (d + d1 - 1) // d1
        X_2d = X_1d.reshape(n, d1, d2)
        return X_2d, d1, d2
    elif X_1d.ndim == 3:
        N, n, d = X_1d.shape
        if d1 is None or d2 is None:
            sqrt_d = int(np.sqrt(d))
            for d1_candidate in range(sqrt_d, 0, -1):
                if d % d1_candidate == 0:
                    d1 = d1_candidate
                    d2 = d // d1_candidate
                    break
            if d1 is None:
                d1 = sqrt_d
                d2 = (d + d1 - 1) // d1
        X_2d = X_1d.reshape(N, n, d1, d2)
        return X_2d, d1, d2
    else:
        raise ValueError(f"X_1d must have 2 or 3 dimensions, got {X_1d.ndim}")

def fepls_safe_2d(X, Y, y_matrix, tau_tuple):
    """we wrap fepls_nd for 2D with two tau coefficients"""
    tau1, tau2 = tau_tuple
    Y_abs = np.abs(Y)
    y_matrix_abs = np.abs(y_matrix)
    tau_max = max(abs(tau1), abs(tau2))
    epsilon = 1e-8 if tau_max >= 0 else 1e-6
    Y_abs = np.maximum(Y_abs, epsilon)
    y_matrix_abs = np.maximum(y_matrix_abs, epsilon)
    try:
        result = fepls_nd(X, Y_abs, y_matrix_abs, (tau1, tau2))
        if result is not None:
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                for i in range(result.shape[0]):
                    norm = np.linalg.norm(result[i, :, :].flatten())
                    if norm > 1e-10:
                        result[i, :, :] = result[i, :, :] / norm
                    else:
                        d1, d2 = result.shape[1], result.shape[2]
                        result[i, :, :] = np.ones((d1, d2)) / np.sqrt(d1 * d2)
        return result
    except Exception as e:
        return None

# we create safe wrappers that handle negative Y values and negative tau
def fepls_safe(X, Y, y_matrix, tau):
    """we wrap fepls to handle negative Y values and negative tau by using absolute value"""
    Y_abs = np.abs(Y)
    y_matrix_abs = np.abs(y_matrix)
    epsilon = 1e-8 if tau >= 0 else 1e-6  # larger epsilon for negative tau
    Y_abs = np.maximum(Y_abs, epsilon)
    y_matrix_abs = np.maximum(y_matrix_abs, epsilon)
    try:
        result = fepls_original(X, Y_abs, y_matrix_abs, tau)
        if result is not None:
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                for i in range(result.shape[0]):
                    norm = np.linalg.norm(result[i, :])
                    if norm > 1e-10:
                        result[i, :] = result[i, :] / norm
                    else:
                        result[i, :] = np.ones(result.shape[1]) / np.sqrt(result.shape[1])
        return result
    except Exception as e:
        if np.all(Y >= 0):
            try:
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
    # we need to temporarily replace fepls with fepls_original inside bitcoin_concomittant_corr
    # because bitcoin_concomittant_corr calls fepls internally, and we've already handled
    # the absolute value conversion here
    import adapted
    original_fepls = adapted.fepls  # we save the original fepls from adapted module
    
    Y_abs = np.abs(Y)
    epsilon = 1e-8 if tau >= 0 else 1e-6
    Y_abs = np.maximum(Y_abs, epsilon)
    
    # we temporarily replace fepls in adapted module with fepls_original
    # so that bitcoin_concomittant_corr uses the original fepls, not fepls_safe
    adapted.fepls = fepls_original
    
    try:
        result = bitcoin_concomittant_corr_original(X, Y_abs, tau, m)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        if np.all(result == 0.0):
            if np.all(Y >= 0):
                try:
                    Y_safe = np.maximum(Y, epsilon)
                    result = bitcoin_concomittant_corr_original(X, Y_safe, tau, m)
                    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass
    except Exception as e:
        if np.all(Y >= 0):
            try:
                Y_safe = np.maximum(Y, epsilon)
                result = bitcoin_concomittant_corr_original(X, Y_safe, tau, m)
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
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
        compute_fepls_direction, estimate_gamma, estimate_kappa
    )
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("large_scale_hypothesisverif", 
                                                   os.path.join(current_dir, "large_scale_hypothesisverif.py"))
    large_scale_hypothesisverif = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(large_scale_hypothesisverif)
    compute_fepls_direction = large_scale_hypothesisverif.compute_fepls_direction
    estimate_gamma = large_scale_hypothesisverif.estimate_gamma
    estimate_kappa = large_scale_hypothesisverif.estimate_kappa

# %%
# we import subsampling functions from subsampling_apple.py
def curate_mid_price(stock, file, folder_path):
    """we curate mid price data by filtering publishers, trading hours, and calculating mid prices"""
    df = pl.read_parquet(f"{folder_path}/{stock}/{file}")
    
    # we check publisher_id distribution and filter
    num_entries_by_publisher = df.group_by("publisher_id").len().sort("len", descending=True)
    if len(num_entries_by_publisher) > 1:
        df = df.filter(pl.col("publisher_id") == 41)
    else:
        df = df.filter(pl.col("publisher_id") == 2)
    
    # we filter by trading hours based on stock
    if stock == "GOOGL":
        df = df.filter(pl.col("ts_event").dt.hour() >= 13)
        df = df.filter(pl.col("ts_event").dt.hour() <= 20)
    else:
        df = df.filter(
            (
                (pl.col("ts_event").dt.hour() == 9) & (pl.col("ts_event").dt.minute() >= 30) |
                (pl.col("ts_event").dt.hour() > 9) & (pl.col("ts_event").dt.hour() < 16) |
                (pl.col("ts_event").dt.hour() == 16) & (pl.col("ts_event").dt.minute() == 0)
            )
        )
    
    # we calculate mid price
    mid_price = (df["ask_px_00"] + df["bid_px_00"]) / 2
    
    # we manage nans, infs, and nulls with preceding value filling
    shifted = mid_price.shift(1)
    mid_price = mid_price.replace([float('inf'), float('-inf')], None)
    mid_price = mid_price.fill_null(shifted)
    mid_price = mid_price.fill_nan(shifted)
    
    df = df.with_columns(mid_price=mid_price)
    return df

def subsample_mid_prices(df, interval_us):
    """we subsample mid prices at a given interval in microseconds"""
    if len(df) == 0:
        return pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Datetime), "mid_price": pl.Series([], dtype=pl.Float64)})
    
    df = df.sort("ts_event")
    
    # we convert interval_us to a duration string for group_by_dynamic
    if interval_us >= 1_000_000_000:
        duration_sec = interval_us / 1_000_000
        if duration_sec >= 60:
            duration_min = duration_sec / 60
            duration = f"{int(duration_min)}m"
        else:
            duration = f"{int(duration_sec)}s"
    elif interval_us >= 1_000_000:
        duration_ms = interval_us / 1_000
        duration = f"{int(duration_ms)}ms"
    elif interval_us >= 1_000:
        duration = f"{int(interval_us)}us"
    else:
        duration = f"{int(interval_us * 1000)}ns"
    
    try:
        result = df.group_by_dynamic(
            "ts_event",
            every=duration,
            closed="left"
        ).agg(
            pl.col("mid_price").first().alias("mid_price")
        )
    except Exception as e:
        logging.warning(f"group_by_dynamic failed with duration {duration}, using fallback method: {e}")
        step = max(1, len(df) // (df["ts_event"].max() - df["ts_event"].min()).total_seconds() * 1_000_000 / interval_us)
        result = df[::int(step)].select(["ts_event", "mid_price"])
    
    return result

# %%
# Configuration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
# we save figures to refs/report/figures for the report
SAVE_DIR = os.path.join(PROJECT_ROOT, "refs", "report", "figures")
os.makedirs(SAVE_DIR, exist_ok=True)
# we also save summary to results directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "subsampling_tau_comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)

# we define configuration
FOLDER_PATH = "/Data/janis.aiad/lobib/DB_MBP_10/"
STOCK = "AAPL"

# time intervals in microseconds: we select a subset for testing
INTERVALS_US = [500_000,
    50_000, 10_000, 1_000   # 50 milliseconds, 10 milliseconds, 1 millisecond
]

INTERVAL_NAMES = ["500ms"
    "50ms","10ms","1ms"
]

# we define dimensions d to test
DIMENSIONS = [10, 20, 50, 100]

# we define prediction horizons k to test (maximum 30)
K_VALUES = [15, 30,60]

# we define tau grid to test (now as tuples for 2D: (tau1, tau2))
# we create combinations of tau values for 2D
TAU_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
TAU_GRID = [(tau1, tau2) for tau1 in TAU_VALUES for tau2 in TAU_VALUES]
# we also keep a single tau grid for backward compatibility in hypothesis verification
TAU_GRID_SINGLE = TAU_VALUES

# we set random seed
np.random.seed(42)

# %%
def create_batches_from_subsampled_prices(
    prices: np.ndarray,
    d: int,
    k: int,
    min_batches: int = 100
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    we create batches from subsampled prices
    X[i] = [x_i, x_{i+1}, ..., x_{i+d-1}] (d consecutive prices)
    Y[i] = max(x_{i+d}, x_{i+d+1}, ..., x_{i+d+k-1}) (max of next k prices)
    
    returns (X, Y) where X has shape (n_batches, d) and Y has shape (n_batches,)
    """
    n_total = len(prices)
    required_length = d + k  # we need at least d+k points for one batch
    
    if n_total < required_length:
        return None, None
    
    # we compute how many batches we can create
    n_batches = n_total - required_length + 1
    
    if n_batches < min_batches:
        return None, None
    
    # we create X: each row is d consecutive prices
    X = np.zeros((n_batches, d))
    for i in range(n_batches):
        X[i, :] = prices[i:i+d]
    
    # we create Y: max of next k prices
    Y = np.zeros(n_batches)
    for i in range(n_batches):
        Y[i] = np.max(prices[i+d:i+d+k])
    
    return X, Y

def load_and_process_stock_data(
    stock: str,
    folder_path: str,
    interval_us: int,
    d: int,
    k: int,
    min_batches: int = 100,
    max_days: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    we load stock data, subsample at given interval, and create batches
    returns (X, Y, n_batches) or (None, None, 0) if insufficient data
    """
    stock_path = os.path.join(folder_path, stock)
    
    if not os.path.exists(stock_path):
        logging.warning(f"path does not exist: {stock_path}")
        return None, None, 0
    
    parquet_files = [f for f in os.listdir(stock_path) 
                     if f.endswith('.parquet') and not f.endswith('_curated.parquet')]
    
    if not parquet_files:
        logging.warning(f"no parquet files found for {stock}")
        return None, None, 0
    
    # we sort files by name (assuming they contain dates)
    parquet_files.sort()
    
    # we limit number of days if specified
    if max_days is not None:
        parquet_files = parquet_files[:max_days]
    
    all_prices = []
    all_dates = []
    
    # we process each day
    for file in tqdm(parquet_files, desc=f"processing {stock} files", leave=False):
        try:
            # we curate mid price data
            df = curate_mid_price(stock, file, folder_path)
            
            if len(df) == 0:
                continue
            
            # we subsample at given interval
            subsampled_df = subsample_mid_prices(df, interval_us)
            
            if len(subsampled_df) == 0:
                continue
            
            # we extract prices and dates
            prices = subsampled_df["mid_price"].to_numpy()
            dates = subsampled_df["ts_event"].to_list()
            
            # we filter out NaN/Inf prices
            valid_mask = np.isfinite(prices)
            prices = prices[valid_mask]
            dates = [d for i, d in enumerate(dates) if valid_mask[i]]
            
            if len(prices) < d + k:
                continue
            
            all_prices.extend(prices)
            all_dates.extend(dates)
            
        except Exception as e:
            logging.warning(f"error processing {stock}/{file}: {e}")
            continue
    
    if len(all_prices) < d + k:
        logging.warning(f"insufficient total prices: {len(all_prices)} < {d + k}")
        return None, None, 0
    
    # we convert to numpy array
    all_prices = np.array(all_prices)
    
    # we create batches
    X, Y = create_batches_from_subsampled_prices(all_prices, d, k, min_batches)
    
    if X is None or Y is None:
        return None, None, 0
    
    n_batches = len(X)
    logging.info(f"created {n_batches} batches from {len(all_prices)} prices (d={d}, k={k})")
    
    return X, Y, n_batches

# %%
# we import plotting functions from large_scale_tau_comparison_plotting.py
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
    config_name: str,
    save_path: str,
    corr_curve: np.ndarray = None,
    tau_tuple: Optional[Tuple[float, float]] = None,  # we pass tau tuple for 2D
) -> None:
    """we create a comprehensive plot for a single tau value (supports 2D beta_hat)"""
    n_samples = Y_fepls.shape[1]
    # we detect if beta_hat is 2D or 1D
    is_2d = beta_hat.ndim == 2
    if is_2d:
        d1, d2 = beta_hat.shape
        d_points = d1 * d2
    else:
        d_points = beta_hat.shape[0]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # we compute correlation curve for this tau if not provided
    m_threshold = int(n_samples / 5)
    if corr_curve is None:
        corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau, m_threshold)
        if np.any(np.isnan(corr_curve)) or np.any(np.isinf(corr_curve)):
            corr_curve = np.nan_to_num(corr_curve, nan=0.0, posinf=0.0, neginf=0.0)
    
    Y_sorted = np.sort(Y_fepls[0])[::-1]
    
    # Plot 1: Correlation curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(corr_curve, 'b-', linewidth=2)
    ax1.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best k={best_k}')
    tau_title = f"tau=({tau_tuple[0]:.1f}, {tau_tuple[1]:.1f})" if tau_tuple else f"tau={tau}"
    ax1.set_title(f'Tail Correlation vs k ({tau_title})', fontsize=12, fontweight='bold')
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
    
    # Plot 4: Beta (1D curve or 2D image)
    ax4 = fig.add_subplot(gs[1, 0])
    if is_2d:
        # we display 2D beta as an image
        im = ax4.imshow(beta_hat, cmap='RdBu_r', aspect='auto', interpolation='nearest')
        tau_title = f"tau=({tau_tuple[0]:.1f}, {tau_tuple[1]:.1f})" if tau_tuple else f"tau={tau}"
        ax4.set_title(f'FEPLS Direction Beta (2D) ({tau_title})', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Dimension d2')
        ax4.set_ylabel('Dimension d1')
        plt.colorbar(im, ax=ax4)
    else:
        # we display 1D beta as a curve
    ax4.plot(beta_hat, color='purple', linewidth=2)
    ax4.set_title(f'FEPLS Direction Beta(t) (tau={tau})', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time Index')
    ax4.set_ylabel('Weight')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Conditional Quantile with Scatter
    ax5 = fig.add_subplot(gs[1, 1:])
    # we compute projections using projection_nd for 2D or standard dot for 1D
    if is_2d:
        proj_vals = projection_nd(X_fepls[0, :, :, :], beta_hat)
        h_univ = 0.2 * np.std(proj_vals)
        h_func = 0.2 * np.mean([np.std(X_fepls[0, i, :, :].flatten()) for i in range(n_samples)])
    else:
        proj_vals = np.dot(X_fepls[0], beta_hat) / d_points
        h_univ = 0.2 * np.std(proj_vals)
    h_func = 0.2 * np.mean(np.std(X_fepls[0], axis=0))
    h_univ_vec = h_univ * np.ones(n_samples)
    h_func_vec = h_func * np.ones(n_samples)
    
    Y_vals = Y_fepls[0]
    
    Y_sorted_idx = np.argsort(Y_vals)[::-1]
    extreme_threshold = Y_vals[Y_sorted_idx[best_k]] if best_k < len(Y_vals) else np.median(Y_vals)
    is_extreme = Y_vals >= extreme_threshold
    
    try:
        # we flatten beta_hat for plot_quantile_conditional_on_sample_new which expects 1D
        if is_2d:
            beta_hat_1d = beta_hat.flatten()
            # we also need to flatten X_fepls for this function
            X_fepls_1d = X_fepls.reshape(X_fepls.shape[0], X_fepls.shape[1], -1)
        else:
            beta_hat_1d = beta_hat
            X_fepls_1d = X_fepls
        quantiles, s_grid = plot_quantile_conditional_on_sample_new(
            X_fepls_1d, Y_fepls,
            dimred=beta_hat_1d,
            x_func=beta_hat_1d,
            alpha=0.95,
            h_univ_vector=h_univ_vec,
            h_func_vector=h_func_vec
        )
        ax5.scatter(proj_vals[~is_extreme], Y_vals[~is_extreme], 
                   alpha=0.4, s=20, color='blue', label='Non-extreme', zorder=1)
        ax5.scatter(proj_vals[is_extreme], Y_vals[is_extreme], 
                   alpha=0.7, s=25, color='red', label='Extreme', zorder=2)
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
    
    tau_display = f"({tau_tuple[0]:.1f}, {tau_tuple[1]:.1f})" if tau_tuple else f"{tau:.1f}"
    summary_text = f"""
    Hypothesis Verification for tau = {tau_display}
    ========================================================================
    gamma_hat = {gamma_hat:.6f}
    kappa_hat = {kappa_hat:.6f}
    2*(kappa + tau_avg)*gamma = {hypothesis_value:.6f}
    
    Conditions:
    - Positive: {hypothesis_value > 0.0} (must be > 0)
    - Less than 1: {hypothesis_value < 1.0} (must be < 1)
    
    Status: {status_text}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor=status_color, alpha=0.2))
    
    fig.suptitle(f'{config_name} - tau = {tau_display}', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_tau_comparison(
    tau_results: List[Dict],
    config_name: str,
    save_path: str,
) -> None:
    """we create a comparison plot across all tau values"""
    valid_taus = [r for r in tau_results if r['hypothesis_valid']]
    invalid_taus = [r for r in tau_results if not r['hypothesis_valid']]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Hypothesis value vs tau (we use tau_avg for x-axis)
    ax1 = axes[0, 0]
    if valid_taus:
        valid_tau_vals = [r['tau_avg'] for r in valid_taus]
        valid_hyp_vals = [r['hypothesis_value'] for r in valid_taus]
        ax1.scatter(valid_tau_vals, valid_hyp_vals, c='green', s=100, 
                   marker='o', label='Valid', zorder=3)
    if invalid_taus:
        invalid_tau_vals = [r['tau_avg'] for r in invalid_taus]
        invalid_hyp_vals = [r['hypothesis_value'] for r in invalid_taus]
        ax1.scatter(invalid_tau_vals, invalid_hyp_vals, c='red', s=100,
                   marker='x', label='Invalid', zorder=3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax1.axhline(y=1, color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax1.axhspan(0, 1, alpha=0.1, color='green', label='Valid region')
    ax1.set_xlabel('tau_avg')
    ax1.set_ylabel('2*(kappa + tau_avg)*gamma')
    ax1.set_title('Hypothesis Value vs tau_avg', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best correlation vs tau (we use tau_avg)
    ax2 = axes[0, 1]
    all_tau_vals = [r['tau_avg'] for r in tau_results]
    all_max_corr = [r['max_correlation'] for r in tau_results]
    colors = ['green' if r['hypothesis_valid'] else 'red' for r in tau_results]
    ax2.scatter(all_tau_vals, all_max_corr, c=colors, s=100, alpha=0.6, marker='o', label='Max Correlation')
    ax2.set_xlabel('tau_avg')
    ax2.set_ylabel('Max Correlation')
    ax2.set_title('Max Correlation vs tau_avg', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best k vs tau (we use tau_avg)
    ax3 = axes[1, 0]
    all_best_k = [r['best_k'] for r in tau_results]
    ax3.scatter(all_tau_vals, all_best_k, c=colors, s=100, alpha=0.6)
    ax3.set_xlabel('tau_avg')
    ax3.set_ylabel('Best k')
    ax3.set_title('Best k vs tau_avg', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if tau_results:
        table_data = []
        headers = ['tau1', 'tau2', 'tau_avg', '2*(k+τ)*γ', 'Max Corr', 'Best k', 'Status']
        for r in sorted(tau_results, key=lambda x: (x['tau1'], x['tau2'])):
            status = '✓ VALID' if r['hypothesis_valid'] else '✗ INVALID'
            table_data.append([
                f"{r['tau1']:+.2f}",
                f"{r['tau2']:+.2f}",
                f"{r['tau_avg']:+.2f}",
                f"{r['hypothesis_value']:.4f}",
                f"{r['max_correlation']:.4f}",
                f"{r['best_k']}",
                status
            ])
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.12, 0.12, 0.12, 0.18, 0.15, 0.12, 0.19])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        for i, r in enumerate(sorted(tau_results, key=lambda x: (x['tau1'], x['tau2']))):
            if r['hypothesis_valid']:
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#90EE90')
    
    fig.suptitle(f'{config_name} - Tau Comparison', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# %%
def analyze_configuration_with_multiple_tau(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    config_name: str,
    tau_grid: List[Tuple[float, float]],
    interval_name: str,
    d: int,
    k: int,
) -> None:
    """we analyze a configuration with multiple tau values (2D FEPLS), verify hypotheses, and create plots"""
    n, d_actual = X_data.shape
    
    if d_actual != d:
        logging.warning(f"dimension mismatch: expected {d}, got {d_actual}")
        return
    
    # we reshape 1D to 2D for FEPLS 2D
    X_data_2d, d1, d2 = reshape_1d_to_2d(X_data)
    logging.info(f"  reshaped X from (n={n}, d={d}) to (n={n}, d1={d1}, d2={d2})")
    
    # we reshape for FEPLS functions (expects (N, n, d1, d2) format)
    X_fepls = np.expand_dims(X_data_2d, axis=0)
    Y_fepls = np.expand_dims(Y_data, axis=0)
    
    n_samples = Y_fepls.shape[1]
    m_threshold = int(n_samples / 5)
    
    # we estimate gamma and kappa once (they don't depend on tau)
    logging.info(f"  estimating gamma and kappa for {config_name}...")
    gamma_hat = estimate_gamma(Y_data)
    if gamma_hat is None:
        logging.error(f"  ERROR: could not estimate gamma for {config_name}")
        return
    
    # we need an initial beta_hat to estimate kappa (use tau=1.0 as default)
    beta_hat_init = compute_fepls_direction(X_data, Y_data, tau=1.0)
    if beta_hat_init is None:
        logging.error(f"  ERROR: could not compute initial beta_hat for {config_name}")
        return
    
    kappa_hat = estimate_kappa(X_data, Y_data, beta_hat_init)
    if kappa_hat is None:
        logging.error(f"  ERROR: could not estimate kappa for {config_name}")
        return
    
    logging.info(f"  gamma_hat = {gamma_hat:.6f}, kappa_hat = {kappa_hat:.6f}")
    
    # we test each tau tuple
    tau_results = []
    valid_tau_count = 0
    
    for tau_tuple in tqdm(tau_grid, desc=f"  Testing tau values for {config_name}", leave=False):
        tau1, tau2 = tau_tuple
        tau_avg = (tau1 + tau2) / 2.0  # we use average for hypothesis verification and correlation
        logging.info(f"  testing tau = ({tau1}, {tau2}), avg = {tau_avg:.2f}...")
        
        # we compute correlation curve using 1D version with tau_avg (bitcoin_concomittant_corr expects 1D)
        X_fepls_1d = np.expand_dims(X_data, axis=0)
        corr_curve = bitcoin_concomittant_corr(X_fepls_1d, Y_fepls, tau_avg, m_threshold)
        
        if np.any(np.isnan(corr_curve)) or np.any(np.isinf(corr_curve)):
            logging.warning(f"    WARNING: correlation curve contains NaN/Inf for tau=({tau1}, {tau2}), replacing with 0")
            corr_curve = np.nan_to_num(corr_curve, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.all(corr_curve == 0.0):
            logging.warning(f"    WARNING: correlation curve is all zeros for tau=({tau1}, {tau2})")
        
        if len(corr_curve) > 0:
            logging.info(f"    DEBUG: max correlation for tau=({tau1}, {tau2}): {np.max(corr_curve):.6f}")
            logging.info(f"    DEBUG: correlation curve first 5 values for tau=({tau1}, {tau2}): {corr_curve[:5]}")
        
        # we find best k using sharpness
        valid_k_start = 10
        if len(corr_curve) > valid_k_start:
            valid_curve = corr_curve[valid_k_start:]
            valid_curve = np.nan_to_num(valid_curve, nan=0.0, posinf=0.0, neginf=0.0)
            
            sharpness_values = np.zeros(len(valid_curve))
            for i in range(1, len(valid_curve) - 1):
                sharpness_values[i] = 2 * valid_curve[i] - valid_curve[i-1] - valid_curve[i+1]
            
            if np.all(sharpness_values <= 0):
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
        
        logging.info(f"    DEBUG: best_k for tau=({tau1}, {tau2}): {best_k}, max_corr={max_corr:.6f}")
        
        # we compute beta_hat for this tau using FEPLS 2D
        Y_sorted = np.sort(Y_fepls[0])[::-1]
        y_n = Y_sorted[best_k] if best_k < len(Y_sorted) else Y_sorted[0]
        y_matrix = y_n * np.ones_like(Y_fepls)
        
        logging.info(f"    DEBUG: y_n (threshold) for tau=({tau1}, {tau2}): {y_n:.6f}")
        
        try:
            # we verify that tau is actually used by checking the weights
            Y_abs = np.abs(Y_fepls)
            epsilon = 1e-8 if tau_avg >= 0 else 1e-6
            Y_abs = np.maximum(Y_abs, epsilon)
            weights_sample = Y_abs[0, :best_k+1]**tau_avg
            logging.info(f"    DEBUG: sample weights (Y**tau_avg) for tau=({tau1}, {tau2}): min={np.min(weights_sample):.6e}, max={np.max(weights_sample):.6e}, mean={np.mean(weights_sample):.6e}")
            
            # we use 2D FEPLS
            E0 = fepls_safe_2d(X_fepls, Y_fepls, y_matrix, tau_tuple)
            if E0 is None:
                logging.warning(f"    ERROR: fepls_2d returned None for tau=({tau1}, {tau2}), skipping")
                continue
            beta_hat = E0[0, :, :]  # shape (d1, d2)
            if np.any(np.isnan(beta_hat)) or np.any(np.isinf(beta_hat)):
                logging.warning(f"    WARNING: beta_hat contains NaN/Inf for tau=({tau1}, {tau2}), trying to fix...")
                beta_hat = np.nan_to_num(beta_hat, nan=0.0, posinf=0.0, neginf=0.0)
                norm = np.linalg.norm(beta_hat.flatten())
                if norm > 1e-10:
                    beta_hat = beta_hat / norm
                else:
                    logging.warning(f"    ERROR: beta_hat norm is zero for tau=({tau1}, {tau2}), skipping")
                    continue
            beta_norm = np.linalg.norm(beta_hat.flatten())
            logging.info(f"    DEBUG: beta_hat norm for tau=({tau1}, {tau2}): {beta_norm:.6f}, shape: {beta_hat.shape}")
            logging.info(f"    DEBUG: beta_hat sum of squares for tau=({tau1}, {tau2}): {np.sum(beta_hat**2):.6f}")
        except Exception as e:
            logging.error(f"    ERROR computing beta_hat for tau=({tau1}, {tau2}): {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # we verify hypothesis using average tau
        hypothesis_value = 2.0 * (kappa_hat + tau_avg) * gamma_hat
        condition_pos = hypothesis_value > 0.0
        condition_upper = hypothesis_value < 1.0
        hypothesis_valid = condition_pos and condition_upper
        
        if hypothesis_valid:
            valid_tau_count += 1
            logging.info(f"    ✓ VALID: 2*(kappa+tau_avg)*gamma = {hypothesis_value:.6f}")
        else:
            logging.info(f"    ✗ INVALID: 2*(kappa+tau_avg)*gamma = {hypothesis_value:.6f}")
        
        # we store results
        tau_results.append({
            'tau': tau_tuple,
            'tau1': tau1,
            'tau2': tau2,
            'tau_avg': tau_avg,
            'beta_hat': beta_hat,
            'best_k': best_k,
            'max_correlation': max_corr,
            'sharpness': sharpness,
            'hypothesis_value': hypothesis_value,
            'hypothesis_valid': hypothesis_valid,
        })
        
        # we create plot for ALL tau
        plot_path = os.path.join(SAVE_DIR, f"{config_name}_tau_{tau1:.1f}_{tau2:.1f}.png")
        plot_single_tau_analysis(
            X_fepls, Y_fepls, beta_hat, tau_avg, best_k,
            gamma_hat, kappa_hat, hypothesis_value, hypothesis_valid,
            config_name, plot_path, corr_curve=corr_curve, tau_tuple=tau_tuple
        )
    
    # we create comparison plot
    comparison_path = os.path.join(SAVE_DIR, f"{config_name}_tau_comparison.png")
    plot_tau_comparison(tau_results, config_name, comparison_path)
    
    logging.info(f"  completed {config_name}: {valid_tau_count}/{len(tau_grid)} valid tau values")
    logging.info(f"  plots saved to {SAVE_DIR}")

# %%
def main():
    """we run tau comparison analysis for subsampled data"""
    logging.info("=" * 80)
    logging.info("Subsampling-based Tau Comparison with Hypothesis Verification")
    logging.info("=" * 80)
    
    # we determine how many days to use based on target batch counts
    target_batches = [100, 200, 5000]
    max_days_options = [None, 10, 50, 100, 200, 500]  # we try different numbers of days
    
    results_summary = []
    
    # we calculate total number of configurations for progress bar
    total_configs = len(INTERVALS_US) * len(DIMENSIONS) * len(K_VALUES)
    config_pbar = tqdm(total=total_configs, desc="Overall progress", position=0)
    
    # we iterate over all configurations
    for interval_idx, (interval_us, interval_name) in enumerate(zip(INTERVALS_US, INTERVAL_NAMES)):
        for d in DIMENSIONS:
            for k in K_VALUES:
                config_pbar.update(1)
                config_name = f"{STOCK}_{interval_name}_d{d}_k{k}"
                logging.info(f"\n{'='*80}")
                logging.info(f"analyzing configuration: {config_name}")
                logging.info(f"{'='*80}")
                
                # we try to get enough batches
                X_data = None
                Y_data = None
                n_batches = 0
                max_days_used = None
                
                for max_days in max_days_options:
                    logging.info(f"  trying with max_days={max_days}...")
                    X_data, Y_data, n_batches = load_and_process_stock_data(
                        STOCK, FOLDER_PATH, interval_us, d, k,
                        min_batches=100, max_days=max_days
                    )
                    
                    if X_data is not None and n_batches >= 100:
                        max_days_used = max_days
                        logging.info(f"  success: {n_batches} batches with max_days={max_days}")
                        break
                
                if X_data is None or n_batches < 100:
                    logging.warning(f"  skipping {config_name}: insufficient batches ({n_batches})")
                    results_summary.append({
                        'config': config_name,
                        'interval': interval_name,
                        'd': d,
                        'k': k,
                        'n_batches': n_batches,
                        'status': 'insufficient_data'
                    })
                    continue
                
                # we log batch count
                logging.info(f"  using {n_batches} batches (max_days={max_days_used})")
                
                # we run analysis
                try:
                    analyze_configuration_with_multiple_tau(
                        X_data, Y_data, config_name, TAU_GRID,
                        interval_name, d, k
                    )
                    results_summary.append({
                        'config': config_name,
                        'interval': interval_name,
                        'd': d,
                        'k': k,
                        'n_batches': n_batches,
                        'max_days': max_days_used,
                        'status': 'completed'
                    })
                except Exception as e:
                    logging.error(f"ERROR analyzing {config_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    results_summary.append({
                        'config': config_name,
                        'interval': interval_name,
                        'd': d,
                        'k': k,
                        'n_batches': n_batches,
                        'status': f'error: {str(e)}'
                    })
                    continue
    
    # we close progress bar
    config_pbar.close()
    
    # we save summary to results directory
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(RESULTS_DIR, "analysis_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"\n{'='*80}")
    logging.info("analysis complete!")
    logging.info(f"summary saved to: {summary_path}")
    logging.info(f"plots saved to: {SAVE_DIR}")
    logging.info(f"{'='*80}")

# %%
if __name__ == "__main__":
    main()

