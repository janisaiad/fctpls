import os
import itertools
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 0. Global configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "large_scale_logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "large_scale_hypothesisverif.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger("").addHandler(console_handler)


# =============================================================================
# 1. Data loading (adapted from adaptedk5.py, without plots)
# =============================================================================

def load_stooq_file(filepath: str) -> Optional[pd.DataFrame]:
    """we parse a single Stooq .txt file into a clean DataFrame"""
    try:
        df = pd.read_csv(filepath)  # we read csv file
        df.columns = [c.replace("<", "").replace(">", "").lower() for c in df.columns]  # we normalize column names
        df["datetime"] = pd.to_datetime(  # we combine date and time
            df["date"].astype(str) + " " + df["time"].astype(str).str.zfill(6),
            format="%Y%m%d %H%M%S",
        )
        df = df.set_index("datetime")  # we set datetime index
        return df[["close", "vol"]]  # we keep relevant columns
    except Exception as e:  # we catch all errors
        logging.warning(f"Error loading {filepath}: {e}")
        return None


def create_functional_data(
    df_dict: Dict[str, pd.DataFrame], ticker_name: str, time_grid: Optional[pd.Index] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Index]]:
    """we transform 5-min time series into a matrix (days x timepoints)"""
    if ticker_name not in df_dict or df_dict[ticker_name] is None:
        return None, None

    df = df_dict[ticker_name].copy()  # we copy df
    df["Date"] = df.index.date  # we add date column
    df["Time"] = df.index.time  # we add time column

    if time_grid is None:
        time_counts = df["Time"].value_counts()  # we count timestamps
        n_days = df["Date"].nunique()  # we count days
        common_times = time_counts[time_counts > n_days * 0.5].index.sort_values()  # we keep frequent times
        time_grid = common_times

    pivot_df = df.pivot_table(index="Date", columns="Time", values="close")  # we pivot to (days x times)
    pivot_df = pivot_df.reindex(columns=time_grid)  # we align columns
    pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)  # we fill missing values along time
    pivot_df = pivot_df.dropna()  # we drop days with remaining NaN

    return pivot_df, time_grid


def build_functional_dataset() -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """we build functional dataset for all available assets"""
    data_dir = os.path.join(
        PROJECT_ROOT,
        "data",
        "stooq",
        "hungary",
        "5_hu_txt",
        "data",
        "5 min",
        "hu",
        "bse stocks",
    )  # we set data directory

    # subset of tickers to keep computation reasonable but still large-scale
    targets = [
        "4ig.hu.txt",
        "akko.hu.txt",
        "alteo.hu.txt",
        "amixa.hu.txt",
        "any.hu.txt",
        "appeninn.hu.txt",
        "autowallis.hu.txt",
        "bif.hu.txt",
        "dunahouse.hu.txt",
        "esense.hu.txt",
        "granit.hu.txt",
        "gspark.hu.txt",
        "masterplast.hu.txt",
        "mol.hu.txt",
        "mtelekom.hu.txt",
        "opus.hu.txt",
        "otp.hu.txt",
        "pannergy.hu.txt",
        "raba.hu.txt",
        "richter.hu.txt",
        "vertikal.hu.txt",
        "waberers.hu.txt",
        "zwack.hu.txt",
    ]  # we define tickers subset

    data_store: Dict[str, pd.DataFrame] = {}
    for t in targets:
        path = os.path.join(data_dir, t)  # we build path
        if os.path.exists(path):
            df = load_stooq_file(path)  # we load data
            if df is not None:
                data_store[t] = df  # we store df

    logging.info(f"Loaded {len(data_store)} assets for large-scale hypothesis verification")

    if "otp.hu.txt" in data_store:
        master, master_grid = create_functional_data(data_store, "otp.hu.txt")  # we use otp as reference grid
    elif len(data_store) > 0:
        first_key = list(data_store.keys())[0]
        master, master_grid = create_functional_data(data_store, first_key)  # we use first asset as reference
    else:
        raise RuntimeError("No data loaded from Stooq directory")

    if master_grid is None:
        raise RuntimeError("Could not establish master time grid")

    logging.info(f"Master time grid has {len(master_grid)} points per day")

    func_data: Dict[str, Dict[str, np.ndarray]] = {}
    for t, df in data_store.items():
        mat, _ = create_functional_data(data_store, t, time_grid=master_grid)  # we align on master grid
        if mat is not None and len(mat) > 0:
            log_prices = np.log(mat.values)  # we compute log-prices
            diff_curves = np.diff(log_prices, axis=1)  # we compute intra-day returns
            func_data[t] = {
                "dates": mat.index,  # we keep dates as pandas Index for intersection
                "curves": diff_curves,  # we store curves (n_days x d-1)
                "max_return": np.max(diff_curves, axis=1),  # we store daily maxima
            }

    tickers = list(func_data.keys())
    logging.info(f"Functional dataset built for {len(tickers)} assets")
    return func_data, tickers


# =============================================================================
# 2. FEPLS direction and parameter estimation (no plots)
# =============================================================================

def compute_fepls_direction(
    X: np.ndarray,  # shape (n_samples, d)
    Y: np.ndarray,  # shape (n_samples,)
    tau: float = 1.0,
    k_fraction: float = 0.2,
) -> Optional[np.ndarray]:
    """we compute a simple FEPLS-like direction beta_hat"""
    n, d = X.shape
    if n < 30:
        return None

    Y_sorted = np.sort(Y)[::-1]  # we sort Y descending
    k = max(int(k_fraction * n), 5)  # we choose number of extremes
    if k >= n:
        k = n - 1
    threshold = Y_sorted[k]  # we set threshold

    weights = (Y >= threshold).astype(float) * (np.clip(Y, a_min=1e-12, a_max=None) ** tau)  # we compute weights
    if np.all(weights == 0.0):
        return None

    v = (weights @ X) / n  # we compute tail-moment vector
    norm_v = np.linalg.norm(v)
    if norm_v <= 0.0:
        return None
    beta_hat = v / norm_v  # we normalize
    return beta_hat


def estimate_gamma(Y: np.ndarray) -> Optional[float]:
    """we estimate tail index gamma using a simple Hill estimator"""
    Y = np.asarray(Y, dtype=float)
    Y_pos = Y[Y > 0.0]  # we keep positive values only
    n = Y_pos.size
    if n < 30:  # we reduce threshold from 50 to 30
        return None

    Y_sorted = np.sort(Y_pos)[::-1]  # we sort descending
    k = max(min(int(0.1 * n), 200), 10)  # we choose k between 10 and 200, reduce min from 20 to 10
    if k + 1 >= n:
        k = max(n - 2, 5)  # we ensure k >= 5
    if k <= 0:
        return None

    log_Y = np.log(Y_sorted)
    if Y_sorted[k] <= 0.0:  # we check for valid threshold
        return None
    gamma_hat = float(np.mean(log_Y[:k] - log_Y[k]))  # we compute Hill estimator
    if gamma_hat <= 0.0 or gamma_hat >= 2.0:
        return None
    return gamma_hat


def estimate_kappa(X: np.ndarray, Y: np.ndarray, beta_hat: np.ndarray) -> Optional[float]:
    """we estimate kappa from log(|<X,beta>|) vs log(Y) in the tail"""
    n, d = X.shape
    proj = X @ beta_hat  # we compute projection (no division by d)

    # we focus on tail of Y
    Y_sorted = np.sort(Y)[::-1]
    k_tail = max(min(int(0.2 * n), 100), 15)  # we reduce min from 20 to 15
    if k_tail >= n:
        k_tail = n - 1
    if k_tail < 0:
        return None
    threshold = Y_sorted[k_tail]

    mask = (Y >= threshold) & (Y > 0.0) & (np.abs(proj) > 1e-10)  # we select valid tail points
    n_valid = np.sum(mask)
    if n_valid < 20:  # we reduce threshold from 30 to 20
        return None

    log_Y = np.log(Y[mask])
    log_proj = np.log(np.abs(proj[mask]) + 1e-12)

    # we perform simple linear regression
    x = log_Y - np.mean(log_Y)
    y = log_proj - np.mean(log_proj)
    denom = float(np.sum(x * x))
    if denom <= 1e-10:  # we add small threshold
        return None
    slope = float(np.sum(x * y) / denom)
    return slope if slope > 0.0 else None


# =============================================================================
# 3. Hypothesis testing for one pair (no plots)
# =============================================================================

def test_hypotheses_for_pair(
    X: np.ndarray,
    Y: np.ndarray,
    tau_grid: List[float],
    pair_name: str,
    log_file_path: str,
) -> None:
    """we test FEPLS hypotheses for one pair and log all results"""
    n, d = X.shape
    lines: List[str] = []

    def add(line: str) -> None:
        lines.append(line)
        logging.info(f"[{pair_name}] {line}")

    add("=" * 80)
    add(f"Pair: {pair_name}")
    add(f"n_samples = {n}, dimension d = {d}")
    add(f"Y statistics: min={np.min(Y):.6f}, max={np.max(Y):.6f}, mean={np.mean(Y):.6f}, std={np.std(Y):.6f}")
    add(f"Y positive count: {np.sum(Y > 0)}/{n}")

    # Step 1: FEPLS direction
    add("")
    add("Step 1: Computing FEPLS direction beta_hat...")
    beta_hat = compute_fepls_direction(X, Y, tau=1.0)
    if beta_hat is None:
        add("ERROR: Could not compute FEPLS direction beta_hat")
        add("  Possible reasons: n < 30, all weights zero, or zero norm")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    beta_norm = np.linalg.norm(beta_hat)
    add(f"  beta_hat computed successfully (norm={beta_norm:.6f})")
    add(f"  beta_hat stats: min={np.min(beta_hat):.6f}, max={np.max(beta_hat):.6f}, mean={np.mean(beta_hat):.6f}")

    # Step 2: estimate gamma
    add("")
    add("Step 2: Estimating gamma (tail index) using Hill estimator...")
    gamma_hat = estimate_gamma(Y)
    if gamma_hat is None:
        add("ERROR: Could not estimate gamma (tail index)")
        add("  Possible reasons: < 50 positive Y values, k out of range, or gamma_hat <= 0 or >= 2")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    add(f"  gamma_hat = {gamma_hat:.6f}")

    # Step 3: estimate kappa
    add("")
    add("Step 3: Estimating kappa (link function index) from log(|<X,beta>|) vs log(Y)...")
    kappa_hat = estimate_kappa(X, Y, beta_hat)
    if kappa_hat is None:
        add("ERROR: Could not estimate kappa (link function index)")
        add("  Possible reasons: < 30 valid tail points, zero denominator, or kappa_hat <= 0")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    add(f"  kappa_hat = {kappa_hat:.6f}")

    # Step 4: hypothesis testing on 2*(kappa+tau)*gamma
    add("")
    add("=" * 80)
    add("Step 4: Testing hypothesis: 0 < 2*(kappa + tau)*gamma < 1")
    add(f"  gamma_hat = {gamma_hat:.6f}")
    add(f"  kappa_hat = {kappa_hat:.6f}")
    add(f"  tau_grid = {tau_grid}")
    add("")

    valid_tau: List[float] = []
    for tau in tau_grid:
        value = 2.0 * (kappa_hat + tau) * gamma_hat  # we compute expression
        condition_pos = value > 0.0
        condition_upper = value < 1.0
        status = "OK" if (condition_pos and condition_upper) else "FAIL"
        add(
            f"  tau = {tau:+.3f} -> 2*(kappa+tau)*gamma = {value:.6f} "
            f"[positive={condition_pos}, <1={condition_upper}] => {status}"
        )
        if status == "OK":
            valid_tau.append(tau)

    add("")
    add("=" * 80)
    if valid_tau:
        add(f"✓ VALID tau values satisfying 0 < 2*(kappa+tau)*gamma < 1: {valid_tau}")
        add(f"  Total valid tau: {len(valid_tau)}/{len(tau_grid)}")
    else:
        add("✗ NO tau in grid satisfies 0 < 2*(kappa+tau)*gamma < 1")
    add("=" * 80)

    # we write all logs to pair-specific file
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# 4. Large-scale driver
# =============================================================================

def main() -> None:
    """we run hypothesis testing for ~100 pairs and log everything"""
    logging.info("Starting large-scale hypothesis verification (no plots)")

    func_data, tickers = build_functional_dataset()  # we load functional data
    all_pairs = list(itertools.permutations(tickers, 2))  # we build all ordered pairs
    logging.info(f"Total candidate pairs: {len(all_pairs)}")

    max_pairs = 120  # we cap number of pairs
    processed = 0
    skipped_reasons: Dict[str, int] = {}  # we track why pairs are skipped

    tau_grid = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]  # we define tau grid

    for ticker_X, ticker_Y in all_pairs:
        if processed >= max_pairs:
            break

        data_X = func_data.get(ticker_X)
        data_Y = func_data.get(ticker_Y)
        if data_X is None or data_Y is None:
            skipped_reasons["missing_data"] = skipped_reasons.get("missing_data", 0) + 1
            continue

        # we align by common dates using pandas Index intersection
        dates_X = data_X["dates"]
        dates_Y = data_Y["dates"]
        try:
            common_dates = dates_X.intersection(dates_Y)  # we use pandas Index intersection
        except Exception as e:
            logging.warning(f"Date intersection failed for {ticker_X} vs {ticker_Y}: {e}")
            skipped_reasons["intersection_error"] = skipped_reasons.get("intersection_error", 0) + 1
            continue

        n_common = len(common_dates)
        if n_common < 100:  # we reduce threshold from 150 to 100
            skipped_reasons["insufficient_common_dates"] = skipped_reasons.get("insufficient_common_dates", 0) + 1
            continue

        idx_X = dates_X.isin(common_dates)  # we use pandas isin (returns Series or array)
        idx_Y = dates_Y.isin(common_dates)

        # we convert to numpy boolean array if needed
        if hasattr(idx_X, 'values'):
            idx_X = idx_X.values
        if hasattr(idx_Y, 'values'):
            idx_Y = idx_Y.values
        idx_X = np.asarray(idx_X, dtype=bool)
        idx_Y = np.asarray(idx_Y, dtype=bool)

        X_curves = data_X["curves"][idx_X]  # we use boolean indexing
        Y_max = data_Y["max_return"][idx_Y]

        # we enforce same length
        n = min(X_curves.shape[0], Y_max.shape[0])
        if n < 50:  # we reduce threshold from 80 to 50
            skipped_reasons["insufficient_samples"] = skipped_reasons.get("insufficient_samples", 0) + 1
            continue
        X_array = X_curves[:n]
        Y_array = Y_max[:n]

        pair_name = f"{ticker_X.replace('.hu.txt', '')}_{ticker_Y.replace('.hu.txt', '')}"
        log_file_path = os.path.join(RESULT_DIR, f"{pair_name}.txt")

        logging.info(f"Processing pair {pair_name} with n={n} (common_dates={n_common})")
        try:
            test_hypotheses_for_pair(X_array, Y_array, tau_grid, pair_name, log_file_path)
            processed += 1
        except Exception as e:
            logging.error(f"Failed on pair {pair_name}: {e}", exc_info=True)
            skipped_reasons["computation_error"] = skipped_reasons.get("computation_error", 0) + 1
            continue

    # we log summary of skipped pairs
    logging.info("Summary of skipped pairs:")
    for reason, count in skipped_reasons.items():
        logging.info(f"  {reason}: {count}")

    logging.info(f"Finished large-scale hypothesis verification for {processed} pairs")


if __name__ == "__main__":
    main()


