"""
Investigation script to understand why beta1 and beta2 values are duplicated
across different tau values or stock pairs.
"""

import numpy as np
import sys
import os

# we add utils to path
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'utils')
sys.path.append(utils_dir)

# we import FEPLS functions
try:
    from fepls_nd import fepls_nd
except ImportError:
    import importlib.util
    fepls_nd_path = os.path.join(utils_dir, 'fepls_nd.py')
    if os.path.exists(fepls_nd_path):
        spec = importlib.util.spec_from_file_location("fepls_nd", fepls_nd_path)
        fepls_nd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fepls_nd_module)
        fepls_nd = fepls_nd_module.fepls_nd
    else:
        raise ImportError(f"Could not find fepls_nd.py at {fepls_nd_path}")

# we import adapted functions
try:
    import adapted
    fepls_original = adapted.fepls
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("adapted", os.path.join(current_dir, "adapted.py"))
    adapted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapted)
    fepls_original = adapted.fepls

from large_scale_hypothesisverif import build_functional_dataset


def compute_beta_with_tau(X_data, Y_data, tau, y_matrix):
    """we compute beta using fepls_nd for given tau"""
    X_fepls = np.expand_dims(X_data, axis=0)  # shape (1, n, d)
    Y_fepls = np.expand_dims(Y_data, axis=0)  # shape (1, n)
    
    Y_abs = np.abs(Y_fepls)
    y_matrix_abs = np.abs(y_matrix)
    epsilon = 1e-8 if tau >= 0 else 1e-6
    Y_abs = np.maximum(Y_abs, epsilon)
    y_matrix_abs = np.maximum(y_matrix_abs, epsilon)
    
    beta = fepls_nd(X_fepls, Y_abs, y_matrix_abs, tau, separate_directions=False)
    if beta is not None:
        beta = beta[0, :]  # shape (d,)
        # we normalize
        norm = np.linalg.norm(beta)
        if norm > 1e-10:
            beta = beta / norm
        return beta
    return None


def investigate_beta_duplication():
    """we investigate why beta values are duplicated"""
    print("=" * 80)
    print("Investigating Beta Duplication Issue")
    print("=" * 80)
    
    # we load functional dataset
    print("\nLoading functional dataset...")
    func_data, tickers = build_functional_dataset()
    print(f"Loaded {len(tickers)} assets")
    
    # we select a few pairs to investigate
    test_pairs = [
        (tickers[0], tickers[1]) if len(tickers) >= 2 else None,
        (tickers[0], tickers[2]) if len(tickers) >= 3 else None,
    ]
    test_pairs = [p for p in test_pairs if p is not None]
    
    if not test_pairs:
        print("ERROR: Not enough tickers to form pairs")
        return
    
    # we test different tau values
    tau_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    
    for ticker_X, ticker_Y in test_pairs[:2]:  # we limit to first 2 pairs
        if ticker_X not in func_data or ticker_Y not in func_data:
            continue
        
        print(f"\n{'='*80}")
        print(f"Investigating pair: {ticker_X} -> {ticker_Y}")
        print(f"{'='*80}")
        
        # we align data
        dates_X = func_data[ticker_X]['dates']
        dates_Y = func_data[ticker_Y]['dates']
        common_dates = dates_X.intersection(dates_Y)
        
        if len(common_dates) < 100:
            print(f"Skipping: insufficient common dates ({len(common_dates)})")
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
            print(f"Skipping: insufficient samples ({n})")
            continue
        
        X_array = X_data[:n]
        Y_array = Y_data[:n]
        
        print(f"\nData shape: X={X_array.shape}, Y={Y_array.shape}")
        print(f"X statistics: mean={np.mean(X_array):.6f}, std={np.std(X_array):.6f}")
        print(f"Y statistics: mean={np.mean(Y_array):.6f}, std={np.std(Y_array):.6f}")
        
        # we compute best_k once (using tau=1.0 as reference)
        X_fepls = np.expand_dims(X_array, axis=0)
        Y_fepls = np.expand_dims(Y_array, axis=0)
        Y_sorted = np.sort(Y_fepls[0])[::-1]
        best_k = 11  # we use a fixed k for investigation
        y_n = Y_sorted[best_k] if best_k < len(Y_sorted) else Y_sorted[0]
        y_matrix = y_n * np.ones_like(Y_fepls)
        
        print(f"\nUsing fixed k={best_k}, y_n={y_n:.6f}")
        print(f"Number of extreme samples (Y >= y_n): {np.sum(Y_fepls[0] >= y_n)}")
        
        # we compute beta for different tau values
        print(f"\n{'tau':>8} | {'beta_norm':>12} | {'beta_signature':>18} | {'first_3_values':>40} | {'identical_to_prev':>20}")
        print("-" * 110)
        
        beta_history = []
        prev_beta = None
        
        for tau in tau_values:
            beta = compute_beta_with_tau(X_array, Y_array, tau, y_matrix)
            
            if beta is not None:
                beta_norm = np.linalg.norm(beta)
                beta_signature = np.sum(beta * np.arange(1, len(beta) + 1))
                beta_first3 = beta[:3]
                
                # we check if this beta is identical to previous ones
                identical_to = None
                for i, prev_b in enumerate(beta_history):
                    if np.allclose(beta, prev_b, atol=1e-6):
                        identical_to = f"tau={tau_values[i]:.1f}"
                        break
                
                if identical_to:
                    print(f"{tau:>8.1f} | {beta_norm:>12.6f} | {beta_signature:>18.6f} | {str(beta_first3):>40} | ⚠️  {identical_to:>15}")
                else:
                    print(f"{tau:>8.1f} | {beta_norm:>12.6f} | {beta_signature:>18.6f} | {str(beta_first3):>40} | {'':>20}")
                
                beta_history.append(beta.copy())
                prev_beta = beta
            else:
                print(f"{tau:>8.1f} | {'ERROR':>12} | {'':>18} | {'':>40} | {'':>20}")
        
        # we investigate the computation process for two different tau values
        print(f"\n{'='*80}")
        print("Detailed investigation: comparing tau=-2.0 and tau=-1.0")
        print(f"{'='*80}")
        
        for tau in [-2.0, -1.0]:
            print(f"\n--- tau = {tau} ---")
            X_fepls = np.expand_dims(X_array, axis=0)
            Y_fepls = np.expand_dims(Y_array, axis=0)
            
            Y_abs = np.abs(Y_fepls)
            epsilon = 1e-8 if tau >= 0 else 1e-6
            Y_abs = np.maximum(Y_abs, epsilon)
            y_matrix_abs = np.maximum(y_matrix, epsilon)
            
            print(f"Y_abs stats: min={np.min(Y_abs):.6e}, max={np.max(Y_abs):.6e}, mean={np.mean(Y_abs):.6e}")
            print(f"Y_abs signature: {np.sum(Y_abs * np.arange(1, Y_abs.size + 1).reshape(Y_abs.shape)):.6f}")
            
            # we compute weights
            weights = Y_abs[0, :] ** tau
            extreme_mask = Y_fepls[0] >= y_n
            if np.any(extreme_mask):
                extreme_weights = weights[extreme_mask]
                print(f"Extreme weights (Y**tau) stats: min={np.min(extreme_weights):.6e}, max={np.max(extreme_weights):.6e}, mean={np.mean(extreme_weights):.6e}")
                print(f"Number of extreme samples: {np.sum(extreme_mask)}")
            
            # we compute beta
            beta = fepls_nd(X_fepls, Y_abs, y_matrix_abs, tau, separate_directions=False)
            if beta is not None:
                beta = beta[0, :]
                print(f"Beta before normalization: norm={np.linalg.norm(beta):.6f}, first_3={beta[:3]}")
                norm = np.linalg.norm(beta)
                if norm > 1e-10:
                    beta = beta / norm
                print(f"Beta after normalization: norm={np.linalg.norm(beta):.6f}, first_3={beta[:3]}")
                print(f"Beta signature: {np.sum(beta * np.arange(1, len(beta) + 1)):.6f}")


if __name__ == "__main__":
    investigate_beta_duplication()

