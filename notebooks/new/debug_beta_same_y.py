"""
Debug script to investigate why beta is the same for different stocks Y
"""

import numpy as np
import sys
import os

# we import adapted functions (1D only)
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    import adapted
    fepls = adapted.fepls
    bitcoin_concomittant_corr = adapted.bitcoin_concomittant_corr
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("adapted", os.path.join(current_dir, "adapted.py"))
    adapted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapted)
    fepls = adapted.fepls
    bitcoin_concomittant_corr = adapted.bitcoin_concomittant_corr

from large_scale_hypothesisverif import build_functional_dataset


def compute_beta_with_tau(X_data, Y_data, tau, y_matrix):
    """we compute beta using fepls (1D) for given tau"""
    X_fepls = np.expand_dims(X_data, axis=0)  # shape (1, n, d)
    Y_fepls = np.expand_dims(Y_data, axis=0)  # shape (1, n)
    
    Y_abs = np.abs(Y_fepls)
    y_matrix_abs = np.abs(y_matrix)
    epsilon = 1e-8 if tau >= 0 else 1e-6
    Y_abs = np.maximum(Y_abs, epsilon)
    y_matrix_abs = np.maximum(y_matrix_abs, epsilon)
    
    beta = fepls(X_fepls, Y_abs, y_matrix_abs, tau)
    if beta is not None:
        beta = beta[0, :]  # shape (d,)
        # we normalize
        norm = np.linalg.norm(beta)
        if norm > 1e-10:
            beta = beta / norm
        return beta
    return None


def debug_beta_for_different_y():
    """we debug why beta is the same for different stocks Y"""
    print("=" * 80)
    print("DEBUG: Investigating why beta is the same for different stocks Y")
    print("=" * 80)
    
    # we load functional dataset
    print("\nLoading functional dataset...")
    func_data, tickers = build_functional_dataset()
    print(f"Loaded {len(tickers)} assets")
    
    # we choose a fixed X (e.g., first ticker) - THIS IS THE KEY: same X, different Y
    ticker_X = tickers[0]
    print(f"\nUsing fixed X: {ticker_X}")
    print("⚠️  IMPORTANT: We're testing with the SAME X but DIFFERENT Y stocks")
    print("   If betas are similar, it means the method is not sensitive enough to Y differences")
    
    # we test with multiple different Y stocks
    test_y_tickers = tickers[1:6]  # test with 5 different Y stocks
    tau = 1.0  # fixed tau
    
    print(f"\nTesting with {len(test_y_tickers)} different Y stocks: {test_y_tickers}")
    print(f"Using tau = {tau}")
    
    # we get X data
    if ticker_X not in func_data:
        print(f"ERROR: {ticker_X} not in func_data")
        return
    
    X_data = func_data[ticker_X]['curves']
    dates_X = func_data[ticker_X]['dates']
    
    results = []
    
    for ticker_Y in test_y_tickers:
        if ticker_Y not in func_data:
            print(f"Skipping {ticker_Y}: not in func_data")
            continue
        
        # we align data
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
        
        X_aligned = X_data[idx_X]
        Y_data = func_data[ticker_Y]['max_return'][idx_Y]
        
        print(f"\n{'='*80}")
        print(f"Analyzing pair: {ticker_X} -> {ticker_Y}")
        print(f"  X shape: {X_aligned.shape}")
        print(f"  Y shape: {Y_data.shape}")
        print(f"  Y stats: min={np.min(Y_data):.6f}, max={np.max(Y_data):.6f}, mean={np.mean(Y_data):.6f}, std={np.std(Y_data):.6f}")
        
        # we compute best_k
        X_fepls = np.expand_dims(X_aligned, axis=0)
        Y_fepls = np.expand_dims(Y_data, axis=0)
        n_samples = Y_fepls.shape[1]
        m_threshold = int(n_samples / 5)
        
        corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau, m_threshold)
        corr_curve = np.nan_to_num(corr_curve, nan=0.0, posinf=0.0, neginf=0.0)
        
        # we find best_k
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
            else:
                best_k_idx = np.argmax(sharpness_values)
                best_k = best_k_idx + valid_k_start
        else:
            best_k = valid_k_start
        
        print(f"  best_k: {best_k}")
        
        # we compute y_matrix
        Y_sorted = np.sort(Y_fepls[0])[::-1]
        y_n = Y_sorted[best_k] if best_k < len(Y_sorted) else Y_sorted[0]
        y_matrix = y_n * np.ones_like(Y_fepls)
        
        print(f"  y_n (threshold): {y_n:.6f}")
        print(f"  Number of extremes (Y >= y_n): {np.sum(Y_fepls[0] >= y_n)}")
        
        # we compute beta
        beta = compute_beta_with_tau(X_aligned, Y_data, tau, y_matrix[0, :])
        
        if beta is None:
            print(f"  ERROR: Could not compute beta")
            continue
        
        # we compute beta signature
        beta_signature = np.sum(beta * np.arange(1, len(beta) + 1))
        beta_norm = np.linalg.norm(beta)
        beta_first_5 = beta[:5]
        
        print(f"  beta norm: {beta_norm:.6f}")
        print(f"  beta first 5 values: {beta_first_5}")
        print(f"  beta signature: {beta_signature:.6f}")
        
        # we check weights
        Y_abs = np.abs(Y_fepls)
        epsilon = 1e-8 if tau >= 0 else 1e-6
        Y_abs = np.maximum(Y_abs, epsilon)
        extreme_mask = Y_fepls[0] >= y_n
        if np.any(extreme_mask):
            weights = Y_abs[0, extreme_mask]**tau
            print(f"  weights (Y**tau) for extremes: min={np.min(weights):.6e}, max={np.max(weights):.6e}, mean={np.mean(weights):.6e}")
        
        results.append({
            'ticker_Y': ticker_Y,
            'beta': beta,
            'beta_signature': beta_signature,
            'best_k': best_k,
            'y_n': y_n,
            'Y_stats': {
                'min': np.min(Y_data),
                'max': np.max(Y_data),
                'mean': np.mean(Y_data),
                'std': np.std(Y_data),
            }
        })
    
    # we compare betas
    print(f"\n{'='*80}")
    print("COMPARISON OF BETAS:")
    print("=" * 80)
    
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results[i+1:], start=i+1):
            beta_diff = np.linalg.norm(r1['beta'] - r2['beta'])
            beta_dot = np.dot(r1['beta'], r2['beta'])
            sig_diff = abs(r1['beta_signature'] - r2['beta_signature'])
            
            print(f"\n{r1['ticker_Y']} vs {r2['ticker_Y']}:")
            print(f"  beta difference (L2 norm): {beta_diff:.6e}")
            print(f"  beta dot product: {beta_dot:.6f}")
            print(f"  signature difference: {sig_diff:.6e}")
            print(f"  best_k: {r1['best_k']} vs {r2['best_k']}")
            print(f"  y_n: {r1['y_n']:.6f} vs {r2['y_n']:.6f}")
            print(f"  Y mean: {r1['Y_stats']['mean']:.6f} vs {r2['Y_stats']['mean']:.6f}")
            print(f"  Y std: {r1['Y_stats']['std']:.6f} vs {r2['Y_stats']['std']:.6f}")
            
            if beta_diff < 1e-6:
                print(f"  ⚠️  WARNING: Betas are IDENTICAL (diff < 1e-6)!")
                print(f"     This is a PROBLEM: same X with different Y should give different betas!")
            elif beta_dot > 0.99:
                print(f"  ⚠️  WARNING: Betas are VERY SIMILAR (dot product > 0.99)!")
                print(f"     This is a PROBLEM: same X with different Y should give different betas!")
            elif beta_dot > 0.7:
                print(f"  ⚠️  CAUTION: Betas are quite similar (dot product > 0.7)")
                print(f"     This might indicate the method is not sensitive enough to Y differences")
            
            # we check if best_k and y_n are the same (which could explain similarity)
            if r1['best_k'] == r2['best_k'] and abs(r1['y_n'] - r2['y_n']) < 1e-6:
                print(f"  ⚠️  NOTE: Same best_k ({r1['best_k']}) and same y_n ({r1['y_n']:.6f})")
                print(f"     This could explain why betas are similar!")


def debug_beta_for_different_tau():
    """we debug why beta might be the same for different tau values (same X, same Y)"""
    print("=" * 80)
    print("DEBUG: Investigating if different tau values give different betas")
    print("=" * 80)
    
    # we load functional dataset
    print("\nLoading functional dataset...")
    func_data, tickers = build_functional_dataset()
    print(f"Loaded {len(tickers)} assets")
    
    # we choose a fixed X and Y pair
    if len(tickers) < 2:
        print("ERROR: Need at least 2 tickers")
        return
    
    ticker_X = tickers[0]
    ticker_Y = tickers[1]
    print(f"\nUsing fixed pair: X={ticker_X}, Y={ticker_Y}")
    print("⚠️  IMPORTANT: We're testing with the SAME X and SAME Y but DIFFERENT tau values")
    print("   If betas are similar, it means the method is not sensitive enough to tau")
    
    # we test with multiple different tau values
    tau_values = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    print(f"\nTesting with {len(tau_values)} different tau values: {tau_values}")
    
    # we get X and Y data
    if ticker_X not in func_data or ticker_Y not in func_data:
        print(f"ERROR: {ticker_X} or {ticker_Y} not in func_data")
        return
    
    X_data = func_data[ticker_X]['curves']
    dates_X = func_data[ticker_X]['dates']
    dates_Y = func_data[ticker_Y]['dates']
    common_dates = dates_X.intersection(dates_Y)
    
    if len(common_dates) < 100:
        print(f"ERROR: insufficient common dates ({len(common_dates)})")
        return
    
    idx_X = dates_X.isin(common_dates)
    idx_Y = dates_Y.isin(common_dates)
    
    if hasattr(idx_X, 'values'):
        idx_X = idx_X.values
    if hasattr(idx_Y, 'values'):
        idx_Y = idx_Y.values
    
    X_aligned = X_data[idx_X]
    Y_data = func_data[ticker_Y]['max_return'][idx_Y]
    
    print(f"\nData shapes: X={X_aligned.shape}, Y={Y_data.shape}")
    print(f"Y stats: min={np.min(Y_data):.6f}, max={np.max(Y_data):.6f}, mean={np.mean(Y_data):.6f}, std={np.std(Y_data):.6f}")
    
    X_fepls = np.expand_dims(X_aligned, axis=0)
    Y_fepls = np.expand_dims(Y_data, axis=0)
    n_samples = Y_fepls.shape[1]
    m_threshold = int(n_samples / 5)
    
    results = []
    
    for tau in tau_values:
        print(f"\n{'='*80}")
        print(f"Testing tau = {tau:.2f}")
        print(f"{'='*80}")
        
        # we compute best_k for this tau
        corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau, m_threshold)
        corr_curve = np.nan_to_num(corr_curve, nan=0.0, posinf=0.0, neginf=0.0)
        
        # we find best_k
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
            else:
                best_k_idx = np.argmax(sharpness_values)
                best_k = best_k_idx + valid_k_start
        else:
            best_k = valid_k_start
        
        print(f"  best_k: {best_k}")
        
        # we compute y_matrix
        Y_sorted = np.sort(Y_fepls[0])[::-1]
        y_n = Y_sorted[best_k] if best_k < len(Y_sorted) else Y_sorted[0]
        y_matrix = y_n * np.ones_like(Y_fepls)
        
        print(f"  y_n (threshold): {y_n:.6f}")
        print(f"  Number of extremes (Y >= y_n): {np.sum(Y_fepls[0] >= y_n)}")
        
        # we compute beta
        beta = compute_beta_with_tau(X_aligned, Y_data, tau, y_matrix[0, :])
        
        if beta is None:
            print(f"  ERROR: Could not compute beta")
            continue
        
        # we compute beta signature
        beta_signature = np.sum(beta * np.arange(1, len(beta) + 1))
        beta_norm = np.linalg.norm(beta)
        beta_first_5 = beta[:5]
        
        print(f"  beta norm: {beta_norm:.6f}")
        print(f"  beta first 5 values: {beta_first_5}")
        print(f"  beta signature: {beta_signature:.6f}")
        
        # we check weights
        Y_abs = np.abs(Y_fepls)
        epsilon = 1e-8 if tau >= 0 else 1e-6
        Y_abs = np.maximum(Y_abs, epsilon)
        extreme_mask = Y_fepls[0] >= y_n
        if np.any(extreme_mask):
            weights = Y_abs[0, extreme_mask]**tau
            print(f"  weights (Y**tau) for extremes: min={np.min(weights):.6e}, max={np.max(weights):.6e}, mean={np.mean(weights):.6e}")
        
        results.append({
            'tau': tau,
            'beta': beta,
            'beta_signature': beta_signature,
            'best_k': best_k,
            'y_n': y_n,
        })
    
    # we compare betas across different tau values
    print(f"\n{'='*80}")
    print("COMPARISON OF BETAS ACROSS DIFFERENT TAU VALUES:")
    print("=" * 80)
    
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results[i+1:], start=i+1):
            beta_diff = np.linalg.norm(r1['beta'] - r2['beta'])
            beta_dot = np.dot(r1['beta'], r2['beta'])
            sig_diff = abs(r1['beta_signature'] - r2['beta_signature'])
            tau_diff = abs(r1['tau'] - r2['tau'])
            
            print(f"\ntau={r1['tau']:.2f} vs tau={r2['tau']:.2f} (diff={tau_diff:.2f}):")
            print(f"  beta difference (L2 norm): {beta_diff:.6e}")
            print(f"  beta dot product: {beta_dot:.6f}")
            print(f"  signature difference: {sig_diff:.6e}")
            print(f"  best_k: {r1['best_k']} vs {r2['best_k']}")
            print(f"  y_n: {r1['y_n']:.6f} vs {r2['y_n']:.6f}")
            
            if beta_diff < 1e-6:
                print(f"  ⚠️  WARNING: Betas are IDENTICAL (diff < 1e-6)!")
                print(f"     This is a PROBLEM: different tau should give different betas!")
            elif beta_dot > 0.99:
                print(f"  ⚠️  WARNING: Betas are VERY SIMILAR (dot product > 0.99)!")
                print(f"     This is a PROBLEM: different tau should give different betas!")
            elif beta_dot > 0.7:
                print(f"  ⚠️  CAUTION: Betas are quite similar (dot product > 0.7)")
                print(f"     This might indicate the method is not sensitive enough to tau")
            
            # we check if best_k and y_n are the same (which could explain similarity)
            if r1['best_k'] == r2['best_k'] and abs(r1['y_n'] - r2['y_n']) < 1e-6:
                print(f"  ⚠️  NOTE: Same best_k ({r1['best_k']}) and same y_n ({r1['y_n']:.6f})")
                print(f"     This could explain why betas are similar!")
                print(f"     But even with same best_k, different tau should give different weights!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tau":
        debug_beta_for_different_tau()
    else:
        debug_beta_for_different_y()

