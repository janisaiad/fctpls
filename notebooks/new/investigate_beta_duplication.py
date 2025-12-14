"""
Investigation script to understand why beta values are duplicated
across different tau values or stock pairs.
"""

import numpy as np
import sys
import os

# we import adapted functions (1D only)
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    import adapted
    fepls = adapted.fepls
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("adapted", os.path.join(current_dir, "adapted.py"))
    adapted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapted)
    fepls = adapted.fepls

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


def investigate_beta_duplication():
    """we investigate why beta values are duplicated across different stock pairs"""
    print("=" * 80)
    print("Investigating Beta Duplication Issue Across Different Stock Pairs")
    print("=" * 80)
    
    # we load functional dataset
    print("\nLoading functional dataset...")
    func_data, tickers = build_functional_dataset()
    print(f"Loaded {len(tickers)} assets")
    
    # we select multiple pairs to investigate
    test_pairs = []
    for i in range(min(5, len(tickers))):
        for j in range(i+1, min(i+4, len(tickers))):  # we test pairs with same X, different Y
            test_pairs.append((tickers[i], tickers[j]))
    
    if not test_pairs:
        print("ERROR: Not enough tickers to form pairs")
        return
    
    print(f"Testing {len(test_pairs)} pairs")
    
    # we test with a fixed tau value to see if different Y give same beta
    tau_test = -2.0
    print(f"\nUsing fixed tau={tau_test} to compare betas across different stock pairs")
    
    # we store all betas with their pair information
    all_betas = []  # list of (pair_name, tau, beta, signature)
    
    for ticker_X, ticker_Y in test_pairs:
        if ticker_X not in func_data or ticker_Y not in func_data:
            continue
        
        pair_name = f"{ticker_X.replace('.hu.txt', '')}_{ticker_Y.replace('.hu.txt', '')}"
        
        # we align data
        dates_X = func_data[ticker_X]['dates']
        dates_Y = func_data[ticker_Y]['dates']
        common_dates = dates_X.intersection(dates_Y)
        
        if len(common_dates) < 100:
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
            continue
        
        X_array = X_data[:n]
        Y_array = Y_data[:n]
        
        # we compute best_k for this tau
        X_fepls = np.expand_dims(X_array, axis=0)
        Y_fepls = np.expand_dims(Y_array, axis=0)
        Y_sorted = np.sort(Y_fepls[0])[::-1]
        
        # we import bitcoin_concomittant_corr to compute best_k
        try:
            import adapted
            bitcoin_concomittant_corr = adapted.bitcoin_concomittant_corr
        except ImportError:
            bitcoin_concomittant_corr = None
        
        if bitcoin_concomittant_corr is not None:
            m_threshold = int(n / 5)
            corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau_test, m_threshold)
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
                best_k = valid_k_start if len(corr_curve) > valid_k_start else 2
        else:
            best_k = 11
        
        y_n = Y_sorted[best_k] if best_k < len(Y_sorted) else Y_sorted[0]
        y_matrix = y_n * np.ones_like(Y_fepls)
        
        # we compute beta for this pair
        beta = compute_beta_with_tau(X_array, Y_array, tau_test, y_matrix)
        
        if beta is not None:
            beta_signature = np.sum(beta * np.arange(1, len(beta) + 1))
            all_betas.append((pair_name, ticker_X, ticker_Y, tau_test, beta, beta_signature, 
                            np.mean(Y_array), np.std(Y_array), best_k, y_n))
    
    # we now compare betas across different pairs
    print(f"\n{'='*80}")
    print(f"Comparing betas across {len(all_betas)} pairs (tau={tau_test})")
    print(f"{'='*80}")
    print(f"\n{'Pair':<30} | {'X_ticker':<15} | {'Y_ticker':<15} | {'Y_mean':>12} | {'Y_std':>12} | {'best_k':>8} | {'y_n':>12} | {'beta_sig':>18} | {'first_3_beta':>40}")
    print("-" * 180)
    
    for pair_name, ticker_X, ticker_Y, tau, beta, sig, y_mean, y_std, k, y_n in all_betas:
        print(f"{pair_name:<30} | {ticker_X.replace('.hu.txt', ''):<15} | {ticker_Y.replace('.hu.txt', ''):<15} | {y_mean:>12.6f} | {y_std:>12.6f} | {k:>8} | {y_n:>12.6f} | {sig:>18.6f} | {str(beta[:3]):>40}")
    
    # we check for duplicate beta signatures
    print(f"\n{'='*80}")
    print("Checking for duplicate beta signatures across different pairs")
    print(f"{'='*80}")
    
    signature_groups = {}
    for pair_name, ticker_X, ticker_Y, tau, beta, sig, y_mean, y_std, k, y_n in all_betas:
        # we round signature to 2 decimal places for grouping
        sig_rounded = round(sig, 2)
        if sig_rounded not in signature_groups:
            signature_groups[sig_rounded] = []
        signature_groups[sig_rounded].append((pair_name, ticker_X, ticker_Y, beta, sig))
    
    duplicates_found = False
    for sig_rounded, pairs in signature_groups.items():
        if len(pairs) > 1:
            duplicates_found = True
            print(f"\n⚠️  WARNING: {len(pairs)} pairs have similar beta signatures (≈{sig_rounded:.2f}):")
            for pair_name, ticker_X, ticker_Y, beta, sig in pairs:
                print(f"   - {pair_name} (X={ticker_X.replace('.hu.txt', '')}, Y={ticker_Y.replace('.hu.txt', '')}): signature={sig:.6f}, first_3={beta[:3]}")
            
            # we check if betas are actually identical
            print(f"   Checking if betas are identical (within tolerance 1e-6):")
            for i, (pair1, ticker_X1, ticker_Y1, beta1, sig1) in enumerate(pairs):
                for j, (pair2, ticker_X2, ticker_Y2, beta2, sig2) in enumerate(pairs[i+1:], i+1):
                    if np.allclose(beta1, beta2, atol=1e-6):
                        print(f"      ⚠️  {pair1} and {pair2} have IDENTICAL betas!")
                    else:
                        diff = np.linalg.norm(beta1 - beta2)
                        print(f"      {pair1} vs {pair2}: beta difference norm = {diff:.6e}")
    
    if not duplicates_found:
        print("✓ No duplicate beta signatures found - all pairs have different betas")
    
    # we also check if pairs with same X but different Y have similar betas
    print(f"\n{'='*80}")
    print("Checking if pairs with same X but different Y have similar betas")
    print(f"{'='*80}")
    
    x_groups = {}
    for pair_name, ticker_X, ticker_Y, tau, beta, sig, y_mean, y_std, k, y_n in all_betas:
        if ticker_X not in x_groups:
            x_groups[ticker_X] = []
        x_groups[ticker_X].append((pair_name, ticker_Y, beta, sig, y_mean, y_std))
    
    for ticker_X, pairs in x_groups.items():
        if len(pairs) > 1:
            print(f"\nX = {ticker_X.replace('.hu.txt', '')} ({len(pairs)} different Y):")
            for pair_name, ticker_Y, beta, sig, y_mean, y_std in pairs:
                print(f"   Y={ticker_Y.replace('.hu.txt', ''):<15} | Y_mean={y_mean:.6f} | Y_std={y_std:.6f} | beta_sig={sig:.6f} | first_3={beta[:3]}")
            
            # we check if betas are similar for same X
            print(f"   Beta similarity for same X:")
            for i, (pair1, ticker_Y1, beta1, sig1, y_mean1, y_std1) in enumerate(pairs):
                for j, (pair2, ticker_Y2, beta2, sig2, y_mean2, y_std2) in enumerate(pairs[i+1:], i+1):
                    diff = np.linalg.norm(beta1 - beta2)
                    dot_prod = np.dot(beta1, beta2)
                    if np.allclose(beta1, beta2, atol=1e-6):
                        print(f"      ⚠️  {ticker_Y1.replace('.hu.txt', '')} vs {ticker_Y2.replace('.hu.txt', '')}: IDENTICAL betas!")
                    else:
                        print(f"      {ticker_Y1.replace('.hu.txt', '')} vs {ticker_Y2.replace('.hu.txt', '')}: diff_norm={diff:.6e}, dot_product={dot_prod:.6f}")
    
    # we investigate the main script issue: same tau1 + same best_k = same beta1?
    print(f"\n{'='*80}")
    print("Investigating main script issue: Does same tau1 + same best_k give same beta1?")
    print(f"{'='*80}")
    print("This simulates what happens in large_scale_tau_comparison_plotting.py")
    print("where best_k is computed with tau_avg but beta1 is computed with tau1")
    
    # we test: if we have same X, same tau1, but different tau2 (so different tau_avg)
    # and if best_k happens to be the same, will beta1 be the same?
    test_X_ticker = tickers[0] if len(tickers) > 0 else None
    test_Y_ticker = tickers[1] if len(tickers) > 1 else None
    
    if test_X_ticker and test_Y_ticker and test_X_ticker in func_data and test_Y_ticker in func_data:
        dates_X = func_data[test_X_ticker]['dates']
        dates_Y = func_data[test_Y_ticker]['dates']
        common_dates = dates_X.intersection(dates_Y)
        
        if len(common_dates) >= 100:
            idx_X = dates_X.isin(common_dates)
            idx_Y = dates_Y.isin(common_dates)
            if hasattr(idx_X, 'values'):
                idx_X = idx_X.values
            if hasattr(idx_Y, 'values'):
                idx_Y = idx_Y.values
            idx_X = np.asarray(idx_X, dtype=bool)
            idx_Y = np.asarray(idx_Y, dtype=bool)
            
            X_data = func_data[test_X_ticker]['curves'][idx_X]
            Y_data = func_data[test_Y_ticker]['max_return'][idx_Y]
            n = min(X_data.shape[0], Y_data.shape[0])
            if n >= 50:
                X_array = X_data[:n]
                Y_array = Y_data[:n]
                X_fepls = np.expand_dims(X_array, axis=0)
                Y_fepls = np.expand_dims(Y_array, axis=0)
                Y_sorted = np.sort(Y_fepls[0])[::-1]
                
                # we test: tau1=-2.0 with different tau2 values
                tau1_fixed = -2.0
                tau2_values = [-3.0, -2.0, -1.0, 0.0]
                
                print(f"\nTesting with X={test_X_ticker.replace('.hu.txt', '')}, Y={test_Y_ticker.replace('.hu.txt', '')}")
                print(f"tau1 is fixed at {tau1_fixed}, testing different tau2 values")
                print(f"\n{'tau1':>8} | {'tau2':>8} | {'tau_avg':>10} | {'best_k':>8} | {'y_n':>12} | {'beta1_sig':>18} | {'beta1_first3':>40}")
                print("-" * 120)
                
                beta1_results = []
                
                for tau2 in tau2_values:
                    tau_avg = (tau1_fixed + tau2) / 2.0
                    
                    # we compute best_k using tau_avg (like in main script)
                    if bitcoin_concomittant_corr is not None:
                        m_threshold = int(n / 5)
                        corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau_avg, m_threshold)
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
                            best_k = valid_k_start if len(corr_curve) > valid_k_start else 2
                    else:
                        best_k = 11
                    
                    y_n = Y_sorted[best_k] if best_k < len(Y_sorted) else Y_sorted[0]
                    y_matrix = y_n * np.ones_like(Y_fepls)
                    
                    # we compute beta1 using tau1 (like in main script)
                    beta1 = compute_beta_with_tau(X_array, Y_array, tau1_fixed, y_matrix)
                    
                    if beta1 is not None:
                        beta1_sig = np.sum(beta1 * np.arange(1, len(beta1) + 1))
                        beta1_results.append((tau2, tau_avg, best_k, y_n, beta1, beta1_sig))
                        print(f"{tau1_fixed:>8.1f} | {tau2:>8.1f} | {tau_avg:>10.2f} | {best_k:>8} | {y_n:>12.6f} | {beta1_sig:>18.6f} | {str(beta1[:3]):>40}")
                
                # we check if beta1 is the same when best_k is the same
                print(f"\n{'='*80}")
                print("Checking if beta1 is identical when best_k is the same")
                print(f"{'='*80}")
                
                k_groups = {}
                for tau2, tau_avg, best_k, y_n, beta1, beta1_sig in beta1_results:
                    if best_k not in k_groups:
                        k_groups[best_k] = []
                    k_groups[best_k].append((tau2, tau_avg, y_n, beta1, beta1_sig))
                
                for best_k, group in k_groups.items():
                    if len(group) > 1:
                        print(f"\n⚠️  best_k={best_k} is used for {len(group)} different (tau1, tau2) combinations:")
                        for tau2, tau_avg, y_n, beta1, beta1_sig in group:
                            print(f"   tau2={tau2:.1f}, tau_avg={tau_avg:.2f}, y_n={y_n:.6f}, beta1_sig={beta1_sig:.6f}")
                        
                        # we check if beta1 is identical
                        print(f"   Checking if beta1 is identical:")
                        for i, (tau2_1, tau_avg_1, y_n_1, beta1_1, sig1) in enumerate(group):
                            for j, (tau2_2, tau_avg_2, y_n_2, beta1_2, sig2) in enumerate(group[i+1:], i+1):
                                if np.allclose(beta1_1, beta1_2, atol=1e-6):
                                    print(f"      ⚠️  tau2={tau2_1:.1f} and tau2={tau2_2:.1f}: beta1 is IDENTICAL!")
                                else:
                                    diff = np.linalg.norm(beta1_1 - beta1_2)
                                    print(f"      tau2={tau2_1:.1f} vs tau2={tau2_2:.1f}: diff_norm={diff:.6e}")


if __name__ == "__main__":
    investigate_beta_duplication()

