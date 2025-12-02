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
import matplotlib.pyplot as plt
import os
import logging

# Set random seed
np.random.seed(42)

# Setup Output
script_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.normpath(os.path.join(script_dir, "../../results/plots_extreme_days/"))
os.makedirs(SAVE_DIR, exist_ok=True)

# %% [markdown]
# # 1. Core Functions (Simplified for Visualization)

# %%
@numba.njit(parallel=True, fastmath=False) 
def fepls_numba(X,Y,y_matrix,tau): 
    N=X.shape[0]
    n=X.shape[1]
    d=X.shape[2]
    out=np.zeros((N,d))
    for j in numba.prange(d):
        aux = np.multiply(X[:,:,j],Y**tau) 
        out2 = np.multiply(aux,np.greater_equal(Y,y_matrix)) 
        out[:,j]= np.sum(out2,axis=1)/n 
    
    norms=np.sqrt(np.sum(out**2,axis=1)/d)
    for i in numba.prange(N):
        if norms[i] < 1e-10: norms[i] = 1.0
    out2 =  out * (norms.reshape((norms.size, 1)))**(-1)
    return out2 

def fepls(X,Y,y_matrix,tau):
    return fepls_numba(X,Y,y_matrix,tau)

# %% [markdown]
# # 2. Data Loading Logic (Reused)

# %%
def load_stooq_file(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.replace('<','').replace('>','').lower() for c in df.columns]
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str).str.zfill(6), format='%Y%m%d %H%M%S')
        df = df.set_index('datetime')
        return df[['close']]
    except Exception as e:
        return None

def create_functional_data(df_dict, ticker_name, time_grid=None):
    if ticker_name not in df_dict or df_dict[ticker_name] is None: return None, None
    df = df_dict[ticker_name].copy()
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    
    if time_grid is None:
        time_counts = df['Time'].value_counts()
        n_days = df['Date'].nunique()
        common_times = time_counts[time_counts > n_days * 0.5].index.sort_values()
        time_grid = common_times
    else:
        # Ensure time_grid is valid
        pass
    
    pivot_df = df.pivot_table(index='Date', columns='Time', values='close')
    pivot_df = pivot_df.reindex(columns=time_grid).ffill(axis=1).bfill(axis=1).dropna()
    return pivot_df, time_grid

# Define Paths
DATA_DIR = os.path.normpath(os.path.join(script_dir, "../../data/stooq/hungary/5_hu_txt/data/5 min/hu/bse stocks/"))

# %% [markdown]
# # 3. Focus Analysis on Interesting Pairs

# %%
# Pairs identified from your previous run (High Corr or Sharp Peaks)
interesting_pairs = [
    ('akko.hu.txt', 'vertikal.hu.txt', 5),      # Top Corr
    ('any.hu.txt', 'vertikal.hu.txt', 5),       # Top Corr
    ('esense.hu.txt', 'mtelekom.hu.txt', 11),   # Sharpest Peak
    ('autowallis.hu.txt', 'mtelekom.hu.txt', 8),# Sharp Peak
    ('bif.hu.txt', 'mol.hu.txt', 7),            # Sharp Peak
    ('otp.hu.txt', 'cigpannonia.hu.txt', 5),    # Major Bank involved
    ('mol.hu.txt', 'otp.hu.txt', 5)             # Energy -> Bank (Classic)
]

# Load needed data only
needed_tickers = set()
for x, y, k in interesting_pairs:
    needed_tickers.add(x)
    needed_tickers.add(y)

print(f"Loading {len(needed_tickers)} specific assets for deep visualization...")

data_store = {}
for t in needed_tickers:
    path = os.path.join(DATA_DIR, t)
    if os.path.exists(path):
        df = load_stooq_file(path)
        if df is not None:
            data_store[t] = df

# %% [markdown]
# # 4. Visualization Loop

# %%
# Master Grid Setup (using first available)
master_grid = None
if len(data_store) > 0:
    first = list(data_store.keys())[0]
    _, master_grid = create_functional_data(data_store, first)

func_data = {}
for t in data_store:
    mat, _ = create_functional_data(data_store, t, master_grid)
    if mat is not None:
        log_prices = np.log(mat.values)
        diff_curves = np.diff(log_prices, axis=1)
        func_data[t] = {
            'dates': mat.index,
            'curves': diff_curves,
            'max_return': np.max(diff_curves, axis=1)
        }

print("Generating detailed plots...")

for ticker_X, ticker_Y, k_val in interesting_pairs:
    if ticker_X in func_data and ticker_Y in func_data:
        
        # Align
        common = func_data[ticker_X]['dates'].intersection(func_data[ticker_Y]['dates'])
        idx_X = func_data[ticker_X]['dates'].isin(common)
        idx_Y = func_data[ticker_Y]['dates'].isin(common)
        
        X_aligned = func_data[ticker_X]['curves'][idx_X]
        Y_aligned = func_data[ticker_Y]['max_return'][idx_Y]
        Dates_aligned = func_data[ticker_X]['dates'][idx_X] # Keep dates to identify specific days
        
        # Prepare FEPLS inputs
        X_fepls = np.expand_dims(X_aligned, axis=0)
        Y_fepls = np.expand_dims(Y_aligned, axis=0)
        
        # Get Beta for specific k
        Y_sorted = np.sort(Y_fepls[0])[::-1]
        # Safety check for k
        current_k = k_val
        if current_k >= len(Y_sorted): current_k = len(Y_sorted) - 1
        
        y_threshold = Y_sorted[current_k] # The threshold value
        
        # Compute Beta
        y_matrix = y_threshold * np.ones_like(Y_fepls)
        E0 = fepls(X_fepls, Y_fepls, y_matrix, 1.0)
        beta_hat = E0[0,:]
        
        # --- Identify the k Extreme Days ---
        # Find indices where Y >= threshold
        # Note: floating point comparison might need tolerance, but usually OK for raw extraction
        extreme_indices = np.where(Y_aligned >= y_threshold)[0]
        
        # If we have too many (tied values), just take top k
        if len(extreme_indices) > current_k + 10: # arbitrary buffer
             # Sort by Y value to get true top
             sub_y = Y_aligned[extreme_indices]
             # Get indices of top k in the sub-array
             top_idx_sub = np.argsort(sub_y)[::-1][:current_k+1]
             extreme_indices = extreme_indices[top_idx_sub]
        
        X_extreme_curves = X_aligned[extreme_indices]
        extreme_dates = Dates_aligned[extreme_indices]
        
        X_mean_extreme = np.mean(X_extreme_curves, axis=0)
        
        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        name_X = ticker_X.replace('.hu.txt', '')
        name_Y = ticker_Y.replace('.hu.txt', '')
        
        # 1. FEPLS Beta
        ax1.plot(beta_hat, color='purple', linewidth=2, label='FEPLS Beta (Direction)')
        ax1.set_title(f"FEPLS Direction $\\beta(t)$ (k={current_k})\nPredictor: {name_X} -> Target: {name_Y}")
        ax1.set_ylabel("Weight / Influence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Underlying Curves
        # Plot all curves in light grey
        # Plot extreme curves in distinct colors (or alpha red)
        # Plot mean extreme in bold red
        
        t_axis = np.arange(X_aligned.shape[1])
        
        # Background: A sample of random days (non-extreme) for context
        non_extreme_idx = np.where(Y_aligned < y_threshold)[0]
        if len(non_extreme_idx) > 0:
            sample_size = min(50, len(non_extreme_idx))
            sample_idx = np.random.choice(non_extreme_idx, size=sample_size, replace=False)
            for idx in sample_idx:
                ax2.plot(t_axis, X_aligned[idx], color='grey', alpha=0.1)
            
        # Foreground: The Extreme Days
        for i in range(len(X_extreme_curves)):
            ax2.plot(t_axis, X_extreme_curves[i], color='orange', alpha=0.6, linewidth=1)
            
        # Mean of Extremes
        ax2.plot(t_axis, X_mean_extreme, color='red', linewidth=3, label=f'Mean of Top {len(X_extreme_curves)} Extremes')
        
        ax2.set_title(f"Intraday Curves of {name_X} on the {len(X_extreme_curves)} days where {name_Y} was extreme")
        ax2.set_xlabel("Intraday Time Index")
        ax2.set_ylabel("Log Returns")
        
        # Add a dummy line for the grey ones in legend
        ax2.plot([], [], color='grey', alpha=0.3, label='Normal Days (Background)')
        ax2.plot([], [], color='orange', alpha=0.6, label='Extreme Days')
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"VISUAL_{name_X}_{name_Y}_k{current_k}.png"
        plt.savefig(os.path.join(SAVE_DIR, filename))
        plt.close(fig)
        print(f"Saved {filename}")

print("Visualization complete.")

