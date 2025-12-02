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
from scipy.stats import linregress
import os
import glob
import itertools
import logging
import time

# Set random seed
np.random.seed(42)

# Setup Logging
LOG_DIR = "../../logs/"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'fepls_run.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # Overwrite each run
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# %% [markdown]
# # 1. Numba Optimized Functions

# %%
@numba.njit(parallel=True, fastmath=False) 
def sort_2d_array(x):
    n,m=np.shape(x)
    for row in numba.prange(n):
        x[row]=np.sort(x[row])
    return x

@numba.njit(parallel=True, fastmath=False) 
def fepls_numba(X,Y,y_matrix,tau): 
    # X of size (N,n,d) and Y of size (N,n)
    # y_matrix of shape (N,n)
    # tau is the tail index of psi
    N=X.shape[0]
    n=X.shape[1]
    d=X.shape[2]
    out=np.zeros((N,d))
    for j in numba.prange(d):
        aux = np.multiply(X[:,:,j],Y**tau) 
        out2 = np.multiply(aux,np.greater_equal(Y,y_matrix)) 
        out[:,j]= np.sum(out2,axis=1)/n 
    
    norms=np.sqrt(np.sum(out**2,axis=1)/d)
    
    # Fix Division by Zero: Check for zero norms
    for i in numba.prange(N):
        if norms[i] < 1e-10:
            norms[i] = 1.0 # Prevent div by zero, vector remains zero
            
    out2 =  out * (norms.reshape((norms.size, 1)))**(-1)
    return out2 

def fepls(X,Y,y_matrix,tau):
    return fepls_numba(X,Y,y_matrix,tau)

@numba.njit(parallel=True, fastmath=False) 
def concomittant_corr(X,Y,Y_sort_index,tau,m): 
    N = X.shape[0]
    n = X.shape[1]
    d = X.shape[2]
    out = np.zeros((N,m))
    YY=np.copy(Y)
    Y_sort=sort_2d_array(YY)
    for k in numba.prange(m):
        y_array = np.zeros((N,n,k+1))
        aux = np.zeros((N,k+1))
        aux3 = Y_sort[:,n-k-1:] 
        for i in numba.prange(k):
            y_array[:,0,i] = Y_sort[:,n-i-1]
            for j_2 in numba.prange(N):
                y_array[j_2,:,i] = y_array[j_2,0,i]
            hat_beta = fepls_numba(X,Y,y_array[:,:,i],tau) 
            for j_1 in numba.prange(N):
                i_c = Y_sort_index[j_1,i]
                aux[j_1,i]=(1/d)*np.sum(np.multiply(hat_beta[j_1,:],X[j_1,i_c,:]))
                out[j_1,k]= np.corrcoef(aux3[j_1,:],aux[j_1,:])[0,1]
    return out

def bitcoin_concomittant_corr(X,Y,tau,m): 
    # Simplified version for N=1 case usually used in main loop
    N = X.shape[0]
    n = X.shape[1]
    d = X.shape[2]
    out = np.zeros((m))
    Y_sort=np.sort(Y,axis=1)
    Y_sort_index = np.argsort(Y,axis=1)
    for k in range(m):
        y_array = np.zeros((N,n,k+1))
        aux = np.zeros((k+1))
        aux3 = Y_sort[0,n-k-1:] 
        for i in range(k):
            y_array[:,:,i] = (Y_sort[0,n-i-1])*np.ones((1,n))
            hat_beta = fepls(X,Y,y_array[:,:,i],tau) 
            i_c = Y_sort_index[0,i]
            aux[i]=(1/d)*np.sum(np.multiply(hat_beta[0,:],X[0,i_c,:]))
            # Check for NaN or constant input to corrcoef
            if np.std(aux3) < 1e-9 or np.std(aux) < 1e-9:
                 out[k] = 0
            else:
                 out[k]= np.corrcoef(aux3,aux)[0,1]
    return np.abs(out)

@numba.njit(parallel=True, fastmath=False)
def get_hill_estimator(ordered_data):
    logs = np.log(ordered_data)
    logs_cumsum = np.cumsum(logs[:-1])
    k_vector = np.arange(1, len(ordered_data))
    m1 = (1./k_vector)*logs_cumsum - logs[1:]
    return m1

def Exponential_QQ_Plot_1D(Y,k):
    n=Y.shape[1]
    out=np.zeros((k))
    out2=np.zeros((k))
    YY=np.sort(Y,axis=1)
    for i in range(k):
        out[i]=np.log((k+1)/(i+1))
        out2[i]=  np.log(YY[0,n-i-1])-np.log(YY[0,-k])
    return np.column_stack((out,out2))

@numba.njit(parallel=False, fastmath=False) 
def Gaussian_kernel(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*np.power(x,2))

@numba.njit(parallel=True, fastmath=False) 
def Epanechnikov_kernel_1D(x):
    out = np.zeros_like(x)
    x=np.asarray(x)
    for i in numba.prange(x.shape[0]):
        if x[i]<=1 and x[i]>=-1: # Fixed range
            out[i] = np.multiply(0.75,1-np.power(x[i],2))
        else:
            out[i]=0
    return out

@numba.njit(parallel=True, fastmath=False) 
def univariate_Nadaraya_weight_2D(X_2D,dimred,x_func,x,h,type,kernel): 
    d=x_func.shape[0]
    # type 2: Y|(X,dimred) = x
    if kernel == 1:
        K_h=Epanechnikov_kernel_1D((np.dot(X_2D,dimred)/d-x)/h) 
    if kernel == 2:
        K_h=Gaussian_kernel((np.dot(X_2D,dimred)/d-x)/h) 
    
    sum_Kh = np.sum(K_h)
    if sum_Kh == 0:
        return np.ones_like(K_h) / K_h.shape[0]
    return K_h/sum_Kh

@numba.njit(parallel=True, fastmath=False) 
def functional_Nadaraya_weight_2D(X_2D,x_func,h,kernel): 
    d=x_func.shape[0]
    aux = (X_2D-x_func*np.ones(d))**2 
    norm = np.sqrt((1/d)*np.sum(aux,axis=1)) 
    if kernel == 1:
        K_h= Epanechnikov_kernel_1D(norm/h) 
    if kernel == 2:
        K_h= Gaussian_kernel(norm/h)
        
    sum_Kh = np.sum(K_h)
    if sum_Kh == 0:
        return np.ones_like(K_h) / K_h.shape[0]
    return K_h/sum_Kh

@numba.njit(parallel=True, fastmath=False)   
def weighted_quantile(data,weight,alpha): 
    sorter = np.argsort(data)
    data = data[sorter]
    weight = weight[sorter]
    weighted_quantiles = np.cumsum(weight) - 0.5 * weight
    weighted_quantiles /= np.sum(weight)
    return np.interp(alpha, weighted_quantiles, data)

@numba.njit(parallel=True, fastmath=False) 
def tail_index_gamma_estimator_2D(Y,weight,alpha,J): 
    subdivision=np.array([(1/s) for s in np.arange(1,J+1)] )
    quantile_data2=weighted_quantile(Y[0,:],weight,alpha) 
    aux=np.zeros((J))
    for j in numba.prange(J):
        quantile_data1=weighted_quantile(Y[0,:],weight,1-subdivision[j]*(1-alpha))
        val = np.log(quantile_data1)-np.log(quantile_data2)
        aux[j] = val / -np.sum(np.log(subdivision))
    return np.sum(aux)

@numba.njit(parallel=True, fastmath=False) 
def plot_quantile_conditional_on_sample_new(X,Y,dimred,x_func,alpha,h_univ_vector,h_func_vector): 
    n=X.shape[1]
    d=X.shape[2]
    # Grid of scalar values <X, x_func>
    proj_vals = np.dot(X[0,:,:],x_func)/d
    S_grid = np.linspace(np.min(proj_vals),np.max(proj_vals),100) # Reduced to 100 for speed
    out = np.zeros((S_grid.shape[0],2))
    
    for p in numba.prange(S_grid.shape[0]):
        # Univariate Weight
        K_h=Gaussian_kernel((np.dot(X[0,:,:],dimred)/d-(np.dot(x_func,dimred)/d)*S_grid[p])/h_univ_vector[0]) # Simplification: using const h
        weight_univ=K_h/np.sum(K_h)        
        
        # Functional Weight
        weight_func=functional_Nadaraya_weight_2D(X[0,:,:],S_grid[p]*x_func,h_func_vector[0],2)
        
        out[p,0]=weighted_quantile(Y[0,:],weight_univ,alpha)
        out[p,1]=weighted_quantile(Y[0,:],weight_func,alpha)
    return out, S_grid

@numba.njit(parallel=True, fastmath=False) 
def beta_func(d):
    grid=np.linspace(0,1,d)
    norm=np.sqrt(np.sum(np.sin(2*np.pi*grid)**2)/d)
    return np.sin(2*np.pi*grid)/norm

# %% [markdown]
# # 2. Data Loading and "Functional" Preprocessing

# %%
def load_stooq_file(filepath):
    """Parses a single Stooq .txt file into a clean DataFrame."""
    try:
        df = pd.read_csv(filepath)
        # Stooq format: <TICKER>,<PER>,<DATE>,<TIME>,...
        # Columns might be standard or capitalized.
        df.columns = [c.replace('<','').replace('>','').lower() for c in df.columns]
        
        # Combine DATE and TIME into a DateTimeIndex
        # DATE is usually YYYYMMDD, TIME is HHMMSS
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str).str.zfill(6), format='%Y%m%d %H%M%S')
        df = df.set_index('datetime')
        
        # Keep relevant columns
        return df[['close', 'vol']]
    except Exception as e:
        logging.warning(f"Error loading {filepath}: {e}")
        return None

def create_functional_data(df_dict, ticker_name, time_grid=None):
    """
    Transforms 5-min time series into a Matrix of shape (Days, TimePoints).
    Row i = Intraday price curve of Day i.
    """
    if ticker_name not in df_dict or df_dict[ticker_name] is None:
        return None, None

    df = df_dict[ticker_name]
    
    # 1. Create a Date column and Time column
    df = df.copy()
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    
    # 2. If no time_grid provided, determine the common intraday times
    # Standard BSE hours 09:00 to 17:00 (approx)
    if time_grid is None:
        # We pick the most common timestamps found in the data
        time_counts = df['Time'].value_counts()
        # Filter times that appear in at least 50% of days to exclude artifacts
        n_days = df['Date'].nunique()
        common_times = time_counts[time_counts > n_days * 0.5].index.sort_values()
        time_grid = common_times
    
    # 3. Pivot: Rows=Date, Cols=Time, Values=Close
    # We need to reindex to ensure exactly the time_grid columns
    pivot_df = df.pivot_table(index='Date', columns='Time', values='close')
    
    # Reindex columns to the standard grid, filling missing intraday values 
    # (forward fill first for missing ticks, then backfill)
    pivot_df = pivot_df.reindex(columns=time_grid)
    pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)
    
    # Drop days that are still completely empty or have too many missing values
    pivot_df = pivot_df.dropna()
    
    return pivot_df, time_grid

# Define path relative to script location to ensure robustness
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(script_dir, "../../data/stooq/hungary/5_hu_txt/data/5 min/hu/bse stocks/"))

# List of all main BSE stocks found in the directory as possible targets
targets = [
    '4ig.hu.txt', 'akko.hu.txt', 'alteo.hu.txt', 'amixa.hu.txt', 'any.hu.txt',
    'appeninn.hu.txt', 'astrasun.hu.txt', 'autowallis.hu.txt', 'bet.hu.txt',
    'bif.hu.txt', 'chome.hu.txt', 'cigpannonia.hu.txt', 'civita.hu.txt', 'delta.hu.txt',
    'dmker.hu.txt', 'dunahouse.hu.txt', 'enefi.hu.txt',
    'epduferr.hu.txt', 'eproliusia.hu.txt', 'esense.hu.txt', 'eu-solar.hu.txt', 'finext.hu.txt', 
    'forras_oe.hu.txt', 'forras_t.hu.txt', 'futuraqua.hu.txt', 'glia.hu.txt', 
    'gloster.hu.txt', 'goodwillphrm.hu.txt',
    'gopd.hu.txt', 'granit.hu.txt', 'gspark.hu.txt', 'kermannit.hu.txt', 'masterplast.hu.txt',
    'mbhbank.hu.txt', 'mbhjb.hu.txt', 'megakran.hu.txt', 'mol.hu.txt', 'mtelekom.hu.txt',
    'multihome.hu.txt', 'nap.hu.txt', 'naturland.hu.txt', 'navigator.hu.txt', 'nordgeneral.hu.txt',
    'nutex.hu.txt', 'o3pnrs.hu.txt', 'opus.hu.txt', 'ormester.hu.txt', 'otp.hu.txt',
    'pannergy.hu.txt', 'pensum.hu.txt', 'polyduct.hu.txt', 'raba.hu.txt', 'richter.hu.txt',
    'splus.hu.txt', 'strt.hu.txt', 'sundell.hu.txt', 'ubm.hu.txt',
    'vertikal.hu.txt', 'vig.hu.txt', 'vvt.hu.txt', 'waberers.hu.txt', 'zwack.hu.txt'
]

logging.info(f"Starting data load from {DATA_DIR}")
data_store = {}
for t in targets:
    path = os.path.join(DATA_DIR, t)
    if os.path.exists(path):
        df = load_stooq_file(path)
        if df is not None:
            data_store[t] = df
    # else:
    #     print(f"File not found: {path}")

logging.info(f"Successfully loaded {len(data_store)} assets.")
print(f"Successfully loaded {len(data_store)} assets.")

# %% [markdown]
# # 3. Alignment and Transformation

# %%
# Create functional data for all loaded assets
master_grid = None
if 'otp.hu.txt' in data_store:
    _, master_grid = create_functional_data(data_store, 'otp.hu.txt')
elif len(data_store) > 0:
    first_key = list(data_store.keys())[0]
    _, master_grid = create_functional_data(data_store, first_key)

if master_grid is not None:
    msg = f"Established master time grid with {len(master_grid)} points per day."
    print(msg)
    logging.info(msg)
else:
    logging.error("Could not establish master grid.")
    raise ValueError("Could not establish master grid.")

# Create matrices
func_data = {}
for t in data_store.keys():
    if master_grid is not None:
        mat, _ = create_functional_data(data_store, t, time_grid=master_grid)
        
        if mat is not None:
            # Method 2: Instantaneous returns (diff)
            log_prices = np.log(mat.values)
            diff_curves = np.diff(log_prices, axis=1)
            
            func_data[t] = {
                'dates': mat.index,
                'curves': diff_curves, # Shape (N_days, d-1)
                'max_return': np.max(diff_curves, axis=1) # The scalar extreme for that day
            }

# %% [markdown]
# # 4. FEPLS Analysis: All Pairs

# %%
# Setup Output Directory
SAVE_DIR = os.path.normpath(os.path.join(script_dir, "../../results/plots/"))
RESULTS_CSV = os.path.normpath(os.path.join(script_dir, "../../results/fepls_stats.csv"))
os.makedirs(SAVE_DIR, exist_ok=True)

# Generate all permutations of loaded assets
all_files = list(data_store.keys())
pairs_to_analyze = list(itertools.permutations(all_files, 2))

print(f"Starting large scale analysis for {len(pairs_to_analyze)} pairs...")
logging.info(f"Starting large scale analysis for {len(pairs_to_analyze)} pairs...")

# Parameters
tau = 1.0
valid_k_start = 5 

count = 0
total_pairs = len(pairs_to_analyze)
results_stats = []

for ticker_X, ticker_Y in pairs_to_analyze:
    count += 1
    
    # Progress every 10 pairs
    if count % 10 == 0:
        print(f"Processed {count}/{total_pairs} pairs...")
        
    # Only proceed if we have functional data for both
    if ticker_X in func_data and ticker_Y in func_data:
        
        try:
            # Align Data
            common_dates = func_data[ticker_X]['dates'].intersection(func_data[ticker_Y]['dates'])
            
            # Skip if not enough overlapping days
            if len(common_dates) < 100:
                continue
                
            idx_X = func_data[ticker_X]['dates'].isin(common_dates)
            idx_Y = func_data[ticker_Y]['dates'].isin(common_dates)
            
            X_data = func_data[ticker_X]['curves'][idx_X]
            Y_data = func_data[ticker_Y]['max_return'][idx_Y]
            
            # Reshape
            X_fepls = np.expand_dims(X_data, axis=0)
            Y_fepls = np.expand_dims(Y_data, axis=0)
            
            n_samples = Y_fepls.shape[1]
            d_points = X_fepls.shape[2]
            m_threshold = int(n_samples / 5)
            
            # 1. Correlation Curve
            corr_curve = bitcoin_concomittant_corr(X_fepls, Y_fepls, tau, m_threshold)
            
            # Find best k
            if len(corr_curve) > valid_k_start:
                # Look for peak in valid range
                valid_curve = corr_curve[valid_k_start:]
                best_k_idx = np.argmax(valid_curve)
                best_k = best_k_idx + valid_k_start
                max_corr = valid_curve[best_k_idx]
                
                # Calculate "Sharpness" (Local Convexity)
                # Sharpness ~ 2*C[k] - C[k-1] - C[k+1]
                # Higher positive value = sharper peak
                sharpness = 0.0
                if 0 < best_k_idx < len(valid_curve) - 1:
                    sharpness = 2 * valid_curve[best_k_idx] - valid_curve[best_k_idx-1] - valid_curve[best_k_idx+1]
            else:
                best_k = valid_k_start if len(corr_curve) > valid_k_start else 2 
                max_corr = 0.0
                sharpness = 0.0
                
            # Log stats
            results_stats.append({
                'X': ticker_X.replace('.hu.txt', ''),
                'Y': ticker_Y.replace('.hu.txt', ''),
                'best_k': best_k,
                'max_corr': max_corr,
                'sharpness': sharpness,
                'n_samples': n_samples
            })
                
            # 2. FEPLS Direction (Beta)
            Y_sorted = np.sort(Y_fepls[0])[::-1]
            y_n = Y_sorted[best_k]
            y_matrix = y_n * np.ones_like(Y_fepls)
            
            E0 = fepls(X_fepls, Y_fepls, y_matrix, tau)
            beta_hat = E0[0,:]
            
            # --- Plotting & Saving ---
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(2, 3)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1:])
            
            name_X = ticker_X.replace('.hu.txt', '')
            name_Y = ticker_Y.replace('.hu.txt', '')
            
            # Plot 1: Correlation
            ax1.plot(corr_curve)
            ax1.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
            ax1.set_title(f'Tail Correlation vs k\n{name_X} -> {name_Y}')
            ax1.set_xlabel('k')
            ax1.set_ylabel('Correlation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Hill Plot
            hill_est = get_hill_estimator(Y_sorted)
            ax2.plot(hill_est)
            ax2.set_title(f'Hill Plot (Tail Index) - {name_Y}')
            ax2.set_xlabel('k')
            ax2.set_xlim(10, m_threshold)
            ax2.set_ylim(0, 1.0)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: QQ Plot
            qq_data = Exponential_QQ_Plot_1D(Y_fepls, best_k)
            if len(qq_data) > 1:
                slope, intercept, r_val, _, _ = linregress(qq_data[:,0], qq_data[:,1])
                ax3.scatter(qq_data[:,0], qq_data[:,1], alpha=0.6)
                ax3.plot(qq_data[:,0], intercept + slope*qq_data[:,0], 'r-', label=f'R2={r_val**2:.2f}')
                ax3.set_title(f'Exp QQ Plot (k={best_k})')
                ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Beta
            ax4.plot(beta_hat, color='purple')
            ax4.set_title(f'FEPLS Direction Beta(t)')
            ax4.set_xlabel('Intraday Time')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Quantile
            h_univ = 0.2 * np.std(np.dot(X_fepls[0], beta_hat)/d_points)
            if h_univ < 1e-6: h_univ = 1e-6 # Avoid zero bandwidth
            h_func = 0.2 * np.mean(np.std(X_fepls[0], axis=0))
            if h_func < 1e-6: h_func = 1e-6
            
            h_univ_vec = h_univ * np.ones(n_samples)
            h_func_vec = h_func * np.ones(n_samples)
            
            try:
                quantiles, s_grid = plot_quantile_conditional_on_sample_new(
                    X_fepls, Y_fepls, 
                    dimred=beta_hat,     
                    x_func=beta_hat,     
                    alpha=0.95,          
                    h_univ_vector=h_univ_vec,
                    h_func_vector=h_func_vec
                )
                
                ax5.plot(s_grid, quantiles[:,0], label='Univariate', linestyle='--')
                ax5.plot(s_grid, quantiles[:,1], label='Functional')
                ax5.set_title(f'Conditional 95% Quantile')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            except Exception as e:
                logging.warning(f"Quantile plot error {name_X}->{name_Y}: {e}")
            
            plt.tight_layout()
            
            # Save
            filename = f"{name_X}_{name_Y}.png"
            plt.savefig(os.path.join(SAVE_DIR, filename))
            plt.close(fig)
            
        except Exception as e:
            logging.error(f"FAILED {ticker_X} -> {ticker_Y}: {e}")
            print(f"Skipping {ticker_X}->{ticker_Y} due to error (check logs)")
            continue

# Save Statistics
print("Analysis complete. Saving statistics...")
df_stats = pd.DataFrame(results_stats)
if not df_stats.empty:
    df_stats = df_stats.sort_values('max_corr', ascending=False)
    df_stats.to_csv(RESULTS_CSV, index=False)
    print(f"Stats saved to {RESULTS_CSV}")
    print("\nTop 10 Pairs by Tail Correlation:")
    print(df_stats.head(10)[['X', 'Y', 'best_k', 'max_corr']])
    
    print("\nTop 5 Sharpest Peaks (Most convex local maxima):")
    print(df_stats.sort_values('sharpness', ascending=False).head(5)[['X', 'Y', 'best_k', 'max_corr', 'sharpness']])
else:
    print("No results generated.")

logging.info("Run completed.")