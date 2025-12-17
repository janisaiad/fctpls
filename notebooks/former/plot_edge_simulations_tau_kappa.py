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
# # Edge-of-Inequality Simulations: Plotting Results (sim.py style)
# 
# This script loads pre-computed simulation results and generates the same plots as sim.py:
# - MSE, Variance, Bias^2 vs k for different kappa
# - k_optimal vs kappa
# Data structure: `data/fepls_grid/rho_{rho}/gamma_{gamma}/kappa_{kappa}/tau_{tau}/d_{d}/results_n{n}_k{k}.npz`

# %%
###################### Packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# we configure matplotlib settings
plt.rcParams['figure.figsize'] = [15, 10]
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
###################### Configuration
# we get the script directory and use it as base for relative paths
# we try multiple methods to find the project root
try:
    if '__file__' in globals():
        SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
    else:
        SCRIPT_DIR = Path.cwd()
except:
    SCRIPT_DIR = Path.cwd()

# we check if data directory exists relative to script, otherwise use current working directory
if (SCRIPT_DIR / "data" / "fepls_grid").exists():
    BASE_DIR = SCRIPT_DIR / "data" / "fepls_grid"
    OUTPUT_DIR = SCRIPT_DIR / "data" / "simuls"
else:
    # we try current working directory
    BASE_DIR = Path("data/fepls_grid")
    OUTPUT_DIR = Path("data/simuls")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# we can filter by specific parameters if needed (set to None to use all)
FILTER_RHO = None  # e.g., [-0.5] or None for all
FILTER_GAMMA = None  # e.g., [0.5] or None for all
FILTER_D = None  # e.g., [50] or None for all
FILTER_N = None  # e.g., [100] or None for all (will collect all k values for the selected n)
FILTER_TAU = None  # e.g., [-2.0] or None for all (if None, will use best tau per kappa or all)

print(f"Scanning data directory: {BASE_DIR}")
if not BASE_DIR.exists():
    raise FileNotFoundError(f"Data directory not found at {BASE_DIR}")

# %%
###################### Helper Functions

def parse_param_from_name(name, param_type):
    """Parse parameter value from directory/file name."""
    # we first extract the numeric part, then replace 'p' with '.' and 'm' with '-'
    # this avoids replacing 'm' in parameter names like 'gamma'
    match = re.search(rf'{param_type}_([-]?[\dpm]+)', name)
    if match:
        num_str = match.group(1)
        # we replace 'p' with '.' and 'm' with '-' only in the numeric part
        num_str = num_str.replace('p', '.').replace('m', '-')
        try:
            return float(num_str)
        except ValueError:
            return None
    return None

def load_results_by_k(base_dir, filter_rho=None, filter_gamma=None, filter_d=None, filter_n=None, filter_tau=None):
    """
    Load all results and organize by k values for plotting.
    
    Returns:
        results_by_k: dict[rho][gamma][kappa][tau][d] = {
            'k': [k1, k2, ...],
            'mse': [mse1, mse2, ...],
            'variance': [var1, var2, ...],
            'bias': [bias1, bias2, ...],
            'n': n_value
        }
        all_params: dict with lists of all unique parameter values
    """
    results_by_k = {}
    all_params = {
        'rho': set(),
        'gamma': set(),
        'kappa': set(),
        'tau': set(),
        'd': set(),
        'n': set(),
        'k': set()
    }
    
    print("Scanning directory structure...")
    rho_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('rho_')]
    print(f"Found {len(rho_dirs)} rho directories")
    
    total_files = 0
    loaded_files = 0
    
    for rho_dir in rho_dirs:
        rho = parse_param_from_name(rho_dir.name, 'rho')
        if rho is None:
            continue
        if filter_rho is not None and rho not in filter_rho:
            continue
        
        all_params['rho'].add(rho)
        results_by_k[rho] = {}
        
        gamma_dirs = [d for d in rho_dir.iterdir() if d.is_dir() and d.name.startswith('gamma_')]
        for gamma_dir in gamma_dirs:
            gamma = parse_param_from_name(gamma_dir.name, 'gamma')
            if gamma is None:
                continue
            if filter_gamma is not None and gamma not in filter_gamma:
                continue
            
            all_params['gamma'].add(gamma)
            results_by_k[rho][gamma] = {}
            
            kappa_dirs = [d for d in gamma_dir.iterdir() if d.is_dir() and d.name.startswith('kappa_')]
            for kappa_dir in kappa_dirs:
                kappa = parse_param_from_name(kappa_dir.name, 'kappa')
                if kappa is None:
                    continue
                
                all_params['kappa'].add(kappa)
                results_by_k[rho][gamma][kappa] = {}
                
                tau_dirs = [d for d in kappa_dir.iterdir() if d.is_dir() and d.name.startswith('tau_')]
                for tau_dir in tau_dirs:
                    tau = parse_param_from_name(tau_dir.name, 'tau')
                    if tau is None:
                        continue
                    if filter_tau is not None and tau not in filter_tau:
                        continue
                    
                    all_params['tau'].add(tau)
                    results_by_k[rho][gamma][kappa][tau] = {}
                    
                    d_dirs = [d for d in tau_dir.iterdir() if d.is_dir() and d.name.startswith('d_')]
                    for d_dir in d_dirs:
                        d_val = parse_param_from_name(d_dir.name, 'd')
                        if d_val is None:
                            continue
                        if filter_d is not None and d_val not in filter_d:
                            continue
                        
                        all_params['d'].add(d_val)
                        
                        # we look for all result files in this directory
                        result_files = list(d_dir.glob('results_*.npz'))
                        if len(result_files) == 0:
                            continue
                        
                        total_files += len(result_files)
                        
                        # we collect data for all k values (grouped by n if filter_n is set)
                        k_data = {}  # k_data[k] = {'alignments': [...], 'errors': [...], 'n': n}
                        
                        for result_file in result_files:
                            # we parse n and k from filename
                            n_match = re.search(r'n(\d+)_', result_file.name)
                            k_match = re.search(r'_k(\d+)\.', result_file.name)
                            if not n_match or not k_match:
                                continue
                            
                            n_val = int(n_match.group(1))
                            k_val = int(k_match.group(1))
                            
                            # we filter by n if specified
                            if filter_n is not None and n_val not in filter_n:
                                continue
                            
                            all_params['n'].add(n_val)
                            all_params['k'].add(k_val)
                            
                            try:
                                # we load the data
                                data = np.load(result_file, allow_pickle=True)
                                
                                # we extract alignments and errors
                                alignments = data['alignments']  # shape: (N_MC,)
                                errors = data['errors']  # shape: (N_MC,)
                                
                                # we store data for this k (if multiple n, we use the first or specified one)
                                if k_val not in k_data or (filter_n is not None and n_val in filter_n):
                                    k_data[k_val] = {
                                        'alignments': alignments,
                                        'errors': errors,
                                        'n': n_val
                                    }
                                
                                loaded_files += 1
                                
                            except Exception as e:
                                print(f"  Warning: Could not load {result_file}: {e}")
                                continue
                        
                        # we organize data by k for this (rho, gamma, kappa, tau, d)
                        if len(k_data) > 0:
                            k_values = sorted(k_data.keys())
                            mse_list = []
                            variance_list = []
                            bias_list = []
                            
                            for k in k_values:
                                alignments = k_data[k]['alignments']
                                errors = k_data[k]['errors']
                                
                                # we compute statistics
                                # MSE = mean of squared errors
                                mse = float(np.mean(errors**2))
                                
                                # variance = variance of alignments (since alignment = <beta_hat, beta>)
                                # we use 1 - alignment as a proxy for error
                                alignment_errors = 1.0 - alignments  # error in alignment
                                variance = float(np.var(alignment_errors))
                                
                                # bias^2 = (mean error)^2
                                mean_error = float(np.mean(alignment_errors))
                                bias_sq = mean_error**2
                                
                                mse_list.append(mse)
                                variance_list.append(variance)
                                bias_list.append(bias_sq)
                            
                            if len(k_values) > 0:
                                results_by_k[rho][gamma][kappa][tau][d_val] = {
                                    'k': k_values,
                                    'mse': mse_list,
                                    'variance': variance_list,
                                    'bias': bias_list,
                                    'n': k_data[k_values[0]]['n']  # we use n from first k
                                }
    
    # we convert sets to sorted lists
    for key in all_params:
        all_params[key] = sorted(list(all_params[key]))
    
    print(f"\nLoaded {loaded_files}/{total_files} result files")
    print(f"Parameter ranges:")
    for key, values in all_params.items():
        if len(values) > 0:
            print(f"  {key}: {len(values)} values, range=[{min(values):.4f}, {max(values):.4f}]")
    
    return results_by_k, all_params

# %%
###################### Load All Results
print("\n" + "=" * 80)
print("Loading all results from directory structure")
print("=" * 80)

results_by_k, all_params = load_results_by_k(
    BASE_DIR,
    filter_rho=FILTER_RHO,
    filter_gamma=FILTER_GAMMA,
    filter_d=FILTER_D,
    filter_n=FILTER_N,
    filter_tau=FILTER_TAU
)

# we extract parameter grids for plotting
RHO_VALUES = all_params['rho']
GAMMA_VALUES = all_params['gamma']
KAPPA_VALUES = all_params['kappa']
TAU_VALUES = all_params['tau']
D_VALUES = all_params['d']
N_VALUES = all_params['n']

print(f"\nUsing parameters:")
print(f"  Rho values: {RHO_VALUES}")
print(f"  Gamma values: {GAMMA_VALUES}")
print(f"  Kappa values: {KAPPA_VALUES[:5]}... (showing first 5 of {len(KAPPA_VALUES)})")
print(f"  Tau values: {TAU_VALUES[:5]}... (showing first 5 of {len(TAU_VALUES)})")
print(f"  D values: {D_VALUES}")
print(f"  N values: {N_VALUES}")

# %%
###################### Create Plots: sim.py style (2x2 subplots)
print("\n" + "=" * 80)
print("Creating plots: sim.py style (MSE, Variance, Bias^2 vs k, and k_optimal vs kappa)")
print("=" * 80)

# we select which gamma, d, and n to plot (use first available if multiple)
plot_gamma = GAMMA_VALUES[0] if len(GAMMA_VALUES) > 0 else None
plot_d = D_VALUES[0] if len(D_VALUES) > 0 else None
plot_n = N_VALUES[0] if len(N_VALUES) > 0 else None

if plot_gamma is None or plot_d is None:
    print("No data available for plotting")
else:
    for rho_idx, rho in enumerate(RHO_VALUES):
        print(f"\nPlotting for rho {rho_idx+1}/{len(RHO_VALUES)}: rho = {rho}")
        
        # we prepare data for plotting
        if rho not in results_by_k or plot_gamma not in results_by_k[rho]:
            print(f"  No results for rho={rho}, gamma={plot_gamma}, skipping plots.")
            continue
        
        # we collect all kappa values that have data
        valid_kappas = []
        for kappa in KAPPA_VALUES:
            if (kappa in results_by_k[rho][plot_gamma] and 
                len(results_by_k[rho][plot_gamma][kappa]) > 0):
                # we check if there's data for the selected d
                has_data = False
                for tau in results_by_k[rho][plot_gamma][kappa].keys():
                    if plot_d in results_by_k[rho][plot_gamma][kappa][tau]:
                        has_data = True
                        break
                if has_data:
                    valid_kappas.append(kappa)
        
        print(f"  Found {len(valid_kappas)} valid kappas for plotting")
        
        if len(valid_kappas) == 0:
            print(f"  No valid data for rho={rho}, skipping plots.")
            continue
        
        # we choose best tau per kappa (or use filter_tau if set)
        plot_data = {}  # plot_data[kappa] = {'k': [...], 'mse': [...], 'variance': [...], 'bias': [...]}
        
        for kappa in valid_kappas:
            # we find the best tau for this kappa (lowest average MSE)
            best_tau = None
            best_avg_mse = float('inf')
            
            for tau in results_by_k[rho][plot_gamma][kappa].keys():
                if plot_d not in results_by_k[rho][plot_gamma][kappa][tau]:
                    continue
                
                data = results_by_k[rho][plot_gamma][kappa][tau][plot_d]
                if len(data['mse']) > 0:
                    avg_mse = np.mean(data['mse'])
                    if avg_mse < best_avg_mse:
                        best_avg_mse = avg_mse
                        best_tau = tau
            
            if best_tau is not None:
                plot_data[kappa] = results_by_k[rho][plot_gamma][kappa][best_tau][plot_d]
        
        if len(plot_data) == 0:
            print(f"  No data points for rho={rho}, skipping plots.")
            continue
        
        # we create the 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax_mse, ax_var, ax_bias, ax_kopt = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        
        fig.suptitle(f'Compromis Biais-Variance FEPLS (rho={rho:.2f}, gamma={plot_gamma:.2f}, d={plot_d}, n={plot_n})', fontsize=16)
        
        # we plot for each kappa
        for kappa in sorted(plot_data.keys()):
            data = plot_data[kappa]
            if len(data['k']) == 0:
                continue
            
            label = rf'$\kappa={kappa:.2f}$'
            
            # Plot 1: MSE vs k
            ax_mse.plot(data['k'], data['mse'], label=label, linewidth=2)
            
            # Plot 2: Variance vs k
            ax_var.plot(data['k'], data['variance'], linestyle='--', label=label, linewidth=2)
            
            # Plot 3: Bias^2 vs k
            ax_bias.plot(data['k'], data['bias'], linestyle=':', label=label, linewidth=2)
        
        # Plot 4: k_optimal vs kappa
        kappas_sorted = sorted(plot_data.keys())
        k_opt_list = []
        for kappa in kappas_sorted:
            data = plot_data[kappa]
            if len(data['mse']) > 0:
                idx_min = np.argmin(data['mse'])
                k_opt = data['k'][idx_min]
                k_opt_list.append(k_opt)
            else:
                k_opt_list.append(np.nan)
        
        ax_kopt.plot(kappas_sorted, k_opt_list, marker='o', color='red', markersize=8, linewidth=2)
        
        # we format axes
        ax_mse.set_title('Mean Squared Error (MSE)')
        ax_mse.set_xlabel('k (Nombre d\'extrêmes)')
        ax_mse.set_ylabel('MSE')
        ax_mse.set_yscale('log')
        ax_mse.legend()
        ax_mse.grid(True, which="both", ls="-", alpha=0.5)
        
        ax_var.set_title('Variance (Estimation Noise)')
        ax_var.set_xlabel('k')
        ax_var.set_ylabel('Variance')
        ax_var.set_yscale('log')
        ax_var.legend()
        ax_var.grid(True)
        
        ax_bias.set_title('Biais au carré (Approximation Error)')
        ax_bias.set_xlabel('k')
        ax_bias.set_ylabel('Biais²')
        ax_bias.set_yscale('log')
        ax_bias.legend()
        ax_bias.grid(True)
        
        ax_kopt.set_title(r'k optimal en fonction de $\kappa$')
        ax_kopt.set_xlabel(r'$\kappa$ (Force du signal)')
        ax_kopt.set_ylabel('k optimal')
        ax_kopt.grid(True)
        
        plt.tight_layout()
        rho_str = f"{rho:.2f}".replace('.', 'p').replace('-', 'm')
        plot_path = OUTPUT_DIR / f"bias_variance_mse_rho{rho_str}.png"
        print(f"  Saving plot to {plot_path}...")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved successfully to {plot_path}")
        plt.close()

# %%
###################### Create Plots: Separate figures for tau left/mid/right (if applicable)
print("\n" + "=" * 80)
print("Creating plots: Separate figures for tau left/mid/right")
print("=" * 80)

def classify_tau(kappa, tau, gamma, eps=1e-2):
    """Classify tau as left, mid, or right based on its position in the valid interval."""
    tau_lower = -kappa + eps
    tau_upper = (1.0 / (2.0 * gamma)) - kappa - eps
    if tau_lower >= tau_upper:
        return None
    
    tau_mid = 0.5 * (tau_lower + tau_upper)
    
    if abs(tau - tau_lower) < abs(tau - tau_mid):
        return "left"
    elif abs(tau - tau_upper) < abs(tau - tau_mid):
        return "right"
    else:
        return "mid"

if plot_gamma is None or plot_d is None:
    print("No data available for plotting")
else:
    tau_labels_order = ["left", "mid", "right"]
    tau_titles = {
        "left": r"Tau à la frontière gauche ($\tau \approx -\kappa$)",
        "mid": "Tau au milieu de l'intervalle",
        "right": r"Tau à la frontière droite ($\tau \approx 1/(2\gamma)-\kappa$)",
    }
    
    for rho_idx, rho in enumerate(RHO_VALUES):
        if rho not in results_by_k or plot_gamma not in results_by_k[rho]:
            continue
        
        for tau_label in tau_labels_order:
            # we collect data for this tau_label
            plot_data_tau = {}  # plot_data_tau[kappa] = {'k': [...], 'mse': [...], ...}
            
            for kappa in KAPPA_VALUES:
                if kappa not in results_by_k[rho][plot_gamma]:
                    continue
                
                # we find tau values classified as tau_label
                for tau in results_by_k[rho][plot_gamma][kappa].keys():
                    if plot_d not in results_by_k[rho][plot_gamma][kappa][tau]:
                        continue
                    
                    classified = classify_tau(kappa, tau, plot_gamma)
                    if classified == tau_label:
                        plot_data_tau[kappa] = results_by_k[rho][plot_gamma][kappa][tau][plot_d]
                        break  # we take the first matching tau
            
            if len(plot_data_tau) == 0:
                print(f"  No data for rho={rho}, tau_label={tau_label}, skipping.")
                continue
            
            # we create the 2x2 subplot figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            ax_mse, ax_var, ax_bias, ax_kopt = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
            
            fig.suptitle(
                f"{tau_titles[tau_label]}  (rho={rho:.2f}, gamma={plot_gamma:.2f}, d={plot_d})",
                fontsize=16
            )
            
            for kappa in sorted(plot_data_tau.keys()):
                data = plot_data_tau[kappa]
                if len(data['k']) == 0:
                    continue
                
                # we find the actual tau value used
                actual_tau = None
                for tau in results_by_k[rho][plot_gamma][kappa].keys():
                    if plot_d in results_by_k[rho][plot_gamma][kappa][tau]:
                        if classify_tau(kappa, tau, plot_gamma) == tau_label:
                            actual_tau = tau
                            break
                
                if actual_tau is None:
                    continue
                
                label = rf'$\kappa={kappa:.2f}, \tau={actual_tau:.2f}$'
                
                # MSE vs k
                ax_mse.plot(data['k'], data['mse'], label=label, linewidth=2)
                
                # variance vs k
                ax_var.plot(data['k'], data['variance'], linestyle="--", label=label, linewidth=2)
                
                # biais^2 vs k
                ax_bias.plot(data['k'], data['bias'], linestyle=":", label=label, linewidth=2)
                
                # k_opt for this (kappa, tau_label)
                if len(data['mse']) > 0:
                    idx_min = int(np.argmin(data['mse']))
                    k_opt = data['k'][idx_min]
                    ax_kopt.scatter(kappa, k_opt, label=label, s=100)
            
            # we format axes
            ax_mse.set_title("Mean Squared Error (MSE)")
            ax_mse.set_xlabel("k (nombre d'extrêmes)")
            ax_mse.set_ylabel("MSE")
            ax_mse.set_yscale("log")
            ax_mse.grid(True, which="both", ls="-", alpha=0.5)
            ax_mse.legend()
            
            ax_var.set_title("Variance (estimation noise)")
            ax_var.set_xlabel("k")
            ax_var.set_yscale("log")
            ax_var.grid(True)
            ax_var.legend()
            
            ax_bias.set_title("Biais au carré (approximation error)")
            ax_bias.set_xlabel("k")
            ax_bias.set_yscale("log")
            ax_bias.grid(True)
            ax_bias.legend()
            
            ax_kopt.set_title(rf"k optimal en fonction de $\kappa$ "
                            f"(tau_label = {tau_label})")
            ax_kopt.set_xlabel(r"$\kappa$")
            ax_kopt.set_ylabel("k optimal")
            ax_kopt.grid(True)
            ax_kopt.legend()
            
            plt.tight_layout()
            rho_str = f"{rho:.2f}".replace('.', 'p').replace('-', 'm')
            plot_path = OUTPUT_DIR / f"bias_variance_mse_rho{rho_str}_tau{tau_label}.png"
            print(f"  Saving plot to {plot_path}...")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Plot saved successfully to {plot_path}")
            plt.close()

print("\n" + "=" * 80)
print("=" * 80)
print("ALL PLOTS COMPLETED!")
print("=" * 80)
print("=" * 80)
print(f"Summary:")
print(f"  - Processed {len(RHO_VALUES)} rho values: {RHO_VALUES}")
print(f"  - Processed {len(KAPPA_VALUES)} kappa values")
print(f"  - Processed {len(TAU_VALUES)} tau values")
print(f"  - Plots saved to: {OUTPUT_DIR}")
print("=" * 80)
