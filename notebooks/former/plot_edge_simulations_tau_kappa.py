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
# # Edge-of-Inequality Simulations: Plotting Results
# 
# This script loads pre-computed simulation results from the hierarchical directory structure
# and generates plots for optimal k and variance analysis.
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
###################### Configuration
BASE_DIR = Path("data/fepls_grid")
OUTPUT_DIR = Path("data/simuls")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# we can filter by specific parameters if needed (set to None to use all)
FILTER_RHO = None  # e.g., [-0.5] or None for all
FILTER_GAMMA = None  # e.g., [0.5] or None for all
FILTER_D = None  # e.g., [50] or None for all
FILTER_N = None  # e.g., [100] or None for all (will use the first available if multiple exist)

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

def load_all_results(base_dir, filter_rho=None, filter_gamma=None, filter_d=None, filter_n=None):
    """
    Load all results from the hierarchical directory structure.
    
    Returns:
        results: dict[rho][gamma][kappa][tau][d] = {
            'optimal_k': int,
            'variance': float,
            'mean_alignment': float,
            'n': int,
            'k_n': int
        }
        all_params: dict with lists of all unique parameter values
    """
    results = {}
    all_params = {
        'rho': set(),
        'gamma': set(),
        'kappa': set(),
        'tau': set(),
        'd': set(),
        'n': set()
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
        results[rho] = {}
        
        gamma_dirs = [d for d in rho_dir.iterdir() if d.is_dir() and d.name.startswith('gamma_')]
        for gamma_dir in gamma_dirs:
            gamma = parse_param_from_name(gamma_dir.name, 'gamma')
            if gamma is None:
                continue
            if filter_gamma is not None and gamma not in filter_gamma:
                continue
            
            all_params['gamma'].add(gamma)
            results[rho][gamma] = {}
            
            kappa_dirs = [d for d in gamma_dir.iterdir() if d.is_dir() and d.name.startswith('kappa_')]
            for kappa_dir in kappa_dirs:
                kappa = parse_param_from_name(kappa_dir.name, 'kappa')
                if kappa is None:
                    continue
                
                all_params['kappa'].add(kappa)  # we add kappa to all_params
                results[rho][gamma][kappa] = {}
                
                tau_dirs = [d for d in kappa_dir.iterdir() if d.is_dir() and d.name.startswith('tau_')]
                for tau_dir in tau_dirs:
                    tau = parse_param_from_name(tau_dir.name, 'tau')
                    if tau is None:
                        continue
                    
                    all_params['tau'].add(tau)  # we add tau to all_params
                    results[rho][gamma][kappa][tau] = {}
                    
                    d_dirs = [d for d in tau_dir.iterdir() if d.is_dir() and d.name.startswith('d_')]
                    for d_dir in d_dirs:
                        d_val = parse_param_from_name(d_dir.name, 'd')
                        if d_val is None:
                            continue
                        if filter_d is not None and d_val not in filter_d:
                            continue
                        
                        all_params['d'].add(d_val)
                        
                        # we look for result files
                        result_files = list(d_dir.glob('results_*.npz'))
                        if len(result_files) == 0:
                            continue
                        
                        total_files += len(result_files)
                        
                        # we select which file to use based on filter_n
                        selected_file = None
                        if filter_n is not None:
                            # we try to find a file with the specified n
                            for f in result_files:
                                n_match = re.search(r'n(\d+)_', f.name)
                                if n_match and int(n_match.group(1)) in filter_n:
                                    selected_file = f
                                    break
                        else:
                            # we use the first file (or could use largest n, etc.)
                            selected_file = result_files[0]
                        
                        if selected_file is None:
                            continue
                        
                        # we parse n and k from filename
                        n_match = re.search(r'n(\d+)_', selected_file.name)
                        k_match = re.search(r'_k(\d+)\.', selected_file.name)
                        if n_match and k_match:
                            n_val = int(n_match.group(1))
                            k_val = int(k_match.group(1))
                            all_params['n'].add(n_val)
                            
                            try:
                                # we load the data
                                data = np.load(selected_file, allow_pickle=True)
                                
                                # we extract results
                                alignments = data['alignments']  # shape: (N_MC,)
                                errors = data['errors']  # shape: (N_MC,)
                                
                                # we compute statistics
                                variance = float(np.var(alignments))
                                mean_alignment = float(np.mean(alignments))
                                
                                results[rho][gamma][kappa][tau][d_val] = {
                                    'optimal_k': k_val,  # this is k_n from the filename
                                    'variance': variance,
                                    'mean_alignment': mean_alignment,
                                    'n': n_val,
                                    'k_n': k_val
                                }
                                
                                loaded_files += 1
                                
                            except Exception as e:
                                print(f"  Warning: Could not load {selected_file}: {e}")
                                continue
    
    # we convert sets to sorted lists
    for key in all_params:
        all_params[key] = sorted(list(all_params[key]))
    
    print(f"\nLoaded {loaded_files}/{total_files} result files")
    print(f"Parameter ranges:")
    for key, values in all_params.items():
        if len(values) > 0:
            print(f"  {key}: {len(values)} values, range=[{min(values):.4f}, {max(values):.4f}]")
    
    return results, all_params

# %%
###################### Load All Results
print("\n" + "=" * 80)
print("Loading all results from directory structure")
print("=" * 80)

results, all_params = load_all_results(
    BASE_DIR,
    filter_rho=FILTER_RHO,
    filter_gamma=FILTER_GAMMA,
    filter_d=FILTER_D,
    filter_n=FILTER_N
)

# we extract parameter grids for plotting
RHO_VALUES = all_params['rho']
GAMMA_VALUES = all_params['gamma']
KAPPA_VALUES = all_params['kappa']
TAU_VALUES = all_params['tau']
D_VALUES = all_params['d']

print(f"\nUsing parameters:")
print(f"  Rho values: {RHO_VALUES}")
print(f"  Gamma values: {GAMMA_VALUES}")
print(f"  Kappa values: {KAPPA_VALUES[:5]}... (showing first 5 of {len(KAPPA_VALUES)})")
print(f"  Tau values: {TAU_VALUES[:5]}... (showing first 5 of {len(TAU_VALUES)})")
print(f"  D values: {D_VALUES}")

# %%
###################### Create Plots: Optimal k vs tau for different kappa
print("\n" + "=" * 80)
print("STEP 4: Creating plots")
print("=" * 80)

# we select which gamma and d to plot (use first available if multiple)
plot_gamma = GAMMA_VALUES[0] if len(GAMMA_VALUES) > 0 else None
plot_d = D_VALUES[0] if len(D_VALUES) > 0 else None

if plot_gamma is None or plot_d is None:
    print("No data available for plotting")
else:
    for rho_idx, rho in enumerate(RHO_VALUES):
        print(f"\n[STEP 4] Plotting for rho {rho_idx+1}/{len(RHO_VALUES)}: rho = {rho}")
        
        # we prepare data for plotting
        if rho not in results or plot_gamma not in results[rho]:
            print(f"  [STEP 4] No results for rho={rho}, skipping plots.")
            continue
        
        valid_kappas = [k for k in KAPPA_VALUES 
                       if k in results[rho][plot_gamma] 
                       and len(results[rho][plot_gamma][k]) > 0]
        print(f"  [STEP 4] Found {len(valid_kappas)} valid kappas for plotting")
        
        if len(valid_kappas) == 0:
            print(f"  [STEP 4] No valid data for rho={rho}, skipping plots.")
            continue
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # we plot optimal k vs tau for different kappa
        ax1 = axes[0]
        for kappa in valid_kappas[:5]:  # we plot first 5 kappa values to avoid clutter
            tau_list = sorted([tau for tau in results[rho][plot_gamma][kappa].keys() 
                              if plot_d in results[rho][plot_gamma][kappa][tau]])
            optimal_k_list = [results[rho][plot_gamma][kappa][tau][plot_d]['optimal_k'] 
                             for tau in tau_list]
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
            tau_list = sorted([tau for tau in results[rho][plot_gamma][kappa].keys() 
                              if plot_d in results[rho][plot_gamma][kappa][tau]])
            variance_list = [results[rho][plot_gamma][kappa][tau][plot_d]['variance'] 
                            for tau in tau_list]
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
        plot_path = OUTPUT_DIR / f"plots_rho{rho_str}.png"
        print(f"  [STEP 4] Saving plot to {plot_path}...")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  [STEP 4] Plot saved successfully to {plot_path}")
        plt.close()

# %%
###################### Create 2D Heatmaps: Optimal k and Variance as function of (tau, kappa)
for rho_idx, rho in enumerate(RHO_VALUES):
    print(f"\n[STEP 4] Creating heatmaps for rho {rho_idx+1}/{len(RHO_VALUES)}: rho = {rho}")
    
    if rho not in results or plot_gamma not in results[rho] or len(results[rho][plot_gamma]) == 0:
        print(f"  [STEP 4] No valid data for rho={rho}, skipping heatmaps.")
        continue
    
    print(f"  [STEP 4] Preparing heatmap data...")
    
    # we collect all tau values for this rho
    all_taus_rho = set()
    for kappa in results[rho][plot_gamma].keys():
        for tau in results[rho][plot_gamma][kappa].keys():
            if plot_d in results[rho][plot_gamma][kappa][tau]:
                all_taus_rho.add(tau)
    all_taus_rho = sorted(all_taus_rho)
    
    # we create matrices for heatmaps
    optimal_k_matrix = np.full((len(all_taus_rho), len(KAPPA_VALUES)), np.nan)
    variance_matrix = np.full((len(all_taus_rho), len(KAPPA_VALUES)), np.nan)
    
    for i, tau in enumerate(all_taus_rho):
        for j, kappa in enumerate(KAPPA_VALUES):
            if kappa in results[rho][plot_gamma] and tau in results[rho][plot_gamma][kappa]:
                if plot_d in results[rho][plot_gamma][kappa][tau]:
                    optimal_k_matrix[i, j] = results[rho][plot_gamma][kappa][tau][plot_d]['optimal_k']
                    variance_matrix[i, j] = results[rho][plot_gamma][kappa][tau][plot_d]['variance']
    
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
    heatmap_path = OUTPUT_DIR / f"heatmaps_rho{rho_str}.png"
    print(f"  [STEP 4] Saving heatmap to {heatmap_path}...")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"  [STEP 4] Heatmap saved successfully to {heatmap_path}")
    plt.close()

# %%
###################### Summary Statistics
print("\n" + "=" * 80)
print("STEP 5: Summary Statistics")
print("=" * 80)

if plot_gamma is None or plot_d is None:
    print("No data available for summary")
else:
    for rho_idx, rho in enumerate(RHO_VALUES):
        if rho not in results or plot_gamma not in results[rho] or len(results[rho][plot_gamma]) == 0:
            print(f"\n[STEP 5] No results for rho={rho}, skipping summary.")
            continue
        print(f"\n{'='*80}")
        print(f"[STEP 5] Summary for rho {rho_idx+1}/{len(RHO_VALUES)}: rho = {rho}")
        print(f"{'='*80}")
        
        total_configs = 0
        for kappa in results[rho][plot_gamma].keys():
            total_configs += len([tau for tau in results[rho][plot_gamma][kappa].keys() 
                                 if plot_d in results[rho][plot_gamma][kappa][tau]])
        print(f"[STEP 5] Total valid (tau, kappa) configurations: {total_configs}")
        print(f"[STEP 5] Number of unique kappas: {len(results[rho][plot_gamma].keys())}")
        
        # we compute average optimal k and variance across all configurations
        print(f"[STEP 5] Computing summary statistics...")
        all_optimal_k = []
        all_variance = []
        all_mean_alignment = []
        for kappa in results[rho][plot_gamma].keys():
            for tau in results[rho][plot_gamma][kappa].keys():
                if plot_d in results[rho][plot_gamma][kappa][tau]:
                    all_optimal_k.append(results[rho][plot_gamma][kappa][tau][plot_d]['optimal_k'])
                    all_variance.append(results[rho][plot_gamma][kappa][tau][plot_d]['variance'])
                    all_mean_alignment.append(results[rho][plot_gamma][kappa][tau][plot_d]['mean_alignment'])
        
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
print(f"  - Plots saved to: {OUTPUT_DIR}")
print("=" * 80)
