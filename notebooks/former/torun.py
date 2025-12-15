import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt  # not strictly needed but often useful
from pathlib import Path
from tqdm import tqdm  # we use tqdm for progress bars

# ======================= CONFIGURATION GLOBALE =======================

Q = 2.1  # we set q > 2 for theory  # we use q slightly above 2
GAMMA_VALUES = np.array([0.5])  # we set some tail indices gamma in (0,1)
RHO_VALUES = np.linspace(-0.5, -5.0, 6)  # we set |rho| in [0.5, 5], used as second-order magnitude
D_VALUES = np.array([50])  # we set d in {5,10,...,50}
N_VALUES = np.unique(np.round(np.logspace(2.0, 4, 6)).astype(int))  # we set n on log scale between 10^2 and ~10000
N_MC = 500  # we set monte carlo replications (increase on cluster)
TRUE_BETA_TYPE = "first_coords"  # we choose how to define true beta
OUTPUT_ROOT = Path("data/fepls_grid")  # we set root directory to save all results
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ======================= OUTIL: VECTEUR BETA VRAI =======================

def make_true_beta(d: int, mode: str = "first_coords") -> np.ndarray:
    if mode == "first_coords":
        beta = np.zeros(d)  # we create zero vector
        m = min(5, d)  # we choose number of nonzeros
        beta[:m] = 1.0  # we set first coordinates to 1
        beta = beta / np.linalg.norm(beta)  # we normalize
        return beta  # we return beta
    else:
        beta = np.random.normal(size=d)  # we sample random beta
        beta = beta / np.linalg.norm(beta)  # we normalize
        return beta  # we return beta

# ======================= OUTIL: GRILLE BIAISÉE SUR LES BORNES =======================

def boundary_biased_grid(lower: float, upper: float, n_points: int, sharpness: float = 3.0) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n_points)  # we build uniform grid in [0,1]
    t = 0.5 * (np.tanh(sharpness * (s - 0.5)) + 1.0)  # we squash to concentrate near 0 and 1
    grid = lower + (upper - lower) * t  # we map to [lower, upper]
    return grid  # we return boundary-biased grid

# ======================= OUTIL: CONDITIONS FEPLS =======================

def admissible_kappa_tau(gamma: float, q: float, kappa: float, tau: float) -> bool:
    cond1 = (q * kappa * gamma > 1.0)  # we check q*kappa*gamma>1
    cond2 = (0.0 < 2.0 * (kappa + tau) * gamma < 1.0)  # we check 0<2(kappa+tau)gamma<1
    return cond1 and cond2  # we return boolean

def kappa_bounds(gamma: float, q: float, eps: float = 1e-3) -> tuple[float, float]:
    kappa_min = 1.0 / (q * gamma) + eps  # we set lower bound from q*kappa*gamma>1
    kappa_max = 3.0  # we choose an arbitrary upper bound for kappa
    return kappa_min, kappa_max  # we return interval

def tau_bounds(gamma: float, kappa: float, eps: float = 1e-3) -> tuple[float, float] | None:
    tau_left = -kappa + eps  # we set left boundary slightly inside
    tau_right = (1.0 / (2.0 * gamma)) - kappa - eps  # we set right boundary slightly inside
    if tau_left >= tau_right:  # we check emptiness
        return None  # we return None if no interval
    return tau_left, tau_right  # we return interval

# ======================= MODÈLE: GÉNÉRATION DE DONNÉES =======================

def generate_data(n: int, d: int, kappa: float, gamma: float, q: float, beta_true: np.ndarray, random_state: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    U = random_state.uniform(0.0, 1.0, size=n)  # we generate uniforms
    Y = U ** (-gamma)  # we generate heavy-tailed response
    Noise = random_state.standard_t(df=q + 0.1, size=(n, d))  # we generate heavy-tailed noise
    signal_strength = Y[:, np.newaxis] ** kappa  # we build g(Y)
    X = signal_strength * beta_true[np.newaxis, :] + Noise  # we build X
    return X, Y  # we return covariates and response

# ======================= ESTIMATEUR FEPLS SCALAIRE =======================

def fepls_estimator(X: np.ndarray, Y: np.ndarray, k: int, tau: float) -> np.ndarray:
    n, d = X.shape  # we get shapes
    if k >= n:
        k = n - 1  # we ensure k<n
    y_sorted = np.sort(Y)  # we sort Y
    threshold = y_sorted[n - k]  # we set threshold
    idx = (Y >= threshold)  # we select extremes
    if np.sum(idx) == 0:
        return np.zeros(d, dtype=float)  # we return zero vector if no extremes
    weights = (Y[idx] ** tau)  # we compute weights
    X_extreme = X[idx]  # we select extreme X
    weighted_sum = np.sum(X_extreme * weights[:, np.newaxis], axis=0)  # we compute weighted sum
    norm = np.linalg.norm(weighted_sum)  # we compute norm
    if norm == 0.0:
        return np.zeros(d, dtype=float)  # we return zero vector if degenerate
    return weighted_sum / norm  # we return normalized estimator

# ======================= CHOIX DE k_n EN FONCTION DE n ET rho =======================

def choose_k_n(n: int, rho_second: float, c_k: float = 5.0, k_min: int = 5) -> int:
    rho = -abs(rho_second)  # we interpret rho_second as |rho| and set rho<0
    alpha_k = -2.0 * rho / (1.0 - 2.0 * rho)  # we compute exponent alpha_k
    k_real = c_k * (n ** alpha_k)  # we compute real-valued k_n
    k_int = int(max(k_min, min(k_real, n // 5)))  # we clip to [k_min, n/5]
    return k_int  # we return integer k_n

# ======================= BOUCLE PRINCIPALE SUR LES PARAMÈTRES =======================

def run_full_grid():
    # we iterate over all parameter combinations with progress bars
    for rho_second in tqdm(RHO_VALUES, desc="rho", position=0):
        rho_dir = OUTPUT_ROOT / f"rho_{rho_second:.2f}".replace(".", "p")  # we build rho directory
        rho_dir.mkdir(parents=True, exist_ok=True)  # we create directory

        for gamma in tqdm(GAMMA_VALUES, desc="gamma", position=1, leave=False):
            kappa_min, kappa_max = kappa_bounds(gamma, Q, eps=1e-3)  # we compute kappa bounds
            kappa_grid = boundary_biased_grid(kappa_min, kappa_max, n_points=10, sharpness=3.0)  # we build kappa grid

            gamma_dir = rho_dir / f"gamma_{gamma:.2f}".replace(".", "p")  # we build gamma directory
            gamma_dir.mkdir(parents=True, exist_ok=True)  # we create directory

            for kappa in tqdm(kappa_grid, desc=f"kappa (gamma={gamma:.2f})", position=2, leave=False):
                tau_int = tau_bounds(gamma, kappa, eps=1e-3)  # we compute tau interval
                if tau_int is None:
                    continue  # we skip invalid kappa
                tau_left, tau_right = tau_int  # we unpack interval
                tau_grid = boundary_biased_grid(tau_left, tau_right, n_points=10, sharpness=3.0)  # we build tau grid

                # we filter admissible tau
                tau_grid = np.array(
                    [t for t in tau_grid if admissible_kappa_tau(gamma, Q, kappa, t)],
                    dtype=float,
                )  # we keep only admissible tau
                if tau_grid.size == 0:
                    continue  # we skip if no admissible tau

                kappa_dir = gamma_dir / f"kappa_{kappa:.3f}".replace(".", "p")  # we build kappa directory
                kappa_dir.mkdir(parents=True, exist_ok=True)  # we create directory

                for tau in tqdm(tau_grid, desc=f"tau (kappa={kappa:.3f})", position=3, leave=False):
                    tau_dir = kappa_dir / f"tau_{tau:.3f}".replace(".", "p")  # we build tau directory
                    tau_dir.mkdir(parents=True, exist_ok=True)  # we create directory

                    for d in D_VALUES:
                        d_dir = tau_dir / f"d_{d:03d}"  # we build d directory
                        d_dir.mkdir(parents=True, exist_ok=True)  # we create directory

                        beta_true = make_true_beta(d, mode=TRUE_BETA_TYPE)  # we build true beta
                        config_meta = {
                            "rho_second": float(rho_second),
                            "gamma": float(gamma),
                            "kappa": float(kappa),
                            "tau": float(tau),
                            "q": float(Q),
                            "d": int(d),
                        }  # we record metadata

                        for n in tqdm(N_VALUES, desc=f"n (d={d})", position=4, leave=False):
                            k_n = choose_k_n(int(n), rho_second, c_k=5.0, k_min=5)  # we choose k_n
                            file_name = f"results_n{int(n)}_k{int(k_n)}.npz"  # we build file name
                            out_path = d_dir / file_name  # we build full path
                            if out_path.exists():
                                continue  # we skip if file already exists

                            errors = np.zeros(N_MC, dtype=float)  # we allocate errors
                            alignments = np.zeros(N_MC, dtype=float)  # we allocate alignments
                            betas = np.zeros((N_MC, d), dtype=float)  # we allocate betas

                            rs = npr.RandomState(seed=12345)  # we set random state (you can randomize per config)

                            for m in range(N_MC):
                                X, Y = generate_data(int(n), int(d), float(kappa), float(gamma), float(Q), beta_true, rs)  # we generate data
                                beta_hat = fepls_estimator(X, Y, k_n, float(tau))  # we compute estimator
                                betas[m, :] = beta_hat  # we store estimator
                                errors[m] = np.linalg.norm(beta_hat - beta_true)  # we compute error
                                alignments[m] = float(np.dot(beta_hat, beta_true))  # we compute alignment

                            np.savez_compressed(
                                out_path,
                                betas=betas,
                                errors=errors,
                                alignments=alignments,
                                n=int(n),
                                k_n=int(k_n),
                                **config_meta,
                            )  # we save all tensors and metadata

if __name__ == "__main__":
    run_full_grid()  # we launch full grid computation