# FEPLS Multi-Dimensional Extension

Ce module étend FEPLS (Functional Extreme Partial Least Squares) pour supporter des données fonctionnelles multi-dimensionnelles (2D et 3D) en plus du cas 1D classique.

## Structure

Le module `fepls_nd.py` fournit des fonctions généralisées qui détectent automatiquement la dimensionnalité des données et appliquent le calcul approprié.

## Fonctions principales

### `fepls_nd(X, Y, y_matrix, tau)`

Fonction principale qui calcule la direction FEPLS pour des données multi-dimensionnelles.

**Paramètres:**
- `X`: Tableau numpy des covariables fonctionnelles
  - **1D**: shape `(N, n, d)` où N=batch, n=échantillons, d=dimension
  - **2D**: shape `(N, n, d1, d2)` pour données 2D (images, spectrogrammes, etc.)
  - **3D**: shape `(N, n, d1, d2, d3)` pour données 3D (volumes, données spatio-temporelles, etc.)
- `Y`: Réponses scalaires, shape `(N, n)`
- `y_matrix`: Matrice de seuil, shape `(N, n)`
- `tau`: Paramètre(s) de la fonction test (index de variation régulière)
  - **1D**: un seul `float`
  - **2D**: `tuple/list` de 2 floats `(tau1, tau2)` ou un seul `float` (utilisé pour les deux)
  - **3D**: `tuple/list` de 3 floats `(tau1, tau2, tau3)` ou un seul `float` (utilisé pour tous)
- `position_dependent`: `bool`, optionnel (défaut: `False`)
  - Si `True`, tau varie avec la position spatiale (seulement pour 2D/3D)
  - Si `False`, utilise la moyenne des valeurs tau

**Retourne:**
- `beta_hat`: Direction FEPLS estimée
  - **1D**: shape `(N, d)`
  - **2D**: shape `(N, d1, d2)`
  - **3D**: shape `(N, d1, d2, d3)`

### `projection_nd(X, beta)`

Calcule la projection `<X, beta>` pour des données multi-dimensionnelles.

**Paramètres:**
- `X`: Covariables fonctionnelles
  - **1D**: shape `(n, d)`
  - **2D**: shape `(n, d1, d2)`
  - **3D**: shape `(n, d1, d2, d3)`
- `beta`: Direction FEPLS
  - **1D**: shape `(d,)`
  - **2D**: shape `(d1, d2)`
  - **3D**: shape `(d1, d2, d3)`

**Retourne:**
- `proj`: Projections, shape `(n,)`

## Exemples d'utilisation

### Cas 1D (compatibilité arrière)

```python
import numpy as np
from utils.fepls_nd import fepls_nd

# données 1D: (N, n, d)
X = np.random.randn(1, 100, 50)  # 1 batch, 100 échantillons, dimension 50
Y = np.random.gamma(2, 2, (1, 100))  # réponses heavy-tailed
y_matrix = np.percentile(Y, 80) * np.ones_like(Y)  # seuil au 80ème percentile
tau = -1.0

beta_hat = fepls_nd(X, Y, y_matrix, tau)
# beta_hat shape: (1, 50)
```

### Cas 2D (images, spectrogrammes)

```python
# données 2D: (N, n, d1, d2)
X = np.random.randn(1, 100, 20, 20)  # 1 batch, 100 échantillons, images 20x20
Y = np.random.gamma(2, 2, (1, 100))
y_matrix = np.percentile(Y, 80) * np.ones_like(Y)

# option 1: deux tau différents (tau1 pour d1, tau2 pour d2)
tau1, tau2 = -1.0, -0.5
beta_hat = fepls_nd(X, Y, y_matrix, (tau1, tau2))
# beta_hat shape: (1, 20, 20) - direction 2D

# option 2: un seul tau (utilisé pour les deux dimensions)
beta_hat = fepls_nd(X, Y, y_matrix, -1.0)

# option 3: tau dépendant de la position spatiale
beta_hat = fepls_nd(X, Y, y_matrix, (tau1, tau2), position_dependent=True)
```

### Cas 3D (volumes, données spatio-temporelles)

```python
# données 3D: (N, n, d1, d2, d3)
X = np.random.randn(1, 100, 10, 10, 10)  # 1 batch, 100 échantillons, volumes 10x10x10
Y = np.random.gamma(2, 2, (1, 100))
y_matrix = np.percentile(Y, 80) * np.ones_like(Y)

# option 1: trois tau différents (tau1 pour d1, tau2 pour d2, tau3 pour d3)
tau1, tau2, tau3 = -1.0, -0.5, -0.8
beta_hat = fepls_nd(X, Y, y_matrix, (tau1, tau2, tau3))
# beta_hat shape: (1, 10, 10, 10) - direction 3D

# option 2: deux tau (le troisième est la moyenne des deux premiers)
beta_hat = fepls_nd(X, Y, y_matrix, (tau1, tau2))

# option 3: un seul tau (utilisé pour toutes les dimensions)
beta_hat = fepls_nd(X, Y, y_matrix, -1.0)

# option 4: tau dépendant de la position spatiale
beta_hat = fepls_nd(X, Y, y_matrix, (tau1, tau2, tau3), position_dependent=True)
```

## Calculs internes

### Produit scalaire multi-dimensionnel

- **1D**: Produit scalaire standard `dot(x, y)`
- **2D**: Produit de Frobenius `sum(x * y)`
- **3D**: Somme sur toutes les dimensions `sum(x * y)`

### Norme multi-dimensionnelle

- **1D**: Norme euclidienne standard
- **2D**: Norme de Frobenius `sqrt(sum(x^2))`
- **3D**: Norme généralisée `sqrt(sum(x^2))`

## Optimisations

Toutes les fonctions sont compilées avec Numba pour des performances optimales:
- `@numba.njit(parallel=True)` pour parallélisation automatique
- Calculs vectorisés pour efficacité maximale

## Compatibilité

Le module est rétro-compatible avec le code existant:
- La fonction `fepls()` est un wrapper vers `fepls_nd()`
- Les données 1D fonctionnent exactement comme avant

## Exemple complet

Voir `notebooks/new/fepls_2d_3d_example.py` pour un exemple complet avec visualisations.

## Notes

- Les données doivent être heavy-tailed pour que FEPLS soit approprié
- Les paramètres `tau` doivent être choisis pour satisfaire les conditions théoriques
- La normalisation est automatique (beta_hat a toujours norme 1)
- **Mode position_dependent**: Quand `position_dependent=True`, les tau sont interpolés linéairement selon la position dans chaque dimension. Par exemple, pour 2D avec `tau1=-1.0` et `tau2=-0.5`, au point `(j1, j2)`, le tau effectif est `tau1 * (j1/d1) + tau2 * (j2/d2)`. Cela permet d'avoir un comportement spatialement variable.
- **Mode par défaut**: Quand `position_dependent=False`, le tau effectif est simplement la moyenne des tau fournis: `(tau1 + tau2)/2` pour 2D, `(tau1 + tau2 + tau3)/3` pour 3D.

