# Explication de la Modélisation du Quantile Conditionnel

## Objectif

On veut estimer le **quantile conditionnel** de Y sachant la projection de X sur β :

\[
Q_\alpha(Y \mid \langle X, \beta \rangle = s)
\]

où :
- \( \alpha = 0.95 \) (quantile à 95%)
- \( \langle X, \beta \rangle = \frac{1}{d} \sum_{j=1}^d X_j \beta_j \) est la projection de X sur β
- \( s \) est une valeur dans la grille de projections

## Méthode : Estimation par Noyau (Kernel Smoothing)

On utilise deux approches différentes pour calculer les poids :

### 1. **Méthode Univariée** (ligne pointillée verte)

Pour chaque point \( s \) dans la grille :

1. **Calcul des projections** :
   \[
   p_i = \langle X_i, \beta \rangle / d = \frac{1}{d} \sum_{j=1}^d X_{i,j} \beta_j
   \]

2. **Calcul des poids** (Nadaraya-Watson avec noyau gaussien) :
   
   Dans le code, on calcule :
   ```python
   K_h = Gaussian_kernel((p_i - s * <beta, beta>/d) / h)
   ```
   
   où :
   - \( p_i = \langle X_i, \beta \rangle / d \) est la projection de X_i
   - \( s \cdot \langle \beta, \beta \rangle / d = s \cdot \|\beta\|^2 / d \)
   
   **Note** : Si β est normalisé (\( \|\beta\| = 1 \)), alors \( \langle \beta, \beta \rangle / d = 1/d \), donc on compare \( p_i \) avec \( s/d \).
   
   Les poids normalisés sont :
   \[
   w_i^{univ}(s) = \frac{K_h((p_i - s \cdot \|\beta\|^2 / d) / h)}{\sum_{j=1}^n K_h((p_j - s \cdot \|\beta\|^2 / d) / h)}
   \]
   
   où \( K_h(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}) \) est le noyau gaussien standardisé.

3. **Quantile pondéré** :
   \[
   Q_\alpha^{univ}(s) = \text{quantile pondéré de } \{Y_i\} \text{ avec poids } \{w_i^{univ}(s)\}
   \]

### 2. **Méthode Fonctionnelle** (ligne orange)

Pour chaque point \( s \) dans la grille :

1. **Calcul des poids** (Nadaraya-Watson fonctionnel) :
   \[
   w_i^{func}(s) = \frac{K_h(\|X_i - s \cdot \beta\|)}{\sum_{j=1}^n K_h(\|X_j - s \cdot \beta\|)}
   \]
   
   où \( \|X_i - s \cdot \beta\| = \sqrt{\frac{1}{d} \sum_{j=1}^d (X_{i,j} - s \cdot \beta_j)^2} \) est la distance L² fonctionnelle.

   **Intuition** : On compare la fonction X_i avec la fonction \( s \cdot \beta \) (qui est une version mise à l'échelle de β).

2. **Quantile pondéré** :
   \[
   Q_\alpha^{func}(s) = \text{quantile pondéré de } \{Y_i\} \text{ avec poids } \{w_i^{func}(s)\}
   \]

## Calcul du Quantile Pondéré

Pour calculer le quantile pondéré :

1. **Trier** les données Y par ordre croissant
2. **Calculer les quantiles cumulatifs pondérés** :
   \[
   q_i = \frac{\sum_{j=1}^i w_j - 0.5 \cdot w_i}{\sum_{j=1}^n w_j}
   \]
3. **Interpoler** pour trouver la valeur correspondant à α :
   \[
   Q_\alpha = \text{interpolation linéaire de } (q_i, Y_i) \text{ à } \alpha
   \]

## Différence entre les Deux Méthodes

- **Univariée** : Utilise seulement la projection scalaire \( \langle X, \beta \rangle \). Plus rapide, mais perd l'information sur la forme de X.

- **Fonctionnelle** : Utilise la distance complète entre les fonctions X et \( s \cdot \beta \). Plus riche en information, mais plus coûteuse.

## Problèmes Potentiels

1. **Bande passante (h)** : Si h est trop petit → sur-ajustement. Si h est trop grand → sous-ajustement.
   - On utilise la règle de Silverman : \( h = 1.06 \cdot \sigma \cdot n^{-1/5} \)

2. **Peu de données** : Si pour un point s donné, très peu de X_i sont proches, les poids peuvent être instables.

3. **Extrapolation** : Les points s en dehors de la plage des projections observées sont moins fiables.

## Interprétation

La courbe montre comment le quantile à 95% de Y varie en fonction de la projection \( \langle X, \beta \rangle \). 

- Si la courbe est croissante : les grandes projections sont associées à de grandes valeurs de Y.
- Si la courbe est plate : la projection n'a pas d'effet sur Y.
- Si la courbe est décroissante : les grandes projections sont associées à de petites valeurs de Y.

