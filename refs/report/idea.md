

\noindent The next result is an extension of~\cite[Lemma~2]{Bousebata2023} to a conditional framework taking into account the random threshold.
\begin{lem}\label{lem:bousebata_random}
    Let any $h\in \RV_{\rho}(+\infty)$ with $\rho \in \R$ and i.i.d. random variables $\{Y_1,\dots,Y_n\}$ with common density $f\in \RV_{-1/\gamma -1}(+\infty)$, $\gamma >0$. Assume that $\rho \gamma < 1$. Let $2\le k\le n$ be some integer. Then, for any $y\geq 0$ and any $1\le i \le n$,  
    \begin{align*}
       \E\left( h(Y_i)1_{\{ Y_i\ge Y_{n-k+1,n}\}}\mid Y_{n-k+1,n}=y\right) & =    \f{(k-1)}{n}\cdot \frac{1}{\bar{F}(y)} \int_y^{+\infty} h(t) f(t)\mathrm{d}t.
    \end{align*}
    Moreover, when $y\to+\infty$, one has
    \begin{align*}
       \E\left( h(Y_i)1_{\{ Y_i\ge Y_{n-k+1,n}\}}\mid Y_{n-k+1,n}=y\right) &\sim \f{(k-1)}{n}\cdot \f{h(y) }{1-\rho\gamma}.
    \end{align*}
\end{lem}


this key point gives the behavior in the extreme case, this is an IMPORTANT lemma, we give a vizualization of that





puis conjointement 


The next lemma establishes a bound on the (random) tail moments of $\varphi(Y)\eps$.
\begin{lem}\label{lem:norm_noise_v2_random}
Assume that $\bar{F}\in \RV_{-1/\gamma}(+\infty)$, \eqref{hyp:test}, \eqref{hyp:link}, \eqref{hyp:noise_cond}, \eqref{hyp:2cgamma} and \eqref{hyp:qcgamma} hold. Let $k:=k_n\to+\infty$ be an integer deterministic sequence such that $ k / n \to 0$ and $y_{n,k}\sim U(n/k)$ as $n\to+\infty$. Let $\delta_{n,k}:=(g(y_{n,k})(k/n)^{1/q})^{-1}$. Then,
\begin{align*}
   \f{\zp[n]{\hat m_{\vfi(Y)\eps}(Y_{n-k+1,n})}}{m_{\vfi \cdot g(Y)}(y_{n,k})} = O_{\p}\left( \delta_{n,k} \right) \xrightarrow[n\to +\infty]{}0.
\end{align*}
\end{lem}



gamma augmente = meilleure confiance (visuellement page 16) car moins de variance vu qu'on a plus souvent des evenemnts extremes

tau less than (1-2/q)/(2gamma) is the recovery condition from phi taking noise into consideration

2/q less than 2kappagamma is the condition for recovery in general, identifiability of the system 




bigger tau towards negative leads sharper correlation decay

vu les plots figure 1, augmenter kappa permet de choisir un plus grand k et donc de dimimnuer la variance ! 




dans le workflow on doit donc choisir k en fonction de rho que l'on peut estimer (estimer combien de données pour estimer rho)



pour la decroissance ou croissance de k ça depend de rho 
le probleme c'est que rho doit etre negatif ce qui fait croitre k attention !!! c'


On cherche à satisfaire la condition classique d'équilibre pour la seconde variation régulière :
\[
k_n A\left(\frac{n}{k_n}\right) = O(1),
\]
avec $A(t) \in \mathrm{RV}_\rho(+\infty)$, c'est-à-dire que $A(t) \approx t^\rho$ pour $\rho < 0$.

On cherche $k_n$ de l'ordre d'une puissance de $n$, soit $k_n \sim c\, n^\alpha$ avec une constante $c > 0$ et un exposant $\alpha$. Alors :
\[
\frac{n}{k_n} \sim n^{1-\alpha},
\]
et donc
\[
A\left(\frac{n}{k_n}\right) \sim \left(n^{1-\alpha}\right)^\rho = n^{\rho(1-\alpha)}.
\]
D'où,
\[
k_n\, A\left(\frac{n}{k_n}\right)
\sim n^\alpha \cdot n^{\rho(1-\alpha)} = n^{\alpha + \rho(1-\alpha)}.
\]

On impose que cette quantité soit $O(1)$, donc l'exposant doit être nul~:
\[
\alpha + \rho(1-\alpha) = 0 \implies \alpha = \frac{-\rho}{1-\rho}.
\]

Pour être conforme à la littérature (voir \textit{e.g.} Haan & Ferreira), on préfère mettre cette relation sous la forme~:
\[
\frac{\alpha}{2} + \rho(1-\alpha) = 0 \implies \alpha = \frac{-2\rho}{1-2\rho}.
\]

Comme $\rho < 0$, on a $\alpha > 0$. Cela donne le choix optimal de $k_n$ selon $\rho$ :

\[
k_n \sim c\, n^{\frac{-2\rho}{1-2\rho}} \quad \text{avec} \quad \rho < 0.
\]




en fonction de rho on croit de 0 à -1, donc au lieu du sqrt on prend un peu plus en fonction de rho, et equivalence pour rho=-0.5

equation 3.2 est L2 convergence

equation 3.3 est l'equation majeur à satisfaire signal versus bruit, qui determine si le modele est identifiable ou non 



avec n(1/q−γκ)/(1−2ρ) , + q est grand et plus on est out, + vite ça converge, plus rho est petit idem plus vite àa converge avec moins de sample
 

Heuristiquement, on retrouve la décomposition classique en théorie des valeurs extrêmes :
\[
\mathrm{erreur}(n, k) \approx \underbrace{\mathrm{biais}(n, k)}_{\propto\, A(n/k)} + \underbrace{\mathrm{variance}^{1/2}(n, k)}_{\propto\, k^{-1/2} \times (\text{bruit})}
\]
c'est-à-dire,
\[
\mathrm{erreur}(n, k) \approx C_1\, A(n/k) + C_2\, k^{-1/2},
\]
où $A(n/k)$ contrôle le biais et $k^{-1/2}$ contrôle la variance bruitée.

Le choix optimal de $k_n$ s'écrit alors :
\[
k_n \sim c\, n^{-2\rho/(1-2\rho)}
\]
pour une constante $c>0$ et $\rho<0$.




En 2d c'est en fait beaucoup plus compliqué car PLS c'est une PCA dans l'objectif de, il n'y a pas de forme exacte, c'est un algo d'optimization donc bonjour les garanties
car on a une dépendance de beta2 envers beta1




in the workflow after estimating you can choose tau and which tau to choose, and there is no dependance in the decay with how we choose tau





tout le code devra etre mis reproductible (pour les finance bro tout tourne rapidement en 1 clic )