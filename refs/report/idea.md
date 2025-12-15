

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

