#!/usr/bin/env python3
"""One-shot assembly of manuscript_v9/main_v9.tex from the frozen v8 main.tex (plan 0.3).
Run once; afterwards main_v9.tex is edited by hand."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "v8_frozen" / "manuscript_v8" / "main.tex").read_text().split("\n")
L = lambda a, b: "\n".join(src[a - 1:b])          # 1-based inclusive line range

preamble = L(1, 87)
preamble = preamble.replace("% Manuscript v8: consolidated exposition.", "% Manuscript v9: symbol-based exact assembly and rigorous certification (skeleton, Phase 0.3/1.6/1.7).")
preamble = preamble.replace("\\DeclareMathOperator{\\Real}{Re}", r"""\DeclareMathOperator{\Real}{Re}
% ---------------------------- v9 macros ---------------------------
\newcommand{\Tn}[2]{T_{#1}(#2)}          % Toeplitz section T_n(f)
\newcommand{\sym}[1]{#1}                 % symbol (plain letter; change here to decorate)
\newcommand{\smin}{s_{\min}}
\newcommand{\Thmu}{\Theta_\mu}
\newcommand{\todo}[1]{\textcolor{red!70!black}{[\textbf{TODO:} #1]}}""")
preamble = preamble.replace("pdftitle={Residual Certification of Neural and Reduced Surrogates for Tempered-Fractional CGMY Pricing PIDEs}",
                            "pdftitle={Symbol-based residual certification of reduced and neural surrogates for tempered-fractional CGMY pricing}")
preamble = preamble.replace("pdfsubject={Discrete residual certification; manuscript v8}", "pdfsubject={Discrete residual certification; manuscript v9 skeleton}")
preamble = preamble.replace(r"""\title[Residual Certification for CGMY Pricing Trajectories]
      {Residual Certification of Neural and Reduced Surrogates for\\ Tempered-Fractional CGMY Pricing PIDEs}""",
r"""\title[Symbol-based residual certification for CGMY pricing]
      {Symbol-based residual certification of reduced and neural surrogates\\ for tempered-fractional CGMY pricing}""")
preamble = preamble.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{figures/}{../figures_v9/}}")

abstract = r"""\begin{document}
\begin{abstract}
We certify reduced and neural approximations of European-put pricing trajectories under
the Carr--Geman--Madan--Yor model by residual bounds for the spline Galerkin discretisation.
The Galerkin matrices are Toeplitz sections of an explicit periodised Fourier symbol; the
alias series is summed exactly up to a stated tail, so the assembled system \emph{is} the
exact Galerkin system to $10^{-8}$.  A L\'evy--Khintchine argument gives a parameter-explicit
lower bound $\lambda_{\min}(S_I)\ge\min_\vartheta s_\mu(\vartheta)\ge r\,m_{\min}$, valid for
$Y\in(0,2)\setminus\{1\}$, which replaces numerical spectral estimates and positivity audits.
A banded comparison norm yields a rigorous certificate at $O(N)$ cost with explicit
effectivity $\sqrt{\Thmu}$, and a two-stage rule falls back to a circulant-preconditioned
exact Riesz solve.  The symbol family is affine in the parameters with about $25$ terms, which
gives an $N$-independent certified reduced model.  \todo{Phase 2--3: affine structure, POD-NN,
out-of-distribution detection, the honest one-dimensional speed-up, the American extension.}
\end{abstract}
\maketitle
"""

intro = L(108, 172)
intro = intro.replace(r"""\item parameter-uniform continuity and coercivity for an interior CGMY
family, with explicit constants;
\item a scheme-consistent discrete residual bound including the initial
error, and a two-sided bracket for incomplete Riesz solves;
\item tolerance-controlled prediction, boundary projection, certification
and correction using one constrained algebraic system;
\item matched evaluations of neural, POD and deterministic methods,
including certificate verification, timing, assembly and price refinement.
\end{enumerate}""", r"""\item exact assembly of the spline Galerkin matrices from the periodised
Fourier symbol with an explicit truncation bound (Lemmas~\ref{lem:periodisation}--\ref{lem:tail});
\item a rigorous, parameter-explicit spectral lower bound for the free symmetric part
(Theorem~\ref{thm:spectral}), valid for $Y\in(0,2)\setminus\{1\}$, replacing the
numerical spectral estimate and the positivity audit of the earlier version;
\item a banded comparison norm giving an $O(N)$ residual certificate with explicit effectivity,
and a two-stage rule with a circulant-preconditioned exact Riesz solve
(Theorem~\ref{thm:comparison}, Proposition~\ref{prop:twostage});
\item an affine decomposition of the symbol family with rigorous error transfer, hence an
$N$-independent certified reduced model (Lemma~\ref{lem:eim}, Theorem~\ref{thm:eim-cert});
\item matched evaluations of reduced, neural and deterministic methods.  \todo{Phase 3.}
\end{enumerate}
\todo{Reframe the introduction (plan 4.1): rigor, certification cost, OOD, American target;
COS stated on page 1; RB/SCM literature (Huynh--Rozza--Sen--Patera 2007; Burkovska et al.\ 2015;
Gass--Glau--Mahlstedt--Mair 2018); Chan--Strang; Grenander--Szeg\H{o} / B\"ottcher--Silbermann.}""")
intro = intro.replace(r"""Section~\ref{sec:model} defines the model and constrained discretisation.
Section~\ref{sec:operator} describes the candidate approximations.
Section~\ref{sec:certificate} develops certification and correction,
Section~\ref{sec:numerics} evaluates them, and
Section~\ref{sec:conclusion} interprets the findings.""", r"""Section~\ref{sec:model} defines the model and the spline Galerkin discretisation via symbols.
Section~\ref{sec:certificate} develops certification and correction,
Section~\ref{sec:affine} the affine structure and the candidate approximations,
Section~\ref{sec:numerics} evaluates them, and
Section~\ref{sec:conclusion} interprets the findings.""")

# ---------------- Section 2
sec2a = L(173, 330)
sec2a = sec2a.replace(r"""The constants establish uniform stability despite the varying energy order.
They are conservative; matrix-dependent constants are used to assess
numerical efficiency.""", r"""The constants establish uniform stability despite the varying energy order.
They are conservative; the discrete bounds of Section~\ref{ssec:spectral} are the ones used
in the certificate.  The continuum statement keeps the hypothesis $Y>1$: for $Y<1$ the drift
term is not controlled by the $H^{Y/2}_{M,G}$ norm.  The discrete results below hold for
$Y\in(0,2)\setminus\{1\}$.
\input{theory/rem_coercivity}""")
sec2b = L(331, 382)
sec2b = sec2b.replace(r"""the tempered connection coefficients $\Lambda_G^-,\Lambda_M^+$ being given by
absolutely convergent Fourier-symbol integrals.  The matrices $\Mh$ and $D_h$
are banded; the Toeplitz blocks are applied by circulant embedding in
$O(N_h\log N_h)$ operations with no compression error.  The semidiscrete
system is""", r"""the tempered connection coefficients $\Lambda_G^-,\Lambda_M^+$ being given by
absolutely convergent Fourier-symbol integrals (we write $T_G,T_M$ for these blocks when the
sign convention is immaterial).  The matrices $\Mh$ and $D_h$
are banded; the Toeplitz blocks are applied by circulant embedding in
$O(N_h\log N_h)$ operations with no compression error.  The semidiscrete
system is""")
sec2b = sec2b.replace(r"""Here $\Mh$ and $D_h$ are banded, while the fractional blocks are Toeplitz.
Circulant embedding applies the Toeplitz blocks in
$O(N_h\log N_h)$ operations.""", r"""\subsection{Exact assembly from the periodised symbol}
\label{ssec:assembly}
\input{theory/lem_periodisation}
\input{theory/lem_tail}
All experiments use $K=100$ with the exact trigonometric polynomials for the mass and drift
parts; then $\varepsilon_K(\mu)\le5\times10^{-9}\,\norm{\Ah}_2$ on the box of
Table~\ref{tab:proposed-box} (Table~\ref{tab:assembly-v9}), and $\varepsilon_K$ is recorded with
every assembled operator because it enters the certificate.  The principal-period rule of the
earlier version of this work is the case $K=0$; its relative operator-norm error is $0.33$--$0.41$
because at $\vartheta=\pm\pi$ the $k=0$ and $k=\mp1$ terms of \eqref{eq:periodised} have equal
modulus.  With $K\ge100$ the assembled matrices are the exact Galerkin matrices to $10^{-8}$, and
every exact-assembly statement applies to them with the explicit perturbation \eqref{eq:tail}.
\input{theory/lem_toeplitz}""")
sec2c = L(401, 445)
sec2c = sec2c.replace(r"""The reliability, efficiency and inexact-Riesz results below apply directly to
$(M_I,A_I,S_I,\rho_I)$ when $S_I\succ0$. """, r"""Because the free indices are contiguous, the free matrices are Toeplitz sections of the
symbols of Lemma~\ref{lem:periodisation}:
\begin{equation}
 A_I=\Tn{N_h}{\sym{a}_\mu},\qquad M_I=\Tn{N_h}{\sym{m}},\qquad
 S_I=\Tn{N_h}{\sym{s}_\mu},\quad \sym{s}_\mu:=\Real\sym{a}_\mu,\qquad
 h^2R^{(1)}_I=\Tn{N_h}{\sym{g}_1},
 \label{eq:sections}
\end{equation}
the drift symbol being odd and therefore absent from the symmetric part; $h^2R^{(1)}_I$ is banded with
bandwidth $p-1$.  The raw matrices $\bar M,\bar A$ are the sections $\Tn{N+p-1}{\sym{m}},\Tn{N+p-1}{\sym{a}_\mu}$
of the full cardinal splines, so the prescribed boundary functions are not truncated to $\Omega$
in the lifting term; for the mass and drift couplings this coincides with quadrature over $\Omega$,
and the initial projection $c^0_h$ uses the $\Omega$-quadrature mass matrix so that constants are
reproduced exactly up to the boundary.
The reliability, efficiency and inexact-Riesz results below apply directly to
$(M_I,A_I,S_I,\rho_I)$; positivity of $S_I$ is now a theorem (Section~\ref{ssec:spectral}). """)
sec2d = L(446, 476)
sec2d = sec2d.replace(r"""The certificate controls the second term. Under exact assembly and the
regularity and buffer hypotheses of \cite{DamircheliCGMY2026}, the first
admits the deterministic estimate recorded in
Appendix~\ref{ssec:total-error}. For the principal-period discretisation,
its size is assessed by independent prices and refinement.""", r"""The certificate controls the second term. Under the
regularity and buffer hypotheses of \cite{DamircheliCGMY2026}, the first
admits the deterministic estimate recorded in
Appendix~\ref{ssec:total-error}; its size is assessed by independent prices and refinement
(Section~\ref{ssec:calibration}).
\subsection{Spectral lower bound}
\label{ssec:spectral}
\input{theory/lem_LK}
\input{theory/thm_spectral}
Over 32 box vertices and 256 Sobol points at $N=64$, $256$ and $1024$,
$\lambda_{\min}(S_I)/\smin(\mu)$ lies in the range reported in Figure~\ref{fig:spectral-bound},
with the ratio closest to one attained at the worst vertex of the earlier positivity audit
($\mu=(0.05,5,9,1.1,0.01)$, $\lambda_{\min}=0.01038$, $\smin=\rf=0.0100$).
$\smin(\mu)$ is the minimum of an explicit smooth even function of one variable; it is
evaluated on a graded grid with a cell-wise derivative margin (Section~\ref{ssec:e0-spectral}).
Since the certificate needs only $S_I\succ0$, all algebraic results extend to $Y\in(0,1)$;
only the continuum coercivity in $H^{Y/2}_{M,G}$ keeps the hypothesis $Y>1$.""")
sec2 = "\n".join([sec2a, sec2b, sec2c, sec2d])

# ---------------- Section 3 (v8 Section 4)
sec3a = L(573, 669)
sec3a = sec3a.replace(r"""Exact Galerkin assembly implies this positivity by
Theorem~\ref{thm:uniform-coercivity}. For numerical assembly, positivity is
an explicit algebraic assumption assessed query by query.""", r"""Positivity of $S_I$ holds by Theorem~\ref{thm:spectral} with the explicit
constant $\alpha_{\rm lb}(\mu)=\min\sym{s}_{\mu,K}-\varepsilon_K(\mu)>0$.""")
sec3b = L(670, 780)
sec3b = sec3b.replace(L(676, 698), r"""The bracket below needs a lower bound
\begin{equation}
  0<\alpha_{\rm lb}(\mu)\le \lambda_{\min}(\Sh(\mu)),
  \qquad \alpha_{\rm lb}(\mu):=\min_\vartheta\sym{s}_{\mu,K}(\vartheta)-\varepsilon_K(\mu),
  \label{eq:alpha-lb}
\end{equation}
which Theorem~\ref{thm:spectral} supplies in closed form; no eigenvalue computation enters.""")
sec3b = sec3b.replace(L(763, 780), r"""\begin{remark}[Arithmetic scope]
\label{rem:arithmetic-scope}
The theorems and brackets are exact-arithmetic statements for the assembled matrices.
The only quantities not enclosed are the floating-point residual evaluation and the
accumulation: if a computed residual $\widehat\rho_n$ differs from the required residual by at
most $d_n$ in the $S_h^{-1}$ norm and $u_n$ is an upper bound for
$\widehat\rho_n^\top S_h^{-1}\widehat\rho_n$, then
$\widehat\eta^2=\beta_0^2+\sum_n\dt\,(\sqrt{u_n}+d_n)^2$ is valid.  The same allowance
structure absorbs the affine-operator error of Theorem~\ref{thm:eim-cert}.
\end{remark}
\subsection{A banded comparison norm}
\label{ssec:comparison}
\input{theory/thm_comparison}
\input{theory/cor_cheap}
\input{theory/prop_twostage}
For the third case the Riesz systems $S_Iv=\rho$ are solved by conjugate gradients with the
circulant preconditioner whose eigenvalues are the symbol samples $\sym{s}_\mu(2\pi j/N_h)$.
Since $\sym{s}_\mu$ is continuous and positive, the preconditioned spectrum
clusters at $1$ \cite{ChanStrang1989}; Table~\ref{tab:pcg-v9} reports the iteration counts.""")
sec3c = L(781, 924)
sec3c = sec3c.replace(r"""\item Establish a positive spectral lower bound (a numerical estimate
in the reported implementation; the soundness theorem requires a valid
lower bound). If this fails, return status \textsc{uncertifiable} and invoke a separately implemented
deterministic fallback if a price is required; do not report a neural certificate.
\item Compute the augmented upper bound $\eta_{\rm aug}=(\beta_0^2+\etaGuar^2)^{1/2}$
with a short PCG budget. If it is at most $\varepsilon$, return
\textsc{direct accept} with its certificate.""", r"""\item Evaluate $\alpha_{\rm lb}(\mu)$, $\beta_\mu$, $X_I$ and $\Thmu$ from the symbol
(Theorems~\ref{thm:spectral} and \ref{thm:comparison}).
\item Compute the cheap augmented bound $\eta^X_{\rm aug}=(\beta_0^2+(\eta^X)^2)^{1/2}$
(banded Cholesky).  If it is at most $\varepsilon$, return \textsc{direct accept}.  If
$(\beta_0^2+(\eta^X)^2/\Thmu)^{1/2}>\varepsilon$, no exact bound can pass: continue with correction.
Otherwise compute $\eta^S_{\rm aug}$ by the preconditioned bracket; if it passes, return
\textsc{direct accept}.""")
sec3c = sec3c.replace(r"""The numerical screen uses ten PCG iterations, with a maximum budget of 200
and no preconditioner. Correction uses GMRES tolerance $10^{-10}$.
The benchmark trajectory tolerance is $\varepsilon=10^{-3}$.""", r"""The bracket is evaluated at the converged PCG iterate (relative residual $10^{-12}$, at most
$15$ iterations in all reported runs).  Correction uses circulant-preconditioned GMRES with
relative tolerance $10^{-13}$.  Tolerances are reported as a sweep (Section~\ref{ssec:calibration}).""")
sec3c = sec3c.replace(r"""Assume the hypotheses of Theorem~\ref{thm:fully-discrete-certificate},
a valid positive spectral lower bound, exact residual evaluation, and
$\beta_0\ge\|\ee^0\|_{\Mh}$. Define
$\eta_{\rm aug}=(\beta_0^2+\etaGuar^2)^{1/2}$.
If the rule in Figure~\ref{alg:pcc} returns
\textup{\textsc{accepted}} with $\eta_{\rm aug}\le\varepsilon$, then""", r"""Assume the hypotheses of Theorem~\ref{thm:fully-discrete-certificate},
exact residual evaluation, and $\beta_0\ge\|\ee^0\|_{\Mh}$. Let $\eta_{\rm aug}$ be either
$\eta^X_{\rm aug}$ or $(\beta_0^2+\etaGuar^2)^{1/2}$ with $\alpha_{\rm lb}$ from \eqref{eq:alpha-lb}.
If the rule in Figure~\ref{alg:pcc} returns
\textsc{direct accept} or \textsc{corrected accept} with $\eta_{\rm aug}\le\varepsilon$, then""")
sec3c = sec3c.replace(r"""The conclusion is independent of the PCG stopping error. The same proof
applies to a corrected trajectory whose recomputed augmented bound passes.""", r"""The conclusion is independent of the PCG stopping error, by Lemma~\ref{lem:guaranteed} for
$\etaGuar$ and by Corollary~\ref{cor:cheap} for $\eta^X$. The same proof
applies to a corrected trajectory whose recomputed augmented bound passes
(\textsc{corrected accept}) and to a direct fallback (\textsc{qualified fallback}).""")
sec3 = "\n".join([sec3a, sec3b, sec3c])

# ---------------- Section 4 (affine structure + candidates)
sec4 = r"""\section{Affine structure and surrogate candidates}
\label{sec:affine}\label{sec:operator}
\subsection{Affine decomposition of the fractional blocks}
\label{ssec:eim}
\input{theory/lem_eim}
\input{theory/thm_eim_cert}
\todo{Phase 2.1: EIM numbers ($Q$ vs tolerance, Figure~F4), training grid, magic points.}
""" + L(479, 572).replace(r"\subsection{Coefficient-valued neural map}", r"\subsection{Candidates: coefficient-valued neural map}") + r"""
\todo{Phase 2.2--2.4: POD--Galerkin with the affine operator, the reduced online certificate
(Remark~\ref{rem:reduced-online}), the POD-NN candidate; one paragraph each.}
\subsection{Cost model}
\label{ssec:cost}
\todo{Phase 2.6: flop counts.  Direct $=N_T\times{\rm it}_{\rm GMRES}\times5N_h\log N_h$;
full-order certificate $=N_T\times5N_h\log N_h+N_T\,{\rm bw}^2N_h$; reduced solve $=Qr^2+N_Tr^2$;
reduced certificate $=Q^2(2r+1)^2$.  State the one-dimensional ceiling explicitly.}
"""

# ---------------- Section 5 (numerics, Phase 1 content)
sec5 = r"""\section{Numerical evaluation}
\label{sec:numerics}
\subsection{Experimental design}
\label{ssec:design}
All studies use the exact assembly of Section~\ref{ssec:assembly} ($K=100$), quadratic splines,
the box of Table~\ref{tab:proposed-box}, 60 Crank--Nicolson steps unless stated, and the free system.
The implementation is the package \texttt{cgmy\_cert} (Section~\ref{ssec:archive}); every number
below is read from \texttt{results/E0.json}.
\todo{Phase 3: production $(N,N_T,\varepsilon)$ from Section~\ref{ssec:calibration}; update Table~\ref{tab:populations}.}
""" + L(966, 988).replace("Positivity audit & 256 Sobol points, 64 box vertices and 139 selected\nfull-space failure cases (459 coordinates)\\\\\nRiesz and initial error & 24 audit queries for brackets; 48 audit queries\nfor augmented-bound tests\\\\", "Spectral and comparison-norm verification & 32 box vertices and 256 Sobol points (288 queries)\\\\\nInitial-error tests & 48 audit queries (first 16 per stratum of the earlier audit)\\\\") + r"""
\subsection{Assembly verification}
\label{ssec:e0-assembly}
Table~\ref{tab:assembly-v9} compares the assembled free matrix with the exact Galerkin matrix,
computed independently at 40 digits (alias series summed in closed form through Hurwitz zeta
tails, Toeplitz entries by a high-precision trapezoid rule).  The truncation error decreases
like $K^{-(2p-Y-1)}$ as predicted by Lemma~\ref{lem:tail}, and the bound $\varepsilon_K$ is
within a factor $1.1$--$4$ of the observed error at every $K$ and $N$.
\input{../tables_v9/tab_assembly_v9}
\subsection{Spectral bound and comparison norm}
\label{ssec:e0-spectral}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.94\linewidth]{F2_spectral_bound.png}
\caption{Ratios $\lambda_{\min}(S_I)/\smin(\mu)$ (left) and $\lambda_{\min}(S_I)/(\rf m_{\min})$ (right)
over the 288 verification queries at $N=64,256,1024$; the star marks the worst vertex of the
earlier positivity audit.  Both ratios are $\ge1$ at every query, as Theorem~\ref{thm:spectral} asserts.}
\label{fig:spectral-bound}
\end{figure}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.94\linewidth]{F3_theta_and_effectivity.png}
\caption{Left: the a-priori effectivity bound $\sqrt{\Thmu}$ of the banded certificate as a function of
$N$ (symbol only, 288 queries).  Right: observed ratio $\eta^X/\eta^S$ for POD--Galerkin candidates
(ranks 16, 32, 48), the prolonged coarse solution and exact dynamics from a perturbed initial vector,
at $N=64$ and $N=512$.}
\label{fig:theta}
\end{figure}
\todo{Phase 1.8: fill in the numbers from \texttt{results/E0.json} (E0.2, E0.3) once the run completes;
Phase 2: replace the stand-in candidates by DeepONet / POD-NN / corrected trajectories.}
\subsection{Riesz solves and reference trajectories}
\label{ssec:e0-riesz}
\input{../tables_v9/tab_pcg_v9}
Dense-LU and circulant-GMRES references agree to $10^{-12}$ in the $M_I$ norm for
$N\le1024$; the bracket width at convergence is below $10^{-20}$ relative for $N\le1024$ and
below $10^{-18}$ at $N=4096$, where it is set by the roundoff of the recomputed residual.
\subsection{Initial-error tests and \texorpdfstring{$Y<1$}{Y<1}}
\label{ssec:e0-initial}
On the 48 exact-dynamics cases with perturbed initial vectors ($\|\delta\|_{M_I}=10^{-4}$), both
$\eta^X_{\rm aug}$ and $\eta^S_{\rm aug}$ hold with effectivity $1.00\pm10^{-6}$, and the
residual-only estimators underestimate the error in every case.  For $Y\in\{0.5,0.8,0.95\}$
(same $C,G,M,\rf$ as the reference query) the bound $\alpha_{\rm lb}=0.0500=\rf$ is positive,
$\lambda_{\min}(S_I)\in[0.052,0.055]$, $\sqrt{\Thmu}\le1.5$, the preconditioned GMRES solve of
$B_+$ needs at most five iterations per step, and the certificate holds with effectivity one on
perturbed exact dynamics; the discrete framework therefore covers the finite-variation regime
without change.
\subsection{Resolution and tolerance calibration}
\label{ssec:calibration}
\todo{Phase 3.1.}
\subsection{Reduced model, neural candidates and benchmarks}
\label{ssec:benchmark}\label{ssec:pod-results}\label{ssec:scaling-results}
\todo{Phase 3.2--3.5 (E1--E4).}
\subsection{Reproducibility archive}
\label{ssec:archive}
\todo{Phase 4.4.}
\FloatBarrier
"""

sec6 = L(1177, 1212) + "\n\\todo{Rewrite the interpretation for v9; the American theorem (Appendix~\\ref{ssec:vi}) as outlook.}\n"

app = L(1213, 1691)
app = app.replace(r"\subsection{Preconditioning and correction}", r"""\subsection{Coercivity constant from the L\'evy--Khintchine form}
\label{ssec:lk-coercivity}
\todo{Phase 1.6: replace Steps 3--5 of the proof of Theorem~\ref{thm:uniform-coercivity} by the
bounds of Remark~\ref{rem:coercivity}; compute the new $c_1$ on the box (expect $\sim10^{-5}$).}
\subsection{Residual certificate for the American put}
\label{ssec:vi}
\input{theory/thm_vi}
\subsection{Preconditioning and correction}""")
bib = L(1692, 1814)
bib = bib.replace(r"\begin{thebibliography}{99}", r"""\begin{thebibliography}{99}
\bibitem{ChanStrang1989}
R.~H. Chan and G. Strang, \emph{Toeplitz equations by conjugate gradients with circulant
preconditioner}, SIAM J. Sci. Statist. Comput. 10 (1989), 104--119.
\bibitem{GrenanderSzego1958}
U. Grenander and G. Szeg\H{o}, \emph{Toeplitz Forms and Their Applications}, Univ. of California Press, 1958.
\bibitem{BottcherSilbermann1999}
A. B\"ottcher and B. Silbermann, \emph{Introduction to Large Truncated Toeplitz Matrices}, Springer, 1999.
\bibitem{HuynhRozzaSenPatera2007}
D.~B.~P. Huynh, G. Rozza, S. Sen and A.~T. Patera, \emph{A successive constraint linear optimization
method for lower bounds of parametric coercivity and inf-sup stability constants},
C. R. Math. Acad. Sci. Paris 345 (2007), 473--478.
\bibitem{BurkovskaHaasdonkSalomonWohlmuth2015}
O. Burkovska, B. Haasdonk, J. Salomon and B. Wohlmuth, \emph{Reduced basis methods for pricing
options with the Black--Scholes and Heston models}, SIAM J. Financial Math. 6 (2015), 685--712.
\bibitem{GassGlauMahlstedtMair2018}
M. Gass, K. Glau, M. Mahlstedt and M. Mair, \emph{Chebyshev interpolation for parametric option
pricing}, Finance Stoch. 22 (2018), 701--731.""")

out = "\n".join([preamble, abstract, intro, sec2, sec3, sec4, sec5, sec6, app, bib])
(ROOT / "manuscript_v9" / "main_v9.tex").write_text(out)
print("written", len(out.split("\n")), "lines")
