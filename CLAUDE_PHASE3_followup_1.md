# Phase 3 follow-up 1: the per-query spectral work dominates the certificate (fix before E2's numbers are final)

The two bugs you found (Q_used inside the timer, the npz re-read) are the right fixes; record both in
`DEVIATIONS.md`. The 2-query timing you printed, however, shows a third problem that is not a bug
but a design cost, and it decides the paper's headline number ("certificate fraction of the solve"):

    direct            247-278 ms
    direct + banded   738-778 ms      -> the banded certificate costs ~490 ms, twice the solve

The banded certificate itself is cheap: 240 residual evaluations (one FFT pair of length 2048 each,
~25 ms in total) plus one banded Cholesky and 240 banded solves (a few ms). The ~450 ms that remain
are `spectral_data` — the symbol and its derivative on the graded grid (~19 400 points × 201 aliases,
twice) to get alpha_lb, beta_mu, Theta_mu. That evaluation was designed in Phase 1 for rigour on
the 288-query verification set, not for the online path, and it is now 15× the certificate it
serves. Section 4.4 of the paper claims the full-order certificate is about 1/it_B of the solve,
which is true of the residual and banded solve but false of the current online path.

## What to do (before the final E2 run; the current run can be left to finish as a dry run)

1. Add a fast mode to `spectral_data` for the online path: evaluate `symbols()` with derivative on a
   coarse graded grid — about 2^9 uniform points plus 200–300 geometric points on the scale
   min(G, M) h near theta = 0 — with the same cell-wise derivative margins (`_grid_extreme`), and use
   the rigorous adaptive alias count already used for the conversion factor (state the bound it
   uses). The margins keep alpha_lb, beta and Theta rigorous on any grid; only tightness is lost.
   Target: <= 10 ms at N = 1024 on one thread. If the uniform part is a multiple of N_h this is also
   where the circulant preconditioner samples come from; otherwise keep reading those from the
   assembly grid (already computed).
2. Validate the fast mode against the fine one on the 288-query spectral set at N = 1024: report the
   ratios alpha_lb_fast / alpha_lb_fine, beta_fast / beta_fine, Theta_fast / Theta_fine (all must be
   <= 1, <= 1, >= 1 respectively, i.e. on the safe side), their medians and extremes, and the
   resulting change of eta^X on the validation30 POD-Galerkin candidates (expect a few percent).
   Keep the fine mode as the default for verification scripts (E0) and tests; the online path
   (direct + certificate, the three-stage rule, correction + recertify) uses the fast mode.
3. Also check where the assembly time goes in the cold direct solve: `symbols()` on the assembly
   grid (L = 4 N_h = 4088 points × 201 aliases) is ~50–100 ms of the 247 ms. The adaptive alias
   count applies there too, with eps_K recomputed for the K actually used (it enters alpha_lb).
   Report the split assembly / forcing / initial projection / preconditioner / march for the cold
   direct solve; the paper needs it for Table 7 (cost model, item 0 of the brief).
4. Then rerun E2 (bench + refine + figs) with the timing protocol, and in E2's JSON record for every
   certified method the split: symbol work (assembly + spectral) / residual / banded or PCG /
   reduced stage / correction / recertify. The "certificate fraction" in the paper will be reported
   as two numbers: the certificate proper (residual + banded solve) and the certificate including
   the per-query symbol work.

E3 can proceed in parallel as planned (its verdicts do not depend on timing); E4 must wait for the
fast mode, since its wall-clock totals are the result.

## For `NOTES_FOR_PAPER.md`
- the three-component split of the full-order certificate cost and the fast/fine spectral ratios;
- that `direct + banded` on the 2-query test went from 740 ms to the value after the fix.
