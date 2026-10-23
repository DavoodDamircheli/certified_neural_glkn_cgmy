#!/usr/bin/env python3
"""Regenerate every table in tables_v9/ and every figure in figures_v9/ (PNG + PDF) from results/*.json only.

No solver call, no data file: each step below is the plot/table function of the script that produced the result file,
called directly (the scripts' main() functions are not run, so no result file is rewritten).
Usage: python3 scripts/build_outputs.py      (or: bash build.sh, make all)
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results"


def js(name):
    return json.loads((RES / name).read_text())


def steps():
    import run_E0, run_eim, run_E3_1, run_cost_model, fig_F5a, run_E1, run_E2, run_E3, run_E4
    e0 = js("E0.json")
    return [
        ("tables_v9/tab_assembly_v9.tex", lambda: run_E0.plot_e01(e0)),
        ("figures_v9/F2_spectral_bound", lambda: run_E0.plot_e02(e0)),
        ("figures_v9/F3_theta_and_effectivity", lambda: run_E0.plot_e03(e0)),
        ("tables_v9/tab_pcg_v9.tex", lambda: run_E0.plot_e04(e0)),
        ("figures_v9/F4_eim", lambda: run_eim.do_figure(js("eim.json"))),
        ("tables_v9/tab_budget_v9.tex", lambda: run_E3_1.do_table(js("E3_1_calibration.json"))),
        ("tables_v9/tab_cost_v9.tex", lambda: run_cost_model.write_table(js("cost_model.json"))),
        ("figures_v9/F5a_acceptance", lambda: _f5a(fig_F5a)),
        ("figures_v9/F5b, F5c, tables_v9/tab_pod_ranks_v9.tex", lambda: run_E1.do_figs(js("E1.json"))),
        ("figures_v9/F6a, F6b, tables_v9/tab_benchmark_v9.tex", lambda: run_E2.do_figs(js("E2.json"))),
        ("figures_v9/F7_ood", lambda: run_E3.do_fig(js("E3.json"))),
        ("tables_v9/tab_ood_v9.tex", lambda: run_E3.do_table_ood(js("E3.json"))),
        ("figures_v9/F8_amortisation", lambda: run_E4.plot_f8()),
    ]


def _f5a(mod):
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        mod.main()
    finally:
        sys.argv = argv


def main():
    t0 = time.time()
    for name, fn in steps():
        t = time.time()
        fn()
        print(f"built {name} ({time.time() - t:.1f}s)", flush=True)
    print(f"all tables and figures rebuilt from results/*.json in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
