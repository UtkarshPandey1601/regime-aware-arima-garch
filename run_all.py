import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--from", dest="start", default=None, help="start from this stage name")
ap.add_argument("--only", nargs="*", default=None)
a = ap.parse_args()
J = str(a.jobs)
S = [("prepare", ["src/prepare_data.py"]), ("finalize", ["src/finalize_sample.py"]), ("diagnostics", ["src/diagnostics.py"]),
     ("splits_arma", ["src/splits_arma.py"]), ("baseline", ["src/walkforward.py"]), ("evaluation", ["src/evaluation.py"]),
     ("grid", ["src/models.py", "--jobs", J]), ("eval_grid", ["src/eval_grid.py"]), ("hmm", ["src/hmm_regimes.py"]),
     ("regime_wf", ["src/regime_wf.py", "--spec", "GARCH-normal", "GJR-normal", "--jobs", J]),
     ("regime_report", ["src/regime_report.py"]), ("var_backtest", ["src/var_backtest.py"]), ("var_compare", ["src/var_compare.py"]),
     ("validation", ["src/select_validation.py", "--jobs", J]), ("robustness", ["src/robustness.py", "--jobs", J]),
     ("mcs", ["src/mcs.py"]), ("figures", ["src/make_figures.py"]), ("tests", ["-m", "pytest", "-q", "tests"])]
names = [n for n, _ in S]
if a.start:
    S = S[names.index(a.start):]
if a.only:
    S = [x for x in S if x[0] in a.only]
env = dict(os.environ, OMP_NUM_THREADS="1")
for name, cmd in S:
    t0 = time.time(); print(f"\n##### {name} #####", flush=True)
    rc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env).returncode
    print(f"[{name}] exit {rc} in {time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        sys.exit(f"Stage '{name}' failed; fix it, then resume with:  python run_all.py --from {name}")
print("\nAll stages finished.")