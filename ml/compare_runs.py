"""Collate eval_results.json from several training runs and apply the predictor success bar.

    # compare the input-length experiment
    python -m ml.compare_runs artifacts/maxlen_128 artifacts/maxlen_256 artifacts/maxlen_512

    # compare the learning curve
    python -m ml.compare_runs artifacts/curve_500 artifacts/curve_1000 artifacts/curve_2000

Reads only; runs nothing. Each directory must contain the eval_results.json that ml.train wrote.

The success bar (unchanged from the project brief) - DistilBERT is selected over the
prompt-length baseline only if all three hold:

  1. lower test MAE than the baseline,
  2. severe-underprediction rate <= the baseline's,
  3. predictor inference overhead < 5% of median Llama service time.

Criterion 3 needs the median service time of the serving backend. It is read from the base label
run's measured latencies by default, so it reflects this machine rather than a guess. A predictor
timed on CPU while the GPU was busy is NOT a final number; the report says which device was used.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from datagen import config
from datagen.status import _iter_tolerant

OVERHEAD_BUDGET = 0.05  # predictor latency must stay under this fraction of service time
BASELINE_KEY = "baseline_prompt_length_linreg"


def median_service_time_ms(base_dir: str | Path = config.LABELS_DIR_REAL) -> float | None:
    """Median measured Llama latency per generation, from the base run's own records."""
    runs = list(_iter_tolerant(Path(base_dir) / config.RUNS_FILE))
    lats = [r["latency_ms"] for r in runs if isinstance(r.get("latency_ms"), (int, float))]
    return float(st.median(lats)) if lats else None


def read_run(d: str | Path) -> dict:
    d = Path(d)
    path = d / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run ml.train with --output-dir {d} first")
    res = json.loads(path.read_text(encoding="utf-8"))
    test = res.get("test", {})
    settings = res.get("run_settings", {})
    db, base, lat = test.get("distilbert", {}), test.get(BASELINE_KEY, {}), test.get("distilbert_latency", {})
    return {
        "run": d.name,
        "data_path": res.get("data_path"),
        "max_length": settings.get("max_length"),
        "n_train": settings.get("n_train"),
        "n_test": db.get("n"),
        "epochs_run": settings.get("epochs_run"),
        "early_stopped": settings.get("early_stopped"),
        "device": lat.get("device"),
        "distilbert_mae": db.get("mae"),
        "distilbert_p90_abs_error": db.get("p90_abs_error"),
        "distilbert_underprediction_rate": db.get("underprediction_rate"),
        "distilbert_severe_underprediction_rate": db.get("severe_underprediction_rate"),
        "baseline_mae": base.get("mae"),
        "baseline_p90_abs_error": base.get("p90_abs_error"),
        "baseline_severe_underprediction_rate": base.get("severe_underprediction_rate"),
        "predictor_latency_median_ms": lat.get("median_ms"),
        "predictor_latency_p95_ms": lat.get("p95_ms"),
        "warning": res.get("warning"),
    }


def apply_success_bar(row: dict, service_ms: float | None) -> dict:
    """Evaluate the three criteria. None for a criterion means 'could not be assessed'."""
    db_mae, b_mae = row["distilbert_mae"], row["baseline_mae"]
    db_sev, b_sev = row["distilbert_severe_underprediction_rate"], row["baseline_severe_underprediction_rate"]
    lat = row["predictor_latency_median_ms"]

    c1 = (db_mae < b_mae) if (db_mae is not None and b_mae is not None) else None
    c2 = (db_sev <= b_sev) if (db_sev is not None and b_sev is not None) else None
    overhead = (lat / service_ms) if (lat is not None and service_ms) else None
    c3 = (overhead < OVERHEAD_BUDGET) if overhead is not None else None

    checks = {"beats_baseline_mae": c1, "severe_underprediction_ok": c2, "overhead_under_5pct": c3}
    if any(v is None for v in checks.values()):
        verdict = "indeterminate"
    elif all(checks.values()):
        verdict = "distilbert"
    else:
        verdict = "baseline"
    return {**checks, "overhead_fraction": round(overhead, 6) if overhead is not None else None,
            "selected_predictor": verdict}


def compare(dirs: list[str | Path], service_ms: float | None) -> dict:
    rows = [read_run(d) for d in dirs]
    for r in rows:
        r["success_bar"] = apply_success_bar(r, service_ms)
    ranked = [r for r in rows if r["distilbert_mae"] is not None]
    ranked.sort(key=lambda r: r["distilbert_mae"])
    return {
        "median_llama_service_time_ms": service_ms,
        "overhead_budget_fraction": OVERHEAD_BUDGET,
        "runs": rows,
        "best_by_distilbert_mae": ranked[0]["run"] if ranked else None,
        "synthetic_warning": next((r["warning"] for r in rows if r.get("warning")), None),
    }


def _fmt(v, nd=1):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def print_table(report: dict) -> None:
    svc = report["median_llama_service_time_ms"]
    print(f"median Llama service time: {_fmt(svc)} ms  "
          f"(5% budget = {_fmt(svc * OVERHEAD_BUDGET) if svc else 'n/a'} ms)\n")
    hdr = (f"{'run':22s} {'maxlen':>6s} {'ntr':>5s} {'dev':>5s} "
           f"{'DB mae':>8s} {'BL mae':>8s} {'DB p90ae':>9s} {'DB sev':>7s} {'BL sev':>7s} "
           f"{'lat ms':>7s} {'ovhd':>7s} {'pick':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in report["runs"]:
        sb = r["success_bar"]
        print(f"{r['run'][:22]:22s} {_fmt(r['max_length'],0):>6s} {_fmt(r['n_train'],0):>5s} "
              f"{(r['device'] or 'n/a')[:5]:>5s} "
              f"{_fmt(r['distilbert_mae']):>8s} {_fmt(r['baseline_mae']):>8s} "
              f"{_fmt(r['distilbert_p90_abs_error']):>9s} "
              f"{_fmt(r['distilbert_severe_underprediction_rate'],3):>7s} "
              f"{_fmt(r['baseline_severe_underprediction_rate'],3):>7s} "
              f"{_fmt(r['predictor_latency_median_ms'],2):>7s} "
              f"{(_fmt(sb['overhead_fraction']*100,2)+'%') if sb['overhead_fraction'] is not None else 'n/a':>7s} "
              f"{sb['selected_predictor']:>9s}")
    print(f"\nbest DistilBERT test MAE: {report['best_by_distilbert_mae']}")
    if report["synthetic_warning"]:
        print(f"\n!! {report['synthetic_warning']}")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dirs", nargs="+", help="artifact directories containing eval_results.json")
    parser.add_argument("--base-dir", default=str(config.LABELS_DIR_REAL),
                        help="base label run used to measure median Llama service time")
    parser.add_argument("--median-service-ms", type=float, default=None,
                        help="override the measured median service time")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)
    svc = args.median_service_ms if args.median_service_ms is not None else median_service_time_ms(args.base_dir)
    report = compare(args.dirs, svc)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_table(report)
    return report


if __name__ == "__main__":
    main()
