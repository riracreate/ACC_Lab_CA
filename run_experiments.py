"""
Run the whole evaluation and save every table/figure you need for the report.

  python run_experiments.py --azure-dir data/azure          # real Azure Functions 2019 trace (recommended)
  python run_experiments.py --synthetic                     # no download needed (smoke test only)
  python run_experiments.py --azure-dir data/azure --pick 3 --peak-rpm 120

Outputs go to ./results :  scorecard_*.csv  accuracy.csv  fig_*.png
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import lambdawarm as lw

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

ap = argparse.ArgumentParser()
ap.add_argument("--azure-dir", default="data/azure", help="folder with invocations_per_function_md.anon.dNN.csv")
ap.add_argument("--synthetic", action="store_true", help="use a synthetic trace instead of the Azure files")
ap.add_argument("--days", type=int, default=3, help="days of the trace to use (3 = 2.4 days train, 0.6 day test)")
ap.add_argument("--pick", type=int, default=0, help="which shortlisted Azure function to use (0 = best daily pattern)")
ap.add_argument("--peak-rpm", type=int, default=60, help="rescale the trace so its busiest minutes are ~this many calls/min")
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
cfg = lw.Config(peak_rpm=args.peak_rpm)

if args.synthetic:
    y, label = lw.synthetic_trace(args.days), "synthetic"
else:
    table, series = lw.azure_candidates(args.azure_dir, days=args.days)
    print("Shortlisted Azure functions (use --pick N to change):")
    print(table.drop(columns="key").to_string(), "\n")
    row = table.iloc[args.pick]
    y, label = series[row["key"]], f"Azure function {row['function']}"
print(f"Workload: {label}  | {len(y)} minutes | peak_rpm={cfg.peak_rpm} | service={cfg.service_s}s init={cfg.init_s}s mem={cfg.mem_mb}MB")

prep = lw.prepare(y, cfg)
print(f"Train minutes: {prep.a0}   Test minutes: {len(prep.y) - prep.a0}   Static-PC level (P95 of training concurrency): {prep.static_level}\n")

for name, spike, outage in [("normal", False, False), ("spike", True, False), ("outage", False, True)]:
    ev = lw.evaluate(prep, spike, outage)
    if name == "normal":
        print("=== 1. FORECAST ACCURACY (test window, unseen data) ===")
        print(ev.accuracy.to_string(), "\n")
        ev.accuracy.to_csv(f"{args.out}/accuracy.csv")
        lw.fig_forecast(ev).savefig(f"{args.out}/fig_forecast.png", dpi=150)
        lw.fig_features(ev).savefig(f"{args.out}/fig_features.png", dpi=150)
    cols = ["requests", "cold_starts", "cold_start_%", "avg_ms", "p95_ms", "p99_ms", "p99.9_ms", "sla_violation_%",
            "pc_utilization_%", "utilization_%", "pc_instance_hours", "cost_usd", "energy_wh",
            "extra_usd_per_1000_cold_starts_avoided"]
    print(f"=== 2. SCORECARD - scenario: {name.upper()} ===")
    print(ev.scorecard[cols].to_string())
    events = {k: v.events for k, v in ev.results.items() if v.events}
    if events:
        print("fail-safe events:", events)
    print()
    ev.scorecard[cols].to_csv(f"{args.out}/scorecard_{name}.csv")
    lw.fig_scaling(ev).savefig(f"{args.out}/fig_scaling_{name}.png", dpi=150)
    lw.fig_latency(ev).savefig(f"{args.out}/fig_latency_{name}.png", dpi=150)
    lw.fig_scorecard(ev).savefig(f"{args.out}/fig_scorecard_{name}.png", dpi=150)

print(f"Saved tables and figures to {os.path.abspath(args.out)}")
