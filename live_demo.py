"""
Live validation on REAL AWS Lambda - proves the mechanism and measures your own numbers.

  python live_demo.py --function CloudWorkloadTarget --alias prod --region ap-south-1 --n 3

What it does (about 8-10 minutes, costs well under Rs 1 - provisioned concurrency is deleted at the end, even on Ctrl+C):
  A  Reactive baseline : no pre-warming -> a burst of n requests hits n brand-new environments (cold starts)
  B  Proactive         : PC set to the n instances the forecast asked for -> same burst runs warm; also measures
                         how long AWS needs to make PC READY (put that number in Config.pc_ready_s)
  C  Wrong prediction  : 2n requests arrive but only n were predicted -> n overflow to on-demand (cold starts,
                         but ZERO failures: Lambda scales on its own)
  D  Fail-safe         : controller reacts (1.25 x observed concurrency) -> the same 2n burst now runs warm

--mode ping   uses free 'warmer' invocations instead of real Provisioned Concurrency (no PC charge, not guaranteed).
"""
import argparse
import json
import math
import os
import sys
import time

import boto3
import pandas as pd

import lambdawarm as lw

ap = argparse.ArgumentParser()
ap.add_argument("--function", default="CloudWorkloadTarget")
ap.add_argument("--alias", default="prod")
ap.add_argument("--region", default="ap-south-1")
ap.add_argument("--profile", default=None, help="AWS CLI profile (default: the one saved by `aws configure`)")
ap.add_argument("--n", type=int, default=3, help="instances the forecast asks for (burst sizes are n and 2n)")
ap.add_argument("--mode", choices=["pc", "ping"], default="pc")
ap.add_argument("--no-fresh", action="store_true", help="do not publish a new version before each phase")
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
fn, alias, n = args.function, args.alias, args.n
n_failsafe = math.ceil(1.25 * 2 * n)

client = lw.lambda_client(args.region, args.profile)
sess = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
print("AWS identity :", sess.client("sts").get_caller_identity()["Arn"])
lim = lw.account_limits(client)
print(f"Account concurrency limit in {args.region}: {lim['concurrent_executions']} (unreserved {lim['unreserved']})")
try:
    ver = client.get_alias(FunctionName=fn, Name=alias)["FunctionVersion"]
except Exception as e:  # noqa: BLE001
    sys.exit(f"Cannot find alias '{alias}' on function '{fn}' ({e}).\nCreate the function, publish a version and an alias first (guide step 2).")
print(f"Function     : {fn}:{alias} -> version {ver}\n")

if args.mode == "pc" and lim["unreserved"] is not None and lim["unreserved"] - 10 < n_failsafe:
    sys.exit(
        f"Your account allows {lim['unreserved']} concurrent executions, and Provisioned Concurrency needs the rest to stay "
        f">= 10 free, so at most {max(lim['unreserved'] - 10, 0)} PC instances can be set; this demo needs {n_failsafe}.\n"
        "Options: (1) Service Quotas -> AWS Lambda -> 'Concurrent executions' -> request an increase (e.g. 100 or 1000),\n"
        "         (2) re-run with a smaller --n, or (3) re-run with --mode ping (free warmer pings, not real PC).")

rows, meta = [], {"mode": args.mode, "n": n}


def new_environments(label):
    if args.no_fresh:
        print("   (skipping fresh version: containers from earlier runs may still be warm)")
    else:
        print(f"   publishing a fresh version so {label} starts with no warm environments ...")
        try:
            lw.fresh_version(client, fn, alias, log=lambda s: print(s))
        except Exception as e:  # noqa: BLE001
            print(f"   WARNING: could not publish a fresh version ({e}). Wait ~15 min between runs instead, or fix the IAM policy.")


def run_burst(phase, size):
    df = lw.burst(client, fn, alias, size)
    df.insert(0, "phase", phase)
    rows.append(df)
    s = lw.summarize_burst(df)
    print(f"   -> {s['requests']} requests: {s['cold_starts']} cold starts, {s['on_PC']} served by provisioned concurrency, "
          f"mean {s['mean_ms']} ms, slowest {s['max_ms']} ms, {s['distinct_environments']} distinct environments")
    if df["error"].notna().any():
        print("   !! some invocations returned a function error:", df["error"].dropna().unique())


def provision(count, label):
    if args.mode == "pc":
        print(f"   put_provisioned_concurrency_config(ProvisionedConcurrentExecutions={count}) ...")
        t0 = time.time()
        lw.set_pc(client, fn, alias, count)
        waited = lw.wait_pc_ready(client, fn, alias, count, log=lambda s: print(s))
        print(f"   READY after {waited:.0f} s")
        meta[f"pc_ready_s_{label}"] = round(waited, 1)
    else:
        print(f"   sending {count} parallel warmer pings ...")
        lw.ping_prewarm(client, fn, alias, count)
        time.sleep(2)


try:
    print("A. REACTIVE BASELINE - nothing pre-warmed")
    new_environments("A")
    run_burst("A reactive (no pre-warming)", n)

    print(f"\nB. PROACTIVE - forecast says {n} concurrent executions are coming")
    new_environments("B")
    provision(n, "B")
    run_burst("B proactive (predicted correctly)", n)

    print(f"\nC. WRONG PREDICTION - {2 * n} requests arrive, only {n} warm instances were predicted")
    run_burst("C under-predicted (surprise burst)", 2 * n)

    print(f"\nD. FAIL-SAFE - controller observed {2 * n} concurrent, sets 1.25 x = {n_failsafe} instances")
    provision(n_failsafe, "D")
    run_burst("D after reactive burst mode", 2 * n)
finally:
    print("\nCLEAN-UP: deleting provisioned concurrency so billing stops ...")
    try:
        lw.set_pc(client, fn, alias, 0)
        lw.wait_pc_gone(client, fn, alias)
        print("   done - no provisioned concurrency left on the alias." if lw.pc_status(client, fn, alias) is None
              else "   !! PC still present - delete it in the console (Configuration -> Provisioned concurrency).")
    except Exception as e:  # noqa: BLE001
        print(f"   !! Could not delete PC automatically ({e}). Delete it in the Lambda console NOW.")

if rows:
    allrows = pd.concat(rows, ignore_index=True)
    allrows.to_csv(f"{args.out}/live_run.csv", index=False)
    summary = pd.DataFrame({p: lw.summarize_burst(d) for p, d in allrows.groupby("phase", sort=True)}).T
    print("\n=== LIVE RESULT (put this table in your report) ===")
    print(summary.to_string())
    json.dump(meta, open(f"{args.out}/live_meta.json", "w"), indent=2)
    if "pc_ready_s_B" in meta:
        print(f"\nMeasured time for PC to become READY: {meta['pc_ready_s_B']} s  ->  use  Config(pc_ready_s={int(round(meta['pc_ready_s_B']))})")
    print(f"Saved {args.out}/live_run.csv")
