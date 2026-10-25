"""
LambdaWarm core library
=======================
ML-based workload forecasting + proactive Provisioned-Concurrency (PC) scaling for AWS Lambda.

Used by:  run_experiments.py (report numbers) | live_demo.py (real AWS test) | app.py (dashboard)

Sections
  1. Config            - every assumption in one place (state these in your report)
  2. Data              - Azure Functions 2019 trace loader (+ synthetic fallback)
  3. Features + model  - 5-minute-ahead forecast of invocations and peak concurrency (LightGBM, mean + 90th percentile)
  4. Sizing            - forecast of peak concurrency -> number of warm instances
  5. Policies          - on-demand, static PC, reactive PC, ML, ML + fail-safes
  6. Simulator         - trace-driven Lambda model -> latency / cost / energy / utilisation
  7. Evaluate + plots
  8. AWS controller    - boto3 calls that apply the decision to a real Lambda alias
"""
from __future__ import annotations

import base64
import gc
import glob
import heapq
import json
import math
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import poisson

# =====================================================================================
# 1. CONFIG
# =====================================================================================


@dataclass(frozen=True)
class Config:
    # --- workload / function model --------------------------------------------------
    peak_rpm: int = 600          # trace is rescaled so its 99.5th-percentile minute = this many calls/min
    service_s: float = 0.2       # mean execution time per request (s)
    init_s: float = 1.8          # cold-start (INIT) time (s); equals INIT_DELAY_S in lambda_function.py
    mem_mb: int = 512            # Lambda memory size
    sla_ms: float = 1000.0       # latency SLA (a request slower than this is a violation)
    burst_frac: float = 0.3      # share of each minute's calls that land inside one random 5-second window
    # --- platform model -------------------------------------------------------------
    pc_ready_s: int = 90         # seconds before a PC *increase* is READY (measure yours with live_demo.py)
    keepalive_s: int = 600       # idle on-demand environments are reclaimed after this long (AWS does not publish it)
    # --- ML + controller ------------------------------------------------------------
    horizon: int = 5             # forecast the busiest minute of the next 5 minutes
    train_frac: float = 0.8      # chronological split
    quantile: float = 0.9        # forecast the 90th percentile of future peak concurrency (a safety margin)
    headroom: float = 1.0        # PC = ceil(headroom x forecast)
    min_conc: float = 0.5        # forecast below this many concurrent executions -> keep PC at 0 (a cold start is cheaper)
    down_hold: int = 3           # minutes a lower target must persist before scaling down (anti-flapping)
    burst_ratio: float = 1.5     # fail-safe watchdog: observed peak > ratio x forecast (+1) AND cold starts -> under-prediction
    burst_mult: float = 1.25     # fail-safe: buffer applied to the observed concurrency after a surprise
    burst_hold: int = 6          # fail-safe: minutes the burst floor is held
    shrink: float = 0.5          # fail-safe: factor applied after a false alarm
    shrink_hold: int = 10        # fail-safe: minutes the shrink is held
    # --- pricing, USD, x86 list prices (US East; Mumbai is a little higher - edit if you want exact) ---
    price_request: float = 0.20 / 1_000_000
    price_gbs_ondemand: float = 0.0000166667
    price_gbs_pc_duration: float = 0.0000097222
    price_gbs_pc_provisioned: float = 0.0000041667
    bill_init: bool = True       # since 1 Aug 2025 Lambda bills the INIT phase of on-demand cold starts
    # --- energy model (ASSUMPTIONS - cite/justify in the report) -----------------------
    p_active_w_per_gb: float = 4.0   # executing or initialising
    p_idle_w_per_gb: float = 1.2     # warm but idle (provisioned or keep-alive)  -> 30 % of active
    seed: int = 42

    @property
    def mem_gb(self) -> float:
        return self.mem_mb / 1024.0


# =====================================================================================
# 2. DATA
# =====================================================================================

MINUTE_COLS = [str(i) for i in range(1, 1441)]


def synthetic_trace(days: int = 3, seed: int = 7) -> np.ndarray:
    """Office-hours serverless workload: quiet nights, daytime ramps, random flash bursts and idle gaps."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    mod = np.arange(n) % 1440
    base = 25 + 80 * np.exp(-((mod - 13 * 60) / 190.0) ** 2) + 45 * np.exp(-((mod - 20 * 60) / 100.0) ** 2)
    base = np.where(mod < 6 * 60, 0.4, base)                       # nights are nearly idle
    y = rng.poisson(base).astype(float)
    for _ in range(days * 7):                                      # flash bursts (3-10 min, x3-6)
        s = int(rng.integers(0, n - 12)); d = int(rng.integers(3, 11))
        y[s:s + d] = np.maximum(y[s:s + d], 8) * rng.integers(3, 7)
    for _ in range(days * 5):                                      # idle gaps (6-25 min)
        s = int(rng.integers(0, n - 30)); d = int(rng.integers(6, 26))
        y[s:s + d] = 0
    return y


def scale_trace(y: np.ndarray, cfg: Config) -> np.ndarray:
    """Rescale so the 99.5th-percentile minute = cfg.peak_rpm calls/min (standard when replaying production traces on a test bed)."""
    y = np.asarray(y, dtype=float)
    nz = y[y > 0]
    ref = np.percentile(nz, 99.5) if len(nz) else 1.0
    k = cfg.peak_rpm / max(ref, 1.0)
    return np.minimum(np.rint(y * k), 5 * cfg.peak_rpm).astype(int)


def _azure_files(folder: str, days: int) -> list[str]:
    files = sorted(glob.glob(os.path.join(folder, "invocations_per_function_md.anon.d*.csv")))[:days]
    if len(files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 files named invocations_per_function_md.anon.dNN.csv in '{folder}' "
            f"(found {len(files)}). Extract the Azure Functions 2019 archive there first.")
    return files


def azure_candidates(folder: str, days: int = 3, top: int = 12) -> tuple[pd.DataFrame, dict]:
    """
    Scan the Azure Functions 2019 invocation files and shortlist functions that make a good Lambda test case:
    HTTP trigger, busy on *every* day, with idle gaps, bounded spikes and a repeating daily pattern.
    Streams one day at a time (~300 MB RAM) and caches the shortlist next to the CSVs, so only the first run is slow.
    Returns (table of top candidates, {function_key: per-minute series}).
    """
    cache = os.path.join(folder, f"_lambdawarm_cache_d{days}.pkl")
    if os.path.exists(cache):
        obj = pd.read_pickle(cache)
        return obj["table"].head(top).copy(), obj["series"]

    files = _azure_files(folder, days)
    nd = len(files)
    dtypes = {c: np.int32 for c in MINUTE_COLS}
    parts, trig = [], {}
    for i, f in enumerate(files):                                    # pass 1: cheap statistics per function
        df = pd.read_csv(f, dtype=dtypes)
        keys = (df["HashOwner"] + "|" + df["HashApp"] + "|" + df["HashFunction"]).to_numpy()
        m = df[MINUTE_COLS].to_numpy()
        parts.append(pd.DataFrame({"day": i, "total": m.sum(1, dtype=np.int64),
                                   "zeros": (m == 0).sum(1), "peak": m.max(1)}, index=keys))
        for k, t in zip(keys, df["Trigger"].to_numpy()):
            trig.setdefault(k, t)
        del df, m
        gc.collect()
    allp = pd.concat(parts)
    per_day = allp.pivot_table(index=allp.index, columns="day", values="total", aggfunc="sum")
    g = allp.groupby(level=0).agg(total=("total", "sum"), zeros=("zeros", "sum"), peak=("peak", "max"), days=("total", "size"))
    g = g.join(per_day.add_prefix("d"))
    g["trigger"] = pd.Series(trig)
    g = g[g["days"] == nd].copy()
    minutes = 1440 * nd
    g["zero_frac"] = g["zeros"] / minutes
    g["mean_rpm"] = g["total"] / minutes
    g["peak_to_mean"] = g["peak"] / g["mean_rpm"].clip(lower=1e-9)
    g["min_day_share"] = g[[f"d{i}" for i in range(nd)]].min(axis=1) / g["total"]

    def shortlist(trigger_http: bool, min_total: int) -> pd.DataFrame:
        c = g[(g["total"] >= min_total) & (g["min_day_share"] >= 0.15) & g["zero_frac"].between(0.03, 0.6)
              & (g["peak_to_mean"] <= 60)]
        if trigger_http:
            c = c[c["trigger"] == "http"]
        return c.drop_duplicates(subset=["total", "zeros", "peak"])

    cand = shortlist(True, 100_000)
    if len(cand) < 5:
        cand = shortlist(False, 50_000)
    if len(cand) == 0:
        raise RuntimeError("No suitable function found - try days=2 or add more day files.")
    cand = cand.sort_values("total", ascending=False).head(150)

    wanted = set(cand.index)                                         # pass 2: pull the full series of the shortlist
    series: dict[str, list[np.ndarray]] = {k: [] for k in wanted}
    for f in files:
        df = pd.read_csv(f, dtype=dtypes)
        keys = (df["HashOwner"] + "|" + df["HashApp"] + "|" + df["HashFunction"])
        mask = keys.isin(wanted).to_numpy()
        for k, row in zip(keys[mask].to_numpy(), df.loc[mask, MINUTE_COLS].to_numpy()):
            series[k].append(row.astype(float))
        del df
        gc.collect()
    full = {k: np.concatenate(v) for k, v in series.items() if len(v) == nd}

    def season(y: np.ndarray) -> float:                              # does day N look like day N-1 (hourly totals)?
        h = y.reshape(nd * 24, 60).sum(1).reshape(nd, 24)
        cs = [np.corrcoef(h[i], h[i + 1])[0, 1] for i in range(nd - 1) if h[i].std() > 0 and h[i + 1].std() > 0]
        return float(np.mean(cs)) if cs else 0.0

    cand = cand.loc[[k for k in cand.index if k in full]].copy()
    cand["daily_pattern"] = [season(full[k]) for k in cand.index]
    cand["score"] = cand["daily_pattern"] * np.minimum(cand["zero_frac"] / 0.2, 1.0)
    cand = cand.sort_values("score", ascending=False)
    table = pd.DataFrame({
        "function": [k.split("|")[-1][:8] for k in cand.index],
        "key": cand.index,
        "calls_total": cand["total"].astype(int).to_numpy(),
        "idle_minutes_%": (100 * cand["zero_frac"]).round(1).to_numpy(),
        "avg_calls_per_min": cand["mean_rpm"].round(1).to_numpy(),
        "peak_calls_per_min": cand["peak"].astype(int).to_numpy(),
        "daily_pattern": cand["daily_pattern"].round(2).to_numpy(),
    })
    pd.to_pickle({"table": table, "series": {k: full[k] for k in table["key"]}}, cache)
    return table.head(top).copy(), {k: full[k] for k in table["key"]}


# =====================================================================================
# 3. FEATURES + MODEL
# =====================================================================================

FEATURES = ["calls_now", "lag_1", "lag_2", "lag_3", "velocity", "idle_streak", "roll_mean_5", "roll_mean_15",
            "roll_mean_60", "roll_max_15", "roll_std_15", "conc_now", "conc_roll_max_5", "conc_roll_max_15",
            "conc_roll_mean_15", "yday_now", "yday_next_calls", "yday_next_conc", "tod_sin", "tod_cos"]


def make_features(y: np.ndarray, conc: np.ndarray, horizon: int = 5) -> pd.DataFrame:
    """One row per minute t, using only information available at the END of minute t (no leakage).
      y     = invocations per minute            conc = peak concurrent executions per minute (CloudWatch: ConcurrentExecutions, Max)
      target_calls = busiest minute among t+1..t+H        target_conc = highest concurrency among t+1..t+H"""
    s = pd.Series(np.asarray(y, dtype=float))
    c = pd.Series(np.asarray(conc, dtype=float))
    n = len(s)
    df = pd.DataFrame({"calls_now": s})
    for k in (1, 2, 3):
        df[f"lag_{k}"] = s.shift(k)
    df["velocity"] = s.diff()
    grp = (s > 0).astype(int).cumsum()
    df["idle_streak"] = (s == 0).astype(int).groupby(grp).cumsum()          # consecutive idle minutes up to t
    for w in (5, 15, 60):
        df[f"roll_mean_{w}"] = s.rolling(w, min_periods=1).mean()
    df["roll_max_15"] = s.rolling(15, min_periods=1).max()
    df["roll_std_15"] = s.rolling(15, min_periods=2).std().fillna(0.0)
    df["conc_now"] = c
    df["conc_roll_max_5"] = c.rolling(5, min_periods=1).max()
    df["conc_roll_max_15"] = c.rolling(15, min_periods=1).max()
    df["conc_roll_mean_15"] = c.rolling(15, min_periods=1).mean()
    t_calls = s.rolling(horizon).max().shift(-horizon)                      # max(y[t+1..t+H])
    t_conc = c.rolling(horizon).max().shift(-horizon)
    df["yday_now"] = s.shift(1440)                                           # same minute yesterday
    df["yday_next_calls"] = t_calls.shift(1440)                              # yesterday's busiest minute of the same window
    df["yday_next_conc"] = t_conc.shift(1440)
    mod = np.arange(n) % 1440
    df["tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * mod / 1440)
    df["naive_calls"] = s.rolling(5, min_periods=1).max()                    # baselines (not model inputs)
    df["naive_conc"] = df["conc_roll_max_5"]
    df["target_calls"] = t_calls
    df["target_conc"] = t_conc
    return df


_LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


@dataclass
class Prepared:
    cfg: Config
    y: np.ndarray                 # scaled trace (calls/min)
    a0: int                       # first test minute
    calls_model: lgb.LGBMRegressor    # forecasts invocations (stage 1)
    conc_model: lgb.LGBMRegressor     # forecasts peak concurrency, mean
    conc_q: lgb.LGBMRegressor         # forecasts peak concurrency, upper quantile  -> drives PC sizing (stage 2)
    static_level: int             # PC level for the static baseline (sized on the training period only)


def build_dataset(y: np.ndarray, cfg: Config):
    """invocations -> request timestamps -> demand concurrency -> feature table."""
    arr = make_arrivals(y, cfg)
    conc = demand_concurrency(arr)
    return arr, conc, make_features(y, conc, cfg.horizon)


def prepare(y_raw: np.ndarray, cfg: Config) -> Prepared:
    y = scale_trace(y_raw, cfg)
    arr, conc, df = build_dataset(y, cfg)
    valid = np.where(df["target_conc"].notna().to_numpy())[0]
    a0 = int(valid[int(len(valid) * cfg.train_frac)])
    train = df.iloc[valid[valid < a0]]
    X = train[FEATURES]
    calls_model = lgb.LGBMRegressor(objective="regression", random_state=cfg.seed, **_LGB).fit(X, train["target_calls"])
    conc_model = lgb.LGBMRegressor(objective="regression", random_state=cfg.seed, **_LGB).fit(X, train["target_conc"])
    conc_q = lgb.LGBMRegressor(objective="quantile", alpha=cfg.quantile, random_state=cfg.seed, **_LGB).fit(X, train["target_conc"])
    static_level = int(math.ceil(np.percentile(conc[:a0], 95)))
    return Prepared(cfg, y, a0, calls_model, conc_model, conc_q, static_level)


def inject(y: np.ndarray, a0: int, cfg: Config, spike: bool = False, outage: bool = False,
           spike_at: float = 0.30, ramp_len: int = 5, hold_len: int = 8, spike_mult: float = 3.0, outage_len: int = 45):
    """Misprediction scenarios applied to the TEST part of the trace only. Returns (trace, windows).
    spike  = a flash crowd that ramps up to spike_mult x the busiest minute in the TRAINING history and holds.
             The model has never seen such load, so it UNDER-predicts (tree models cannot extrapolate).
    outage = traffic vanishes for outage_len minutes during the busiest stretch, so history-based forecasts are
             too high -> OVER-prediction (false alarm)."""
    y = y.copy()
    n_test = len(y) - a0
    win: dict = {}
    if spike:
        s = int(spike_at * n_test)
        top = spike_mult * y[:a0].max()
        shape = np.r_[np.linspace(top / ramp_len, top, ramp_len), np.full(hold_len, top)]
        y[a0 + s:a0 + s + len(shape)] += shape.astype(int)
        win["spike"] = (s, s + len(shape))
    if outage:
        roll = pd.Series(y[a0:]).rolling(outage_len).sum().to_numpy()
        end = int(np.nanargmax(roll))
        y[a0 + end - outage_len + 1:a0 + end + 1] = 0
        win["outage"] = (end - outage_len + 1, end + 1)
    return y, win


# =====================================================================================
# 4. SIZING   (forecast of peak concurrency -> number of warm instances)
# =====================================================================================


def size_from_forecast(conc_forecast: float, cfg: Config) -> int:
    """PC = ceil(headroom x forecast). Below min_conc the forecast is not worth a paid warm instance -> 0 (scale to zero)."""
    if conc_forecast < cfg.min_conc:
        return 0
    return int(math.ceil(cfg.headroom * conc_forecast - 1e-9))


# =====================================================================================
# 5. POLICIES   (each returns the PC level to REQUEST at the start of minute m)
# =====================================================================================


@dataclass
class Telemetry:
    """What a real controller can read from CloudWatch: complete minutes only (index < m)."""
    calls: np.ndarray
    peak_conc: np.ndarray        # ConcurrentExecutions (max per minute)
    cold: np.ndarray             # requests that paid a cold start (spillover / no PC)


class Policy:
    name = "policy"

    def reset(self, T: int) -> None: ...

    def decide(self, m: int, tel: Telemetry) -> int:
        raise NotImplementedError


class OnDemand(Policy):
    name = "On-demand only (no PC)"

    def decide(self, m, tel):
        return 0


class StaticPC(Policy):
    def __init__(self, level: int):
        self.level = level
        self.name = f"Static PC = {level}"

    def decide(self, m, tel):
        return self.level


class ReactivePC(Policy):
    """Target tracking on PC utilisation, like Application Auto Scaling: react only AFTER load was observed."""
    name = "Reactive target-tracking PC"

    def __init__(self, target: float = 0.7, out_after: int = 2, in_after: int = 15):
        self.target, self.out_after, self.in_after = target, out_after, in_after

    def reset(self, T):
        self.cur = self.out = self.inn = 0

    def decide(self, m, tel):
        if m == 0:
            return self.cur
        peak = int(tel.peak_conc[m - 1])
        util = (peak / self.cur) if self.cur > 0 else (math.inf if peak > 0 else 0.0)
        want = math.ceil(peak / self.target)
        if util > self.target:
            self.out, self.inn = self.out + 1, 0
            if self.out >= self.out_after:
                self.cur, self.out = max(self.cur, want), 0
        elif util < 0.9 * self.target:
            self.inn, self.out = self.inn + 1, 0
            if self.inn >= self.in_after:
                self.cur, self.inn = want, 0
        else:
            self.out = self.inn = 0
        return self.cur


class MLPolicy(Policy):
    """Forecast -> size -> asymmetric cooldown, optionally wrapped in the two misprediction fail-safes."""

    def __init__(self, fc_conc_q: np.ndarray, cfg: Config, failsafe: bool):
        self.fc, self.cfg, self.failsafe = fc_conc_q, cfg, failsafe
        self.name = "LambdaWarm: ML + fail-safes" if failsafe else "ML forecast only (no fail-safe)"
        self.events = {"burst_mode": 0, "false_alarm": 0}

    def reset(self, T):
        self.cur = self.low = 0
        self.burst_until = self.shrink_until = -1
        self.floor = 0
        self.events = {"burst_mode": 0, "false_alarm": 0}

    def decide(self, m, tel):
        c = self.cfg
        target = size_from_forecast(float(self.fc[m]), c)
        if self.failsafe:
            # (a) over-prediction: the window forecast 'horizon' minutes ago has now ended -> compare with reality
            if m >= c.horizon:
                f_old = float(self.fc[m - c.horizon])
                realised = float(tel.peak_conc[m - c.horizon:m].max())
                if f_old >= 2 and realised < 0.25 * f_old and m >= self.shrink_until:
                    self.shrink_until = m + c.shrink_hold
                    self.events["false_alarm"] += 1
            if m < self.shrink_until:
                target = math.ceil(c.shrink * target)
            # (b) under-prediction: cold starts happened last minute -> reactive burst mode
            if m >= 1 and tel.cold[m - 1] > 0 and tel.peak_conc[m - 1] > c.burst_ratio * float(self.fc[m - 1]) + 1:
                if m >= self.burst_until:
                    self.events["burst_mode"] += 1
                new_floor = math.ceil(c.burst_mult * int(tel.peak_conc[m - 1]))
                self.floor = max(self.floor if m < self.burst_until else 0, new_floor)
                self.burst_until = m + c.burst_hold
            if m < self.burst_until:
                target = max(target, self.floor)
        # asymmetric cooldown: scale up immediately, scale down only after `down_hold` consecutive lower targets
        if target >= self.cur:
            self.cur, self.low = target, 0
        else:
            self.low += 1
            if self.low >= c.down_hold:
                self.cur, self.low = target, 0
        return self.cur


# =====================================================================================
# 6. SIMULATOR
# =====================================================================================


@dataclass
class Arrivals:
    times: np.ndarray            # seconds from start of the test window (sorted)
    svc: np.ndarray              # service time of each request (s)
    ptr: np.ndarray              # ptr[m]:ptr[m+1] = requests of minute m
    calls: np.ndarray            # calls per minute


def make_arrivals(calls: np.ndarray, cfg: Config) -> Arrivals:
    """Turn per-minute counts into request timestamps. The SAME arrivals are replayed through every policy.
    Each minute has its own random stream, so injecting a scenario does not reshuffle the rest of the trace."""
    calls = np.asarray(calls, dtype=int)
    T = len(calls)
    ptr = np.zeros(T + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(calls)
    N = int(ptr[-1])
    times, svc = np.empty(N), np.empty(N)
    sigma = 0.25
    mu = math.log(cfg.service_s) - sigma ** 2 / 2                              # lognormal with mean = service_s
    for m in range(T):
        n = int(calls[m])
        if n == 0:
            continue
        rng = np.random.default_rng((cfg.seed, m))
        nb = int(rng.binomial(n, cfg.burst_frac))                              # micro-burst: nb calls inside one 5-s window
        t = np.minimum(np.concatenate([rng.uniform(0, 60, n - nb), rng.uniform(0, 55) + rng.uniform(0, 5, nb)]), 59.999)
        t.sort()
        times[ptr[m]:ptr[m + 1]] = m * 60 + t
        svc[ptr[m]:ptr[m + 1]] = rng.lognormal(mu, sigma, n)
    return Arrivals(times, svc, ptr, calls)


def slice_arrivals(arr: Arrivals, a0: int) -> Arrivals:
    i0 = int(arr.ptr[a0])
    return Arrivals(arr.times[i0:] - a0 * 60.0, arr.svc[i0:], arr.ptr[a0:] - i0, arr.calls[a0:])


def demand_concurrency(arr: Arrivals) -> np.ndarray:
    """Peak number of requests in flight per minute if every request were served instantly (M/G/inf).
    This is what CloudWatch's ConcurrentExecutions (Maximum, 1-min) would show for a warm function."""
    T = len(arr.calls)
    out = np.zeros(T, dtype=int)
    ends: list[float] = []
    push, pop = heapq.heappush, heapq.heappop
    for m in range(T):
        peak = 0
        for i in range(int(arr.ptr[m]), int(arr.ptr[m + 1])):
            a = arr.times[i]
            while ends and ends[0] <= a:
                pop(ends)
            push(ends, a + arr.svc[i])
            if len(ends) > peak:
                peak = len(ends)
        out[m] = peak
    return out


@dataclass
class SimResult:
    name: str
    t: np.ndarray
    lat_ms: np.ndarray
    cold: np.ndarray
    req: np.ndarray              # requested PC per minute
    cold_pm: np.ndarray
    calls_pm: np.ndarray
    peak_conc: np.ndarray
    metrics: dict
    events: dict = field(default_factory=dict)


def run_policy(policy: Policy, arr: Arrivals, cfg: Config) -> SimResult:
    T = len(arr.calls)
    policy.reset(T)
    tel = Telemetry(np.zeros(T), np.zeros(T, dtype=int), np.zeros(T, dtype=int))
    req = np.zeros(T, dtype=int)
    av = np.zeros(T * 60, dtype=np.int32)                   # PC that is READY, per second
    D = int(cfg.pc_ready_s)
    n = len(arr.times)
    lat = np.empty(n)
    is_cold = np.zeros(n, dtype=bool)
    pc_busy: list[float] = []                               # finish times of requests running on provisioned instances
    od_busy: list[float] = []                               # ... on on-demand environments
    od_idle: deque = deque()                                # last-used times of idle on-demand environments (ascending)
    busy_pc = busy_od = init_total = idle_od = 0.0
    keep, init = float(cfg.keepalive_s), float(cfg.init_s)
    times, svc = arr.times, arr.svc
    prev_av = 0
    push, pop = heapq.heappush, heapq.heappop

    for m in range(T):
        r = int(policy.decide(m, tel))
        req[m] = r
        for s in range(m * 60, m * 60 + 60):               # increases need D seconds to become READY; decreases are immediate
            so = s - D
            r_old = req[so // 60] if so >= 0 else req[0]
            prev_av = min(r, max(prev_av, int(r_old)))
            av[s] = prev_av
        peak = cold_m = 0
        for i in range(int(arr.ptr[m]), int(arr.ptr[m + 1])):
            a, sv = times[i], svc[i]
            while pc_busy and pc_busy[0] <= a:
                pop(pc_busy)
            while od_busy and od_busy[0] <= a:
                od_idle.append(pop(od_busy))
            while od_idle and a - od_idle[0] > keep:
                od_idle.popleft()
                idle_od += keep
            if len(pc_busy) < av[int(a)]:                  # a free provisioned instance -> warm, served by PC
                push(pc_busy, a + sv); lat[i] = sv; busy_pc += sv
            elif od_idle:                                  # reuse a kept-alive on-demand environment -> warm
                idle_od += a - od_idle.pop()
                push(od_busy, a + sv); lat[i] = sv; busy_od += sv
            else:                                          # nothing warm -> COLD START
                push(od_busy, a + init + sv); lat[i] = init + sv; is_cold[i] = True
                busy_od += sv; init_total += init; cold_m += 1
            c = len(pc_busy) + len(od_busy)
            if c > peak:
                peak = c
        tel.calls[m] = arr.ptr[m + 1] - arr.ptr[m]
        tel.peak_conc[m] = peak
        tel.cold[m] = cold_m

    end = T * 60.0
    while od_busy:
        idle_od += min(keep, max(end - pop(od_busy), 0.0))
    for last in od_idle:
        idle_od += min(keep, max(end - last, 0.0))

    metrics = summarize(lat, is_cold, req, busy_pc, busy_od, init_total, idle_od, cfg)
    return SimResult(policy.name, times, lat * 1000, is_cold, req, tel.cold.copy(), tel.calls.copy(),
                     tel.peak_conc.copy(), metrics, dict(getattr(policy, "events", {})))


def summarize(lat, is_cold, req, busy_pc, busy_od, init_total, idle_od, cfg: Config) -> dict:
    n = len(lat)
    ms = lat * 1000
    prov_s = float(req.sum()) * 60.0                        # PC is billed from request to deletion
    idle_pc = max(prov_s - busy_pc, 0.0)
    gb = cfg.mem_gb
    busy_total = busy_pc + busy_od
    cost = (n * cfg.price_request
            + (busy_od + (init_total if cfg.bill_init else 0.0)) * gb * cfg.price_gbs_ondemand
            + busy_pc * gb * cfg.price_gbs_pc_duration
            + prov_s * gb * cfg.price_gbs_pc_provisioned)
    energy_wh = (((busy_total + init_total) * cfg.p_active_w_per_gb + (idle_pc + idle_od) * cfg.p_idle_w_per_gb) * gb) / 3600.0
    warm_time = busy_total + init_total + idle_pc + idle_od
    pct = (lambda q: float(np.percentile(ms, q))) if n else (lambda q: float("nan"))
    return {
        "requests": n,
        "cold_starts": int(is_cold.sum()),
        "cold_start_%": round(100 * is_cold.mean(), 2) if n else 0.0,
        "avg_ms": round(float(ms.mean()), 1) if n else 0.0,
        "p50_ms": round(pct(50), 1), "p95_ms": round(pct(95), 1), "p99_ms": round(pct(99), 1), "p99.9_ms": round(pct(99.9), 1),
        "sla_violation_%": round(100 * float((ms > cfg.sla_ms).mean()), 2) if n else 0.0,
        "utilization_%": round(100 * busy_total / warm_time, 1) if warm_time > 0 else 0.0,
        "pc_utilization_%": round(100 * busy_pc / prov_s, 1) if prov_s > 0 else float("nan"),
        "pc_instance_hours": round(prov_s / 3600, 2),
        "cost_usd": round(cost, 5),
        "energy_wh": round(energy_wh, 2),
    }


# =====================================================================================
# 7. EVALUATE + PLOTS
# =====================================================================================


@dataclass
class Evaluation:
    prep: Prepared
    y: np.ndarray                # trace after scenario injection
    conc: np.ndarray             # demand concurrency per minute after injection
    p_calls: np.ndarray          # forecast of busiest minute (calls/min)
    p_conc: np.ndarray           # forecast of peak concurrency, mean
    p_conc_q: np.ndarray         # forecast of peak concurrency, upper quantile (drives PC)
    feats: pd.DataFrame
    accuracy: pd.DataFrame
    scorecard: pd.DataFrame
    results: dict
    windows: dict


def _errors(y, p) -> dict:
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    e = p - y
    nz = y > 0
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "WAPE_%": float(100 * np.abs(e).sum() / max(y.sum(), 1e-9)),
            "MAPE_%": float(100 * np.mean(np.abs(e[nz]) / y[nz])) if nz.any() else float("nan")}


def accuracy_table(feats: pd.DataFrame, p_calls, p_conc, p_conc_q, a0: int, q: float) -> pd.DataFrame:
    tc, tn = feats["target_calls"].to_numpy()[a0:], feats["target_conc"].to_numpy()[a0:]
    ok = ~np.isnan(tn)
    rows = {
        ("Peak concurrency", "LightGBM (mean)"): _errors(tn, p_conc[a0:]),
        ("Peak concurrency", f"LightGBM ({int(q * 100)}th pct, drives PC)"): {
            **_errors(tn, p_conc_q[a0:]), "coverage_%": float(100 * np.mean(tn[ok] <= p_conc_q[a0:][ok]))},
        ("Peak concurrency", "Baseline: max of last 5 min"): _errors(tn, feats["naive_conc"].to_numpy()[a0:]),
        ("Peak concurrency", "Baseline: same window yesterday"): _errors(tn, feats["yday_next_conc"].to_numpy()[a0:]),
        ("Invocations/min", "LightGBM (mean)"): _errors(tc, p_calls[a0:]),
        ("Invocations/min", "Baseline: max of last 5 min"): _errors(tc, feats["naive_calls"].to_numpy()[a0:]),
        ("Invocations/min", "Baseline: same window yesterday"): _errors(tc, feats["yday_next_calls"].to_numpy()[a0:]),
    }
    out = pd.DataFrame(rows).T.round(2)
    out.index.names = ["Forecast target", "Model"]
    return out


def evaluate(prep: Prepared, spike: bool = False, outage: bool = False, cfg: Config | None = None) -> Evaluation:
    cfg = cfg or prep.cfg
    y, windows = inject(prep.y, prep.a0, cfg, spike, outage)
    arr_all, conc, feats = build_dataset(y, cfg)
    X = feats[FEATURES]
    p_calls = np.maximum(prep.calls_model.predict(X), 0)
    p_conc = np.maximum(prep.conc_model.predict(X), 0)
    p_conc_q = np.maximum(prep.conc_q.predict(X), 0)
    a0 = prep.a0
    T = len(y) - a0
    fc = np.r_[p_conc_q[a0 - 1], p_conc_q[a0:a0 + T - 1]]       # forecast available at the START of each test minute
    arr = slice_arrivals(arr_all, a0)
    policies = [OnDemand(), StaticPC(prep.static_level), ReactivePC(),
                MLPolicy(fc, cfg, failsafe=False), MLPolicy(fc, cfg, failsafe=True)]
    results = {p.name: run_policy(p, arr, cfg) for p in policies}
    sc = pd.DataFrame({k: v.metrics for k, v in results.items()}).T
    base = sc.iloc[0]                                           # on-demand baseline
    avoided = base["cold_starts"] - sc["cold_starts"]
    sc["extra_usd_per_1000_cold_starts_avoided"] = np.where(avoided > 0, 1000 * (sc["cost_usd"] - base["cost_usd"]) / avoided.where(avoided > 0, 1), np.nan).round(3)
    acc = accuracy_table(feats, p_calls, p_conc, p_conc_q, a0, cfg.quantile)
    return Evaluation(prep, y, conc, p_calls, p_conc, p_conc_q, feats, acc, sc, results, windows)


COLORS = {"On-demand only (no PC)": "#9aa0a6", "Reactive target-tracking PC": "#e8a33d",
          "ML forecast only (no fail-safe)": "#6c8ebf", "LambdaWarm: ML + fail-safes": "#1b9e77"}


def _color(name: str) -> str:
    return "#7f5aa8" if name.startswith("Static") else COLORS.get(name, "#333333")


def _shade(ax, ev: Evaluation):
    if "spike" in ev.windows:
        ax.axvspan(*ev.windows["spike"], color="#d95f02", alpha=0.18, lw=0)
    if "outage" in ev.windows:
        ax.axvspan(*ev.windows["outage"], color="#7570b3", alpha=0.18, lw=0)


def fig_forecast(ev: Evaluation):
    import matplotlib.pyplot as plt
    a0 = ev.prep.a0
    t = ev.feats["target_conc"].to_numpy()[a0:]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, color="#222", lw=1, label="Actual: peak concurrency in the next 5 min")
    ax.plot(ev.p_conc[a0:], color="#1b9e77", lw=1.2, label="LightGBM forecast")
    ax.plot(ev.p_conc_q[a0:], color="#d95f02", lw=0.9, ls="--", label=f"{int(ev.prep.cfg.quantile * 100)}th percentile (sizes the warm pool)")
    _shade(ax, ev)
    ax.set_xlabel("minutes into test window"); ax.set_ylabel("concurrent executions"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("5-minute-ahead concurrency forecast on unseen data")
    fig.tight_layout()
    return fig


def fig_scaling(ev: Evaluation, names: list[str] | None = None):
    import matplotlib.pyplot as plt
    a0 = ev.prep.a0
    names = names or [n for n in ev.results if not n.startswith("Static")]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True)
    ax[0].plot(ev.y[a0:], color="#222", lw=0.9)
    ax[0].set_ylabel("calls / min"); ax[0].set_title("Incoming traffic")
    for n in names:
        r = ev.results[n]
        if n != "On-demand only (no PC)":
            ax[1].step(range(len(r.req)), r.req, where="post", color=_color(n), lw=1.4,
                       ls="--" if "no fail-safe" in n else "-", label=n)
    ax[1].set_ylabel("PC instances requested"); ax[1].set_title("Warm instances allocated"); ax[1].legend(fontsize=8, loc="upper right")
    for n in names:
        r = ev.results[n]
        ax[2].plot(r.cold_pm, color=_color(n), lw=1.0, label=f"{n} ({r.metrics['cold_starts']} cold starts)")
    ax[2].set_ylabel("cold starts / min"); ax[2].set_xlabel("minutes into test window"); ax[2].legend(fontsize=8, loc="upper right")
    ax[2].set_title("Cold starts")
    for a in ax:
        _shade(a, ev)
    fig.tight_layout()
    return fig


def fig_latency(ev: Evaluation, name: str = "LambdaWarm: ML + fail-safes", max_points: int = 6000):
    import matplotlib.pyplot as plt
    r = ev.results[name]
    idx = np.arange(len(r.t))
    if len(idx) > max_points:
        idx = np.random.default_rng(0).choice(idx, max_points, replace=False)
    c = r.cold[idx]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.scatter(r.t[idx][~c] / 60, r.lat_ms[idx][~c], s=3, c="#1b9e77", label="warm", alpha=0.5)
    ax.scatter(r.t[idx][c] / 60, r.lat_ms[idx][c], s=7, c="#d62728", label="cold start", alpha=0.8)
    ax.axhline(ev.prep.cfg.sla_ms, color="#555", ls=":", lw=1); ax.text(0, ev.prep.cfg.sla_ms * 1.05, "SLA", fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("minutes into test window"); ax.set_ylabel("latency (ms, log)")
    ax.set_title(f"Request latency - {name}"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_scorecard(ev: Evaluation):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    sc = ev.scorecard
    cols = [("cold_starts", "Cold starts (count)"), ("p99.9_ms", "P99.9 latency (ms)"), ("cost_usd", "Cost (USD, test window)"),
            ("energy_wh", "Energy (Wh, test window)"), ("pc_utilization_%", "Provisioned-capacity utilisation (%)"),
            ("sla_violation_%", f"SLA violations (> {ev.prep.cfg.sla_ms:.0f} ms, %)")]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6))
    for ax, (c, title) in zip(axs.ravel(), cols):
        vals = sc[c].astype(float).fillna(0)
        ax.bar(range(len(sc)), vals, color=[_color(n) for n in sc.index])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=10); ax.set_xticks([])
    fig.legend(handles=[Patch(color=_color(n), label=n) for n in sc.index], loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def fig_features(ev: Evaluation):
    import matplotlib.pyplot as plt
    imp = pd.Series(ev.prep.conc_model.booster_.feature_importance("gain"), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp.index, 100 * imp / imp.sum(), color="#1b9e77")
    ax.set_xlabel("share of model gain (%)"); ax.set_title("Which input features matter")
    fig.tight_layout()
    return fig


# =====================================================================================
# 8. AWS CONTROLLER   (applies the decision to a real Lambda alias with boto3)
# =====================================================================================

_INIT_RE = re.compile(r"Init Duration: ([\d.]+) ms")


def lambda_client(region: str = "ap-south-1", profile: str | None = None):
    """Uses the credentials saved by `aws configure` (never paste keys into code or the dashboard)."""
    import boto3
    from botocore.config import Config as BotoConfig
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("lambda", region_name=region,
                          config=BotoConfig(max_pool_connections=64, read_timeout=70, retries={"max_attempts": 3, "mode": "standard"}))


def account_limits(client) -> dict:
    lim = client.get_account_settings()["AccountLimit"]
    return {"concurrent_executions": lim.get("ConcurrentExecutions"), "unreserved": lim.get("UnreservedConcurrentExecutions")}


def pc_status(client, fn: str, alias: str) -> dict | None:
    try:
        r = client.get_provisioned_concurrency_config(FunctionName=fn, Qualifier=alias)
    except client.exceptions.ProvisionedConcurrencyConfigNotFoundException:
        return None
    return {"status": r.get("Status"), "requested": r.get("RequestedProvisionedConcurrentExecutions"),
            "available": r.get("AvailableProvisionedConcurrentExecutions"),
            "allocated": r.get("AllocatedProvisionedConcurrentExecutions"), "reason": r.get("StatusReason")}


def set_pc(client, fn: str, alias: str, n: int) -> None:
    """n > 0 -> set Provisioned Concurrency on the alias; n == 0 -> delete it (scale to zero, stop billing)."""
    from botocore.exceptions import ClientError
    if n <= 0:
        try:
            client.delete_provisioned_concurrency_config(FunctionName=fn, Qualifier=alias)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("ResourceNotFoundException", "ProvisionedConcurrencyConfigNotFoundException"):
                raise
    else:
        client.put_provisioned_concurrency_config(FunctionName=fn, Qualifier=alias, ProvisionedConcurrentExecutions=int(n))


def wait_pc_ready(client, fn: str, alias: str, n: int, timeout: int = 900, poll: int = 5, log=print) -> float:
    """Block until PC reports READY with >= n available. Returns seconds waited (use it to calibrate Config.pc_ready_s)."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        st = pc_status(client, fn, alias)
        if st and st["status"] == "FAILED":
            raise RuntimeError(f"Provisioned concurrency FAILED: {st['reason']}")
        if st and st["status"] == "READY" and (st["available"] or 0) >= n:
            return time.time() - t0
        desc = f"{st['status']} {st['available']}/{st['requested']}" if st else "not found"
        if desc != last:
            log(f"    PC status: {desc}")
            last = desc
        time.sleep(poll)
    raise TimeoutError("Provisioned concurrency did not become READY in time")


def wait_pc_gone(client, fn: str, alias: str, timeout: int = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout and pc_status(client, fn, alias) is not None:
        time.sleep(3)


def invoke_once(client, fn: str, alias: str, payload: dict | None = None) -> dict:
    t0 = time.perf_counter()
    r = client.invoke(FunctionName=fn, Qualifier=alias, InvocationType="RequestResponse", LogType="Tail",
                      Payload=json.dumps(payload or {}).encode())
    rtt = (time.perf_counter() - t0) * 1000
    raw = json.loads(r["Payload"].read() or b"{}")
    body = json.loads(raw["body"]) if isinstance(raw.get("body"), str) else {}
    log = base64.b64decode(r.get("LogResult", "")).decode("utf-8", "ignore")
    m = _INIT_RE.search(log)
    init_ms = float(m.group(1)) if m else 0.0
    return {"rtt_ms": round(rtt, 1), "cold": init_ms > 0, "init_ms": init_ms, "init_type": body.get("init_type"),
            "env_id": body.get("env_id"), "handler_ms": body.get("handler_ms"), "error": r.get("FunctionError")}


def burst(client, fn: str, alias: str, n: int, work_ms: int = 600) -> pd.DataFrame:
    """n simultaneous requests. Each holds its environment for work_ms, so n requests need n environments."""
    with ThreadPoolExecutor(max_workers=n) as ex:
        rows = list(ex.map(lambda _: invoke_once(client, fn, alias, {"work_ms": work_ms}), range(n)))
    return pd.DataFrame(rows)


def ping_prewarm(client, fn: str, alias: str, n: int, hold_ms: int = 1500) -> pd.DataFrame:
    """FREE fallback (not real PC): n parallel 'warmer' invocations create n warm on-demand environments.
    Lambda keeps them for a few minutes, but offers no guarantee."""
    with ThreadPoolExecutor(max_workers=n) as ex:
        rows = list(ex.map(lambda _: invoke_once(client, fn, alias, {"warmer": True, "hold_ms": hold_ms}), range(n)))
    return pd.DataFrame(rows)


def fresh_version(client, fn: str, alias: str, log=print) -> str:
    """Publish a NEW immutable version (by changing an env var) and point the alias at it, so the next burst is
    guaranteed to hit brand-new (cold) environments. Deletes PC first because PC is attached to the alias."""
    set_pc(client, fn, alias, 0)
    wait_pc_gone(client, fn, alias)
    cur = client.get_function_configuration(FunctionName=fn)
    env = dict(cur.get("Environment", {}).get("Variables", {}))
    env["NONCE"] = str(time.time())
    client.update_function_configuration(FunctionName=fn, Environment={"Variables": env})
    client.get_waiter("function_updated_v2").wait(FunctionName=fn)
    ver = client.publish_version(FunctionName=fn)["Version"]
    client.update_alias(FunctionName=fn, Name=alias, FunctionVersion=ver)
    log(f"    alias '{alias}' now points to new version {ver} (no warm environments exist)")
    return ver


def summarize_burst(df: pd.DataFrame) -> dict:
    return {"requests": len(df), "cold_starts": int(df["cold"].sum()),
            "on_PC": int((df["init_type"] == "provisioned-concurrency").sum()),
            "mean_ms": round(float(df["rtt_ms"].mean()), 1), "max_ms": round(float(df["rtt_ms"].max()), 1),
            "distinct_environments": int(df["env_id"].nunique())}
