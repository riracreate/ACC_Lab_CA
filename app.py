"""
LambdaWarm dashboard  ->  streamlit run app.py
Tabs: forecast & scaling | 5-metric scorecard | data & features | live AWS controls
"""
import os
import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import lambdawarm as lw

warnings.filterwarnings("ignore")
st.set_page_config(page_title="LambdaWarm", layout="wide")
st.title("LambdaWarm: ML-based proactive Provisioned Concurrency for AWS Lambda")
st.caption("Forecasts the next 5 minutes of Lambda concurrency, warms instances before the traffic arrives, "
           "and falls back to reactive scaling when the forecast turns out wrong.")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data(show_spinner="Scanning the Azure trace (first run takes about a minute, later runs are instant) ...")
def load_candidates(folder: str, days: int):
    return lw.azure_candidates(folder, days=days)


@st.cache_resource(show_spinner="Training the forecasting models ...")
def get_prepared(trace_id: str, y_bytes: bytes, cfg_dict: dict):
    cfg = lw.Config(**cfg_dict)
    return lw.prepare(np.frombuffer(y_bytes, dtype=float), cfg)


@st.cache_resource(show_spinner="Replaying the trace through all five policies ...")
def get_eval(trace_id: str, y_bytes: bytes, cfg_dict: dict, spike: bool, outage: bool):
    prep = get_prepared(trace_id, y_bytes, cfg_dict)
    return lw.evaluate(prep, spike, outage)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Workload")
    source = st.radio("Trace", ["Azure Functions 2019 (real)", "Synthetic demo trace"])
    y_raw, label = None, "synthetic"
    if source.startswith("Azure"):
        folder = st.text_input("Folder with invocations_per_function_md.anon.dNN.csv", "data/azure")
        try:
            table, series = load_candidates(folder, 3)
            pick = st.selectbox("Function", range(len(table)), format_func=lambda i: (
                f"{table.iloc[i]['function']} - {table.iloc[i]['calls_total']:,} calls, "
                f"{table.iloc[i]['idle_minutes_%']}% idle, daily pattern {table.iloc[i]['daily_pattern']}"))
            y_raw, label = series[table.iloc[pick]["key"]], f"Azure function {table.iloc[pick]['function']}"
        except FileNotFoundError as e:
            st.error(f"{e}\n\nExtract the Azure archive into that folder, or switch to the synthetic trace.")
            st.stop()
    if y_raw is None:
        y_raw = lw.synthetic_trace(3)

    st.header("Function and platform")
    peak_rpm = st.slider("Busiest traffic after rescaling (calls/min)", 30, 600, 60, step=10)
    service_s = st.slider("Execution time (s)", 0.05, 1.0, 0.2, step=0.05)
    init_s = st.slider("Cold-start time (s)", 0.5, 5.0, 1.8, step=0.1)
    mem_mb = st.select_slider("Memory (MB)", [128, 256, 512, 1024, 2048], value=512)
    pc_ready_s = st.slider("Seconds for a PC increase to become READY", 30, 300, 90, step=10)
    headroom = st.slider("Safety headroom on the forecast", 0.8, 2.0, 1.0, step=0.1)

    st.header("What if the forecast is wrong?")
    spike = st.checkbox("Flash crowd the model has never seen (under-prediction)")
    outage = st.checkbox("Traffic outage during the busiest hour (over-prediction)")

cfg = lw.Config(peak_rpm=peak_rpm, service_s=service_s, init_s=init_s, mem_mb=mem_mb, pc_ready_s=pc_ready_s, headroom=headroom)
trace_id = f"{label}|{len(y_raw)}|{float(np.sum(y_raw))}"
y_bytes = np.asarray(y_raw, dtype=float).tobytes()
ev = get_eval(trace_id, y_bytes, asdict(cfg), spike, outage)
prep = ev.prep
sc = ev.scorecard
od, lwm = sc.iloc[0], sc.loc["LambdaWarm: ML + fail-safes"]

st.subheader(f"{label}  |  {prep.a0} minutes of training, {len(ev.y) - prep.a0} minutes of unseen test traffic, {int(od['requests']):,} requests")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cold starts", f"{int(lwm['cold_starts'])}", f"{int(lwm['cold_starts'] - od['cold_starts'])} vs on-demand", delta_color="inverse")
c2.metric("P99.9 latency", f"{lwm['p99.9_ms']:.0f} ms", f"{lwm['p99.9_ms'] - od['p99.9_ms']:.0f} ms", delta_color="inverse")
c3.metric("Cost (test window)", f"${lwm['cost_usd']:.3f}", f"{lwm['cost_usd'] - od['cost_usd']:+.3f} USD", delta_color="inverse")
c4.metric("Energy (test window)", f"{lwm['energy_wh']:.1f} Wh", f"{lwm['energy_wh'] - od['energy_wh']:+.1f} Wh", delta_color="inverse")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast and scaling", "Five-metric scorecard", "Data and features", "Live AWS"])

with tab1:
    show(lw.fig_forecast(ev))
    show(lw.fig_scaling(ev))
    show(lw.fig_latency(ev))
    if ev.windows:
        st.info("Shaded bands mark the injected misprediction scenario. The 'fail-safe events' below count how often the "
                "watchdogs fired: **burst_mode** = under-prediction detected (cold starts while demand exceeded the forecast), "
                "**false_alarm** = over-prediction detected (a forecast that never materialised).")
    st.write({k: v.events for k, v in ev.results.items() if v.events})

with tab2:
    st.markdown("**1. Prediction accuracy** on the unseen test window")
    st.dataframe(ev.accuracy)
    st.markdown("**2-5. Utilisation, latency, cost, energy** for each policy")
    cols = ["requests", "cold_starts", "cold_start_%", "avg_ms", "p95_ms", "p99_ms", "p99.9_ms", "sla_violation_%",
            "pc_utilization_%", "pc_instance_hours", "cost_usd", "energy_wh", "extra_usd_per_1000_cold_starts_avoided"]
    st.dataframe(sc[cols])
    show(lw.fig_scorecard(ev))
    with st.expander("Assumptions behind cost and energy (state these in your report)"):
        st.json({k: v for k, v in asdict(cfg).items() if k.startswith(("price", "p_", "keepalive", "bill_init", "burst_frac"))})
        st.caption("Prices are x86 list prices (US East); Provisioned Concurrency is billed from the moment it is requested until it "
                   "is deleted. Energy uses a linear power model: active 4 W/GB, warm-idle 1.2 W/GB, scale-to-zero 0 W.")

with tab3:
    st.markdown("**Rescaled workload** (calls per minute); the model trains on the first 80 % and is tested on the rest")
    st.line_chart(pd.Series(ev.y, name="calls per minute"))
    st.markdown("**Features the model sees** (test window sample)")
    st.dataframe(ev.feats[lw.FEATURES + ["target_calls", "target_conc"]].iloc[prep.a0:prep.a0 + 40].round(2))
    show(lw.fig_features(ev))

with tab4:
    st.markdown("Controls for the **real** Lambda function. Uses the credentials saved by `aws configure`; never type keys here. "
                "Provisioned Concurrency is billed while it exists, so delete it when you are done.")
    a, b, c = st.columns(3)
    fn = a.text_input("Function", "CloudWorkloadTarget")
    alias = b.text_input("Alias", "prod")
    region = c.text_input("Region", "ap-south-1")
    minute = st.slider("Pretend it is this minute of the test window", 0, len(ev.y) - prep.a0 - 1, 100)
    fc = float(ev.p_conc_q[prep.a0 + minute - 1])
    rec = lw.size_from_forecast(fc, cfg)
    st.write(f"Forecast for the next 5 minutes: **{fc:.1f}** concurrent executions at the 90th percentile  ->  "
             f"controller requests **{rec}** warm instances.")
    n_set = st.number_input("Instances to set", 0, 100, rec)
    k = st.number_input("Burst size (simultaneous requests)", 1, 40, max(rec, 3))
    log = st.session_state.setdefault("live_log", [])

    def run(action):
        try:
            client = lw.lambda_client(region)
            if action == "status":
                st.write({"account": lw.account_limits(client), "provisioned_concurrency": lw.pc_status(client, fn, alias)})
            elif action == "set":
                lw.set_pc(client, fn, alias, int(n_set))
                st.success(f"Requested {int(n_set)} instances. Status: {lw.pc_status(client, fn, alias)}")
            elif action == "wait":
                with st.spinner("Waiting for READY ..."):
                    waited = lw.wait_pc_ready(client, fn, alias, int(n_set), log=lambda s: None)
                st.success(f"READY after {waited:.0f} s")
            elif action == "burst":
                df = lw.burst(client, fn, alias, int(k))
                st.write(lw.summarize_burst(df))
                st.dataframe(df)
                log.append({"burst": int(k), **lw.summarize_burst(df)})
            elif action == "delete":
                lw.set_pc(client, fn, alias, 0)
                st.success("Provisioned concurrency deleted - billing stopped.")
        except Exception as e:  # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")

    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button("Check status"):
        run("status")
    if b2.button("Set PC"):
        run("set")
    if b3.button("Wait until READY"):
        run("wait")
    if b4.button("Send burst"):
        run("burst")
    if b5.button("Delete PC"):
        run("delete")
    if log:
        st.markdown("**Bursts sent in this session**")
        st.dataframe(pd.DataFrame(log))
