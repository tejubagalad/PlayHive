#!/usr/bin/env python3
import os
import time
import datetime as dt
import requests
import numpy as np

# ---------- Config (overridable via env vars) ----------
PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    # Local test: http://localhost:9090
    # In-cluster (kube-prometheus-stack service):
    # "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
    "http://localhost:9090",
)

# Regex of pods to watch; by default, all three game pods
TARGET_POD_REGEX = os.getenv(
    "TARGET_POD_REGEX",
    "gamehub-.*|snake-.*|game2048-.*",
)

WINDOW_MINUTES = int(os.getenv("WINDOW_MINUTES", "10"))           # lookback window
STEP_SECONDS = int(os.getenv("STEP_SECONDS", "30"))               # sample step
PREDICTION_HORIZON_SEC = int(os.getenv("PREDICTION_HORIZON_SEC", "120"))  # +2 min

# Default much lower so Minikube traffic can trigger actions
CPU_THRESHOLD = float(os.getenv("CPU_THRESHOLD", "20.0"))          # %

LOOP_INTERVAL_SEC = int(os.getenv("LOOP_INTERVAL_SEC", "60"))     # run every 60s


# ---------- Prometheus query + preprocessing ----------
def query_cpu_time_series():
    """
    Query Prometheus for CPU usage of target pods over the last WINDOW_MINUTES.
    Returns (timestamps, cpu_percent_list).
    """
    end_ts = time.time()
    start_ts = end_ts - WINDOW_MINUTES * 60

    query = (
        f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{TARGET_POD_REGEX}"}}[1m]))'
    )

    params = {
        "query": query,
        "start": start_ts,
        "end": end_ts,
        "step": STEP_SECONDS,
    }

    print("=== AI Orchestrator: Predict -> Think ===")
    print(f"[INFO] Querying Prometheus: {PROMETHEUS_URL}/api/v1/query_range")
    print(f"[INFO] Query: {query}")
    print(f"[INFO] Window: last {WINDOW_MINUTES} min, step={STEP_SECONDS}s")

    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus response not success: {data}")

    results = data["data"]["result"]
    if not results:
        print("[WARN] No time series returned for this query.")
        return [], []

    # We aggregate all returned series into one time-series (sum is already in PromQL)
    series = results[0]["values"]  # [ [timestamp, value_str], ... ]

    timestamps = []
    cpu_percent = []

    for ts_str, val_str in series:
        ts = float(ts_str)
        val_cores = float(val_str)           # CPU cores
        cpu_pct = max(val_cores * 100.0, 0)  # 1 core = 100%
        timestamps.append(ts)
        cpu_percent.append(cpu_pct)

    print(f"[INFO] Got {len(cpu_percent)} samples from Prometheus")

    # Show last few samples (for debugging)
    print("[INFO] Last 5 CPU % samples:")
    for t, v in list(zip(timestamps, cpu_percent))[-5:]:
        human = dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")
        print(f"  {human} -> {v:.4f}%")

    # Also show raw core values for clarity
    print("[DEBUG] Last 5 raw CPU core values:")
    for t, v in list(zip(timestamps, cpu_percent))[-5:]:
        human = dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")
        cores = v / 100.0
        print(f"  {human} -> {cores:.6f} cores")

    return timestamps, cpu_percent


# ---------- Prediction (simple linear regression) ----------
def predict_future_cpu(timestamps, cpu_percent, horizon_sec):
    """
    Fit y = a * t + b over the window and extrapolate CPU at t_last + horizon_sec.
    Returns predicted_cpu_percent.
    """
    if not cpu_percent:
        print("[WARN] No CPU samples; returning 0.0%")
        return 0.0

    if len(cpu_percent) == 1:
        print("[WARN] Only one sample; using that as prediction.")
        return cpu_percent[0]

    t0 = timestamps[0]
    x = np.array([t - t0 for t in timestamps], dtype=float)
    y = np.array(cpu_percent, dtype=float)

    # Fit line using numpy polyfit
    slope, intercept = np.polyfit(x, y, 1)
    print(f"[DEBUG] Regression slope={slope:.4f}, intercept={intercept:.2f}")

    future_t = timestamps[-1] + horizon_sec
    future_x = future_t - t0
    predicted = slope * future_x + intercept

    # Clip to [0, 100] for sanity
    predicted = max(0.0, min(predicted, 100.0))

    print(f"[DEBUG] Predicted CPU at t+{horizon_sec}s = {predicted:.4f}%")
    return predicted


# ---------- Think: Decide action based on prediction ----------
def decide_action(predicted_cpu_pct, threshold=CPU_THRESHOLD):
    """
    Simple 'Think' step:

    - If predicted CPU > threshold  -> SCALE_UP
    - Else                          -> NO_ACTION
    """
    print(
        f"[RESULT] Predicted CPU in ~{PREDICTION_HORIZON_SEC//60} "
        f"minutes: {predicted_cpu_pct:.4f}%"
    )

    if predicted_cpu_pct > threshold:
        action = "SCALE_UP"
    else:
        action = "NO_ACTION"

    print(f"[DECISION] Action = {action}")
    return action



# ---------- One full cycle: Predict -> Think ----------
def run_cycle():
    try:
        ts_list, cpu_pct_list = query_cpu_time_series()
        predicted = predict_future_cpu(ts_list, cpu_pct_list, PREDICTION_HORIZON_SEC)
        action = decide_action(predicted)

        # Phase 3: here we will hook GitOps / ArgoCD (e.g., commit to Git if SCALE_UP)
        # For now, we just print the decision.
        return action
    except Exception as e:
        print(f"[ERROR] Failed cycle: {e}")
        return "ERROR"


if __name__ == "__main__":
    print("[START] AI Orchestrator main loop")
    while True:
        action = run_cycle()
        print(f"[LOOP] Sleeping {LOOP_INTERVAL_SEC}s before next cycle...\n")
        time.sleep(LOOP_INTERVAL_SEC)

