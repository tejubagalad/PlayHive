#!/usr/bin/env python3
import os
import time
import datetime as dt
import requests
import numpy as np

# NEW: imports for GitOps
import subprocess
import tempfile
import yaml


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


# ---------- GitOps config helpers (Phase 3) ----------
def get_git_env():
    """
    Read Git-related configuration from environment variables.
    These are set via the Kubernetes Deployment.
    """
    return {
        "GIT_REPO_URL": os.getenv("GIT_REPO_URL", ""),
        "GIT_BRANCH": os.getenv("GIT_BRANCH", "main"),
        # Path to the deployment YAML file inside the repo
        "GIT_FILE_PATH": os.getenv("GIT_FILE_PATH", "microservice-arch/yml/app.yml"),
        # Name of the Deployment resource to scale (metadata.name)
        "GIT_DEPLOYMENT_NAME": os.getenv("GIT_DEPLOYMENT_NAME", "gamehub"),
        "GIT_USER_NAME": os.getenv("GIT_USER_NAME", "AI Orchestrator Bot"),
        "GIT_USER_EMAIL": os.getenv("GIT_USER_EMAIL", "orchestrator@example.com"),
        "GIT_COMMIT_MESSAGE_PREFIX": os.getenv(
            "GIT_COMMIT_MESSAGE_PREFIX", "[AI-Autoscale]"
        ),
        # Personal access token for HTTPS auth
        "GIT_TOKEN": os.getenv("GIT_TOKEN", ""),
    }


def run_cmd(cmd, cwd=None, extra_env=None):
    """
    Run a shell command, log it, and raise on error.
    Used for git clone/commit/push operations.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"[GIT] Running: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("[GIT] Command failed:", result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    if result.stdout:
        print("[GIT] Output:", result.stdout.strip())
    return result.stdout


def bump_replicas_in_file(file_path, deployment_name):
    """
    Load a YAML file (possibly multi-doc), find the Deployment with the given name,
    and increment its .spec.replicas by 1. Writes file back in-place.
    """
    with open(file_path, "r") as f:
        docs = list(yaml.safe_load_all(f))

    changed = False
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if doc.get("kind") == "Deployment" and \
           doc.get("metadata", {}).get("name") == deployment_name:
            current = doc.get("spec", {}).get("replicas", 1)
            new_val = current + 1
            doc.setdefault("spec", {})["replicas"] = new_val
            print(f"[ACT] Deployment {deployment_name}: replicas {current} -> {new_val}")
            changed = True

    if not changed:
        raise RuntimeError(f"Deployment {deployment_name} not found in {file_path}")

    with open(file_path, "w") as f:
        yaml.safe_dump_all(docs, f, sort_keys=False)


def act_via_git_scale_up():
    """
    Phase 3 'Act' step:
    - Clone the Git repo for manifests
    - Bump replicas for the target Deployment in the YAML
    - Commit and push
    ArgoCD will detect the change and sync it to the cluster.
    """
    cfg = get_git_env()

    if not cfg["GIT_REPO_URL"]:
        print("[ACT] GIT_REPO_URL is empty; skipping GitOps action.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[GIT] Using temp dir: {tmpdir}")

        # Build authenticated URL if token is provided
        repo_url = cfg["GIT_REPO_URL"]
        if cfg["GIT_TOKEN"] and repo_url.startswith("https://"):
            # Insert token into https:// URL for non-interactive clone
            repo_url = repo_url.replace("https://", f"https://{cfg['GIT_TOKEN']}@")

        # Clone repo
        run_cmd(["git", "clone", "--branch", cfg["GIT_BRANCH"], repo_url, "repo"], cwd=tmpdir)
        repo_dir = os.path.join(tmpdir, "repo")

        # Configure git user
        run_cmd(["git", "config", "user.name", cfg["GIT_USER_NAME"]], cwd=repo_dir)
        run_cmd(["git", "config", "user.email", cfg["GIT_USER_EMAIL"]], cwd=repo_dir)

        # Path to YAML file inside repo
        target_file = os.path.join(repo_dir, cfg["GIT_FILE_PATH"])

        # Modify replicas
        bump_replicas_in_file(target_file, cfg["GIT_DEPLOYMENT_NAME"])

        # Commit and push
        run_cmd(["git", "add", cfg["GIT_FILE_PATH"]], cwd=repo_dir)

        commit_msg = f"{cfg['GIT_COMMIT_MESSAGE_PREFIX']} scale up {cfg['GIT_DEPLOYMENT_NAME']}"
        run_cmd(["git", "commit", "-m", commit_msg], cwd=repo_dir)
        run_cmd(["git", "push", "origin", cfg["GIT_BRANCH"]], cwd=repo_dir)

        print("[ACT] GitOps scale-up commit pushed successfully.")


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


# ---------- One full cycle: Predict -> Think -> (optional) Act ----------
def run_cycle():
    try:
        ts_list, cpu_pct_list = query_cpu_time_series()
        predicted = predict_future_cpu(ts_list, cpu_pct_list, PREDICTION_HORIZON_SEC)
        action = decide_action(predicted)

        # Phase 3: GitOps / ArgoCD hook
        if action == "SCALE_UP":
            try:
                act_via_git_scale_up()
            except Exception as e:
                print(f"[ACT] Failed to perform GitOps scale-up: {e}")
        else:
            print("[ACT] No action taken this cycle.")

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

