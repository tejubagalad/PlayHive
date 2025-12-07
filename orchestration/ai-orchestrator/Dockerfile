# Simple Python image
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# --- OS deps: git + TLS certs (needed for GitHub clone) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- Python deps ---
# requests + numpy (metrics + regression) + pyyaml (edit app.yml)
RUN pip install --no-cache-dir requests numpy pyyaml

# Copy orchestrator
COPY orchestrator.py .

# Env defaults (can be overridden by Kubernetes)
ENV PROMETHEUS_URL="http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
ENV TARGET_POD_REGEX="gamehub-.*|snake-.*|game2048-.*"
ENV WINDOW_MINUTES="10"
ENV STEP_SECONDS="30"
ENV PREDICTION_HORIZON_SEC="120"
ENV CPU_THRESHOLD="80.0"
ENV LOOP_INTERVAL_SEC="60"

# GitOps config defaults (all overridden in Deployment)
ENV GIT_REPO_URL=""
ENV GIT_BRANCH="main"
ENV GIT_FILE_PATH="microservice-arch/yml/app.yml"
ENV GIT_DEPLOYMENT_NAME="gamehub"
ENV GIT_USER_NAME="AI Orchestrator Bot"
ENV GIT_USER_EMAIL="orchestrator@example.com"
ENV GIT_COMMIT_MESSAGE_PREFIX="[AI-Autoscale]"
ENV GIT_TOKEN=""

# Run the orchestrator loop
CMD ["python", "orchestrator.py"]

