# Simple Python image
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir requests numpy

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

# Run the orchestrator loop
CMD ["python", "orchestrator.py"]
