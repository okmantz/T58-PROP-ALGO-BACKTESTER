# T58 Prop Algo Backtester -- web/phone edition, containerized for Cloud Run.
#
# NOTE: this Dockerfile lives in docker/, but it must still be built with
# the repo ROOT as the build context (it COPYs the whole app in), e.g.:
#   docker build -f docker/Dockerfile -t <image> .
#
# Firebase Hosting only serves static files; it cannot run this Flask app
# directly (pandas/numpy backtests, Monte Carlo, GA search -- all real
# compute). So the actual app runs here, on Cloud Run, and Firebase Hosting
# is configured (see firebase.json) to transparently proxy every request to
# this container. End result from a phone/browser's point of view: one
# single URL (your Firebase Hosting domain), installable as a PWA, that is
# in fact this exact same engine as the desktop .exe.
FROM python:3.12-slim

WORKDIR /app

# System deps for pandas/numpy wheels and building any source packages.
RUN apt-get update && apt-get install -y --no-install-recursive \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY config/requirements.txt ./requirements.txt

# MetaTrader5 and pywebview are Windows-only / desktop-only (see
# requirements.txt's own platform_system markers) -- pip already skips them
# automatically on Linux, so no separate "web requirements" file is needed.
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Cloud Run sets $PORT; gunicorn must bind to it (not the hardcoded 5000
# app.web.server uses for local/LAN use). 1 worker, several threads: this
# app's own heavy jobs (Search Lab, Iterative Refinement, Full Pipeline) are
# already offloaded to background threads/process pools internally, so a
# single gunicorn worker keeps memory bounded and avoids duplicating those
# pools per worker.
ENV PYTHONUNBUFFERED=1
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 0 app.web.server:app
