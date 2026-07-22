#!/usr/bin/env bash
#
# host_notebook.sh — serve this repo from a Beaker session so Google Colab
# can connect to it as a "local runtime".
#
# One-time per session: create the session with a port mapping for 8888:
#
#   beaker session create --remote --bare --cluster ai2/<cluster> \
#       --gpus 1 --port 8888 --name "colab-jupyter-$(whoami)"
#
# Then inside the session:
#
#   git clone https://github.com/pjreddie/ssl_tutorial.git
#   cd ssl_tutorial && ./host_notebook.sh
#
# Follow the printed instructions: start the ssh tunnel on your laptop
# (VPN on), then in Colab use Connect ▾ → "Connect to a local runtime…".
#
# Note: Beaker captures session I/O, so the token printed below ends up in
# the session logs (visible to workspace members). To keep it out of the
# logs, export JUPYTER_TOKEN yourself before running this.

set -euo pipefail

PORT="${PORT:-8888}"
cd "$(dirname "$0")"

# Colab's file browser expects a content/ dir in the server's working directory.
mkdir -p content

if ! command -v jupyter >/dev/null || ! jupyter notebook --version >/dev/null 2>&1; then
    echo "error: 'jupyter notebook' not found; install with: pip install notebook" >&2
    exit 1
fi

TOKEN="${JUPYTER_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_hex(24))')}"

# Classic notebook (<7) is configured via NotebookApp; notebook 7+ via ServerApp.
APP=NotebookApp
NB_MAJOR="$(jupyter notebook --version 2>/dev/null | head -1 | cut -d. -f1)"
if [[ "$NB_MAJOR" =~ ^[0-9]+$ ]] && [ "$NB_MAJOR" -ge 7 ]; then
    APP=ServerApp
fi

# Look up the host port Beaker mapped to $PORT. This works whenever the
# beaker CLI is authenticated in here — always true in shells opened with
# `beaker session attach/exec --remote`, which set BEAKER_TOKEN.
HOST_PORT=""
if [ -n "${BEAKER_JOB_ID:-}" ] && command -v beaker >/dev/null; then
    HOST_PORT="$(beaker session describe --format json "$BEAKER_JOB_ID" 2>/dev/null \
        | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for p in d.get('runtime', {}).get('tcp_ports') or []:
        if p['container_port'] == $PORT:
            print(p['host_port'])
except Exception:
    pass
")" || true
fi

NODE="${BEAKER_NODE_HOSTNAME:-<node-hostname>}"

echo "======================================================================"
if [ -n "$HOST_PORT" ]; then
    echo "1. On your laptop (VPN on), start the tunnel:"
    echo
    echo "     ssh -N -L ${PORT}:localhost:${HOST_PORT} ${NODE}"
else
    echo "Couldn't look up the Beaker port mapping from inside the session."
    echo "1. On your laptop, find the host port with:"
    echo
    echo "     beaker session describe ${BEAKER_JOB_ID:-<session-id>}"
    echo
    echo "   Look for '<host-port>->${PORT}/tcp'. If no port is listed, the"
    echo "   session was created without '--port ${PORT}' — recreate it."
    echo
    echo "   Then start the tunnel (VPN on):"
    echo
    echo "     ssh -N -L ${PORT}:localhost:<host-port> ${NODE}"
fi
echo
echo "2. In Colab: Connect ▾ → 'Connect to a local runtime…' and paste:"
echo
echo "     http://localhost:${PORT}/?token=${TOKEN}"
echo "======================================================================"
echo

exec jupyter notebook \
    --ip 0.0.0.0 \
    --port "$PORT" \
    --no-browser \
    --allow-root \
    --"$APP".allow_origin='https://colab.research.google.com' \
    --"$APP".port_retries=0 \
    --"$APP".allow_credentials=True \
    --"$APP".token="$TOKEN"
