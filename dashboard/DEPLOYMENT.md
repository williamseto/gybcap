# Dashboard Deployment Guide

Two modes: **local + Cloudflare Tunnel** (prototyping) or **Fly.io** (always-on cloud).

---

## Mode 1: Local + Cloudflare Tunnel (Recommended for Prototyping)

No account required. Gives a temporary public URL that changes on each restart.

### 1. Start the server locally

```bash
source ~/ml-venv/bin/activate

# Install new deps if needed
pip install fastapi "uvicorn[standard]" apscheduler

# Run dashboard (bootstraps from local CSV on first run, ~60-90s)
PYTHONPATH=/home/william/gybcap python -m dashboard.server
```

On first run you will see:
```
No cached state — running initial refresh (may take 1-2 min)...
Initial refresh complete — state computed in 73.2s
Starting dashboard server on http://0.0.0.0:8000
```

Subsequent restarts load the cached `dashboard/cache/state_snapshot.json` instantly.

### 2. Expose with Cloudflare Tunnel

Install cloudflared (one-time):
```bash
# On Debian/Ubuntu
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

Start the tunnel (no account required for temp URLs):
```bash
cloudflared tunnel --url http://localhost:8000
# → prints e.g. https://random-words.trycloudflare.com
```

The URL is valid until you stop cloudflared. Share it to view the dashboard remotely.

### 3. Verify endpoints

```bash
# Today's state (lightweight)
curl http://localhost:8000/api/today

# Full state
curl http://localhost:8000/api/state | python -m json.tool | head -40

# Health check
curl http://localhost:8000/health

# Manual refresh (requires secret header)
curl -X POST http://localhost:8000/api/refresh \
     -H "X-Refresh-Secret: changeme"

# WebSocket test (browser console)
# new WebSocket('ws://localhost:8000/ws/signals').onmessage = e => console.log(e.data)
```

### 4. Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_DATA_MODE` | `csv_plus_yfinance` | `yfinance_only` for cloud |
| `DASHBOARD_REFRESH_SECRET` | `changeme` | Protect POST /api/refresh |
| `DASHBOARD_PORT` | `8000` | Server port |
| `DASHBOARD_HOST` | `0.0.0.0` | Bind address |

Example with custom secret:
```bash
DASHBOARD_REFRESH_SECRET=mysecret123 \
PYTHONPATH=/home/william/gybcap \
python -m dashboard.server
```

---

## Mode 2: Fly.io Cloud Deployment (Always-On)

Fly.io provides free-tier VMs with persistent volumes. No credit card for hobby tier.

### 1. Install flyctl

```bash
curl -L https://fly.io/install.sh | sh
export PATH="$HOME/.fly/bin:$PATH"
```

### 2. Sign up / log in

```bash
fly auth signup   # or: fly auth login
```

### 3. Create app from dashboard/ directory

```bash
cd /home/william/gybcap

# Initialize Fly app (generates fly.toml)
fly launch \
  --name es-swing-dashboard \
  --region ord \
  --no-deploy \
  --image-label Dockerfile
```

Edit the generated `fly.toml` if needed:
```toml
[build]
  dockerfile = "dashboard/Dockerfile"

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]
  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

[env]
  DASHBOARD_DATA_MODE = "yfinance_only"
  DASHBOARD_PORT = "8080"
```

### 4. Create persistent volume for cache

```bash
fly volumes create dashboard_cache --size 1 --region ord
```

Add volume mount to `fly.toml`:
```toml
[[mounts]]
  source = "dashboard_cache"
  destination = "/app/dashboard/cache"
```

### 5. Set secret for refresh endpoint

```bash
fly secrets set DASHBOARD_REFRESH_SECRET=your-secure-secret-here
```

### 6. Deploy

```bash
fly deploy --dockerfile dashboard/Dockerfile
```

### 7. Open the dashboard

```bash
fly open
```

### 8. Trigger a manual refresh via CLI

```bash
curl -X POST https://es-swing-dashboard.fly.dev/api/refresh \
     -H "X-Refresh-Secret: your-secure-secret-here"
```

### Notes

- **First deploy**: yfinance will fetch ~15 years of ES=F history (~60-90s). Cache is persisted on the Fly volume.
- **Daily refresh**: The scheduler fires at 4:30 PM and 5:00 PM ET automatically.
- **VP features**: Unavailable in yfinance-only mode (11/86 features will be 0). This is acceptable — the model degrades gracefully.
- **Scaling**: A single `fly vm size shared-cpu-1x` (256MB) is sufficient. The pipeline is single-threaded and CPU-bound for ~60-90s once per day.

---

## Connecting the Realtime Engine

To feed live intraday signals from the realtime engine to the dashboard:

```python
import asyncio
from dashboard.signal_handler import DashboardSignalHandler
from strategies.realtime import RealtimeEngine, EngineConfig

# Get current state from runner
state = runner.last_state

# Shared WebSocket queue (same instance as used by the FastAPI app)
ws_queue = asyncio.Queue()

# Attach handler
handler = DashboardSignalHandler(state, ws_queue)
engine = RealtimeEngine(EngineConfig.default())
engine.signal_handler.add(handler)
engine.run()
```

Alternatively, POST signals directly from any process:
```bash
curl -X POST http://localhost:8000/api/signals \
     -H "Content-Type: application/json" \
     -d '{"strategy_name":"breakout","direction":"long","level_name":"vwap","entry_price":5250.0,"pred_proba":0.72}'
```
