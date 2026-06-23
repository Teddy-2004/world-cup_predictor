# Deploying WC2026 Predictor to Railway

Railway gives you a free tier with 500 compute hours/month — plenty for a
tournament predictor that only runs when someone hits the URL.

## Prerequisites

- Git installed
- [Railway CLI](https://docs.railway.app/develop/cli) installed: `npm install -g @railway/cli`
- Your trained models in `data/trained_models/`
- Your parquet files in `data/parquet/`

## Step 1 — Prepare your repo

```bash
cd wc2026_predictor/

# Initialise git if not already done
git init
git add .
git commit -m "Initial commit"
```

## Step 2 — Log in to Railway

```bash
railway login
```

This opens a browser. Sign in with GitHub.

## Step 3 — Create a new Railway project

```bash
railway new
# Choose "Empty project"
# Name it: wc2026-predictor
```

## Step 4 — Link and deploy

```bash
railway link   # links this folder to the project you just created
railway up     # builds the Docker image and deploys
```

Railway will:
1. Build the Dockerfile
2. Start the container with your API
3. Give you a public URL like `wc2026-predictor.up.railway.app`

## Step 5 — Upload your data files

Your trained models and parquet files are too large for git (typically 50-500MB).
Upload them as a Railway Volume:

```bash
# In the Railway dashboard:
# Project → Add Volume → Mount at /app/data

# Then copy your data:
railway run -- bash -c "mkdir -p /app/data"
rsync -avz data/trained_models/ railway-volume:/app/data/trained_models/
rsync -avz data/parquet/        railway-volume:/app/data/parquet/
```

Or simpler — use Railway's shell to upload:

```bash
railway shell
# Then inside the shell:
mkdir -p data/trained_models data/parquet data/simulation_results
exit

# Copy files using scp (Railway provides the connection details in dashboard)
```

## Step 6 — Verify

```bash
# Check health
curl https://your-project.up.railway.app/health

# Test a prediction
curl -X POST https://your-project.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"France","away_team":"Germany","venue":"MetLife Stadium","stage":"GROUP_STAGE"}'
```

## Step 7 — Open the dashboard

Visit: `https://your-project.up.railway.app`

The dashboard auto-detects the API URL and starts making real predictions.

---

## Updating predictions after matches

After each WC2026 match result:

```bash
# 1. Update your local data
python collect.py --since 2026

# 2. Re-build features
python features/assembler.py

# 3. Re-train (optional — only needed for major updates)
python train.py

# 4. Re-deploy
git add data/trained_models/ data/parquet/
git commit -m "Update after match results YYYY-MM-DD"
railway up
```

---

## Environment variables (optional)

Set in Railway dashboard → Variables:

| Variable         | Default  | Description                    |
|-----------------|---------|-------------------------------|
| `PORT`          | 8000    | Set automatically by Railway   |
| `LOG_LEVEL`     | info    | uvicorn log level              |
| `N_SIMS`        | 5000    | Simulations for forecast       |

---

## Free tier limits

Railway free tier: 500 compute hours / month.
With 1 worker and typical traffic (predictions take ~200ms each), you can
handle ~8,000 requests before hitting the limit. More than enough for a
tournament predictor.

To stay within limits:
- The `/forecast` endpoint is cached — it only re-runs simulations when
  you hit `/forecast/refresh`
- Predictions run in <500ms each with the full model loaded

## Troubleshooting

**"Model not found" on startup**: Your `data/trained_models/` wasn't uploaded.
The API falls back to ELO-only mode and still works — just less accurate.

**Slow first response**: Railway spins down containers on the free tier after
inactivity. First request takes ~5 seconds to wake up. Subsequent requests
are fast.

**Out of memory**: The full model + assembler uses ~1-2GB RAM. Railway free
tier gives 512MB. If you hit memory limits, use `--elo-only` mode by setting
the startup command to use ELO predictions, or upgrade to Railway's $5/month
plan for 8GB RAM.