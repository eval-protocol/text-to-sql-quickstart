# Text-to-SQL with GEPA Prompt Optimization + RFT

### GEPA Prompt Optimization (Text-to-SQL)

This path fine-tunes the **system prompt** for text-to-SQL using GEPA (guided evaluation prompt optimization).

#### GEPA Quickstart
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Set your Fireworks API key:
```bash
export FIREWORKS_API_KEY="your-key"
```

3) Run GEPA training (this will start the MCP server locally as needed):
```bash
python evaluator/sql_gepa_training.py
```

4) Compare original vs GEPA-optimized prompts on the test set:
```bash
python scripts/eval_baseline.py --prompt both
```

---

### RFT with Eval Protocol

This path uses **Reinforcement Fine-Tuning (RFT)** to train a model to generate correct SQL against the synthetic database, evaluated via MCP.

#### Prerequisites
- Python 3.11+ (recommend `uv` or `venv`)
- `FIREWORKS_API_KEY` (Fireworks account)
- Google Cloud SDK (for Cloud Run MCP deployment) if you want remote server
- Optional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` for benchmarking additional models

#### RFT Quickstart
1) Create a Python environment in this folder and install:
```bash
pip install -r requirements.txt
```

The below steps we've already done, you can just skip to step 5 and kick off an RFT job.

2) Generate data (OpenFlights → prod → synthetic → queries → ground-truth → NL):
```bash
make all-data
```
You typically only need to run this once to produce `data/synthetic_openflights.db` and the JSONL datasets.
We’ve already generated these artifacts, copied the DB into `mcp_server/data`, and deployed an MCP server using that database.
You can reuse this setup and skip regeneration unless you explicitly want to create a new synthetic dataset and deploy your own MCP server.

3) Build and deploy MCP server to Cloud Run (from the `mcp_server/` directory):
```bash
cd mcp_server
gcloud run deploy mcp-sql-rft-server \
  --source . \
  --project YOUR_GCP_PROJECT_ID \
  --region YOUR_GCP_REGION \
  --allow-unauthenticated \
  --port 8080
```
This uses `mcp_server/Dockerfile` (and `mcp_server/.gcloudignore`). Make sure `mcp_server/data/synthetic_openflights.db`
exists before deploying (for example by copying it from `../data/synthetic_openflights.db`).
Copy the Cloud Run service URL (without trailing `/mcp/`) and either:
- export it as `MCP_SERVER_URL`, or
- hard-code it into the evaluator if desired.

4) Test evaluator locally:
```bash
ep local-test
```

Then select `test_sql_rft_local` in `sql_rft_evaluator.py`.

5) Launch RFT:
```bash
ep create rft \
  --base-model accounts/fireworks/models/qwen3-32b \
  --chunk-size 10 \
  --epochs 8
```

 Again, select `test_sql_rft_local` in `sql_rft_evaluator.py` as the evaluation function.
