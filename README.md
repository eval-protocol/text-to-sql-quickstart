# Text-to-SQL with GEPA Prompt Optimization

A quickstart example demonstrating GEPA prompt optimization on a text-to-SQL task using Eval Protocol.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export FIREWORKS_API_KEY="your-key"
```

### 3. Run GEPA training

The training script automatically starts the MCP server (which executes SQL queries against the database).

```bash
python evaluator/sql_gepa_training.py
```

This will:
- Load the pre-generated dataset from `datasets/`
- Split into train/validation sets
- Run GEPA to optimize the system prompt
- Print the optimized prompt

### 4. Evaluate results

Compare the original vs GEPA-optimized prompt on the test set:

```bash
python scripts/eval_baseline.py --prompt both
```

## Project Structure

```
text-to-sql-quickstart/
├── data/
│   └── synthetic_openflights.db    # DuckDB database with airlines/airports/routes
├── datasets/
│   ├── final_rft_sql_train_data.jsonl   # Training examples (183 rows)
│   └── final_rft_sql_test_data.jsonl    # Held-out test set (60 rows)
├── mcp_server/
│   └── run_mcp_server.py           # HTTP server that executes SQL queries
├── evaluator/
│   └── sql_gepa_training.py        # GEPA training script
└── scripts/
    ├── eval_baseline.py            # Evaluate prompts on test set
    └── 08_regenerate_balanced_data.py  # Data generation script
```

## Data Generation (Optional)

The dataset is already included in this repo. If you want to regenerate it:

```bash
python scripts/08_regenerate_balanced_data.py
```

This script:
- Uses the existing `data/synthetic_openflights.db` database
- Generates SQL query templates with consistent column naming
- Creates natural language questions for each query using an LLM
- Executes queries to get ground truth results
- Splits data into train/test sets with stratified sampling by query type

The generated data is saved to `datasets/final_rft_sql_train_data.jsonl` and `datasets/final_rft_sql_test_data.jsonl`.
