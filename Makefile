PYTHON ?= python
PIP ?= pip

.PHONY: install
install:
	$(PIP) install -r requirements.txt

.PHONY: all-data
all-data: sim-prod extract-schema synth gen-queries augment ground-truth gen-nl

.PHONY: sim-prod
sim-prod:
	$(PYTHON) scripts/01_simulate_prod_db.py

.PHONY: extract-schema
extract-schema:
	$(PYTHON) scripts/02_extract_schema.py

.PHONY: synth
synth:
	$(PYTHON) scripts/03_generate_synthetic_data.py

.PHONY: gen-queries
gen-queries:
	$(PYTHON) scripts/04_generate_queries.py

.PHONY: augment
augment:
	$(PYTHON) scripts/05_augment_sandbox.py

.PHONY: ground-truth
ground-truth:
	$(PYTHON) scripts/06_ground_truth.py

.PHONY: gen-nl
gen-nl:
	$(PYTHON) scripts/07_generate_nl_questions.py

.PHONY: test
test:
	pytest -q

# ============================================================
# Verification targets - validate synthetic data pipeline
# ============================================================

.PHONY: verify
verify: verify-db verify-queries verify-gt verify-dataset verify-nl-sql
	@echo "✅ All verifications complete!"

.PHONY: verify-db
verify-db:
	$(PYTHON) scripts/verify_database.py

.PHONY: verify-queries
verify-queries:
	$(PYTHON) scripts/verify_queries.py

.PHONY: verify-gt
verify-gt:
	$(PYTHON) scripts/verify_ground_truth.py

.PHONY: verify-dataset
verify-dataset:
	$(PYTHON) scripts/verify_dataset.py

.PHONY: verify-nl-sql
verify-nl-sql:
	$(PYTHON) scripts/verify_nl_sql_match.py

.PHONY: verify-all
verify-all:
	$(PYTHON) scripts/verify_all.py

# ============================================================
# Docker targets
# ============================================================

.PHONY: mcp-build
mcp-build:
	docker build -t text-to-sql-mcp:latest .

.PHONY: mcp-deploy
# Usage: make mcp-deploy PROJECT_ID=your-id REGION=us-central1
mcp-deploy:
	gcloud run deploy mcp-sql-rft-server \
	  --source mcp_server \
	  --project $(PROJECT_ID) \
	  --region $(REGION) \
	  --allow-unauthenticated \
	  --port 8080
