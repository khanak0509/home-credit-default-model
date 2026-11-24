PYTHON ?= .venv/bin/python
RUN = PYTHONHASHSEED=0 $(PYTHON) -m
REPORTS ?= reports
export CREDIT_RISK_PROFILE

.PHONY: setup download data quality features audit train evaluate ablation explain readme all quick test test-unit lint score serve docker clean

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install --no-deps -e .

download:
	$(RUN) credit_risk.download

data: download
	$(RUN) credit_risk.data

quality: data
	$(RUN) credit_risk.quality

features: data
	$(RUN) credit_risk.split
	$(RUN) credit_risk.features

audit: features
	$(RUN) credit_risk.leakage

train: features
	$(RUN) credit_risk.train

evaluate:
	$(RUN) credit_risk.evaluate

ablation:
	$(RUN) credit_risk.ablation

explain:
	$(RUN) credit_risk.explain

readme:
	$(RUN) credit_risk.score --example --output $(REPORTS)/score_example.json
	$(RUN) credit_risk.readme

all: quality audit train evaluate ablation explain readme

quick: quality audit
	$(MAKE) train evaluate ablation explain readme CREDIT_RISK_PROFILE=quick REPORTS=build/quick/reports

test:
	$(PYTHON) -m pytest -q

test-unit:
	$(PYTHON) -m pytest -q -m "not integration"

lint:
	$(PYTHON) -m ruff check src tests

score:
	$(RUN) credit_risk.score $(ID)

serve:
	$(PYTHON) -m uvicorn credit_risk.api:app --host 0.0.0.0 --port 8000

docker:
	docker build -t credit-risk-lab .

clean:
	rm -rf data/staged data/processed models build
