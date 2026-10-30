SHELL := /bin/bash
PDFS ?= case_files
QUESTIONS ?= eval/public_questions.jsonl
STAMP := $(shell date -u +%Y%m%dT%H%M%SZ)
OUT ?= eval/out/$(STAMP)
JUDGE_CACHE ?= eval/out/judge_cache
NAME ?= local
ARGS ?=
EXTERNAL := eval/external
LINT_PATHS := docqa tests tools docker eval
QMOUNTS = $(foreach q,$(QUESTIONS),-v "$(abspath $(q)):/input/questions/$(notdir $(q)):ro")
QARGS = $(foreach q,$(QUESTIONS),--questions "/input/questions/$(notdir $(q))")
.PHONY: up down logs lint test tenant eval eval-fixtures eval-robustness eval-external-fetch eval-external check-key

check-key:
	@test -n "$${GEMINI_API_KEY:-}" || grep -qE '^GEMINI_API_KEY=.+' .env 2>/dev/null || { echo "missing GEMINI_API_KEY: cp .env.example .env and set it, or export it in the shell"; exit 2; }
up: check-key
	docker compose up -d --build --wait
down:
	docker compose down
logs:
	docker compose logs -f api worker
lint:
	docker compose --profile test run --rm --build --no-deps test \
	  sh -c "ruff check --no-cache $(LINT_PATHS) && python tools/check_style.py $(LINT_PATHS)"
test:
	docker compose --profile test run --rm --build test pytest -q -p no:cacheprovider $(ARGS)
tenant: up
	docker compose exec -T api python -m docqa.cli create-tenant --name "$(NAME)"
eval: up
	@mkdir -p "$(OUT)" "$(JUDGE_CACHE)"
	docker compose --profile eval run --rm --user "$$(id -u):$$(id -g)" \
	  -v "$(abspath $(PDFS)):/input/pdfs:ro" $(QMOUNTS) \
	  -v "$(abspath $(OUT)):/output" -v "$(abspath $(JUDGE_CACHE)):/cache" \
	  eval run --pdfs /input/pdfs $(QARGS) --out /output --judge-cache /cache $(ARGS)
	@echo "report: $(OUT)/report.md"
eval-fixtures: up
	docker compose --profile eval run --rm --no-deps --user "$$(id -u):$$(id -g)" --entrypoint python \
	  -v "$(abspath case_files):/input/case_files:ro" -v "$(abspath eval/fixtures):/fixtures" \
	  eval /fixtures/make_fixtures.py --source /input/case_files/LB5001.pdf --out /fixtures/pdfs
eval-robustness: eval-fixtures
	$(MAKE) eval PDFS=eval/fixtures/pdfs QUESTIONS=eval/fixtures/questions_robustness.jsonl
eval-external-fetch: up
	@mkdir -p "$(EXTERNAL)/pdfs" "$(EXTERNAL)/originals"
	docker compose --profile eval run --rm --no-deps --user "$$(id -u):$$(id -g)" --entrypoint python \
	  -v "$(abspath $(EXTERNAL)):/external:ro" -v "$(abspath $(EXTERNAL)/pdfs):/external/pdfs" \
	  -v "$(abspath $(EXTERNAL)/originals):/external/originals" \
	  eval -I /external/fetch.py
eval-external: eval-external-fetch
	$(MAKE) eval PDFS=$(EXTERNAL)/pdfs QUESTIONS=$(EXTERNAL)/questions_external.jsonl
