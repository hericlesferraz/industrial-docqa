# docqa

Question answering over industrial PDF manuals, with verifiable citations. The design decisions are in
[`DECISIONS.md`](DECISIONS.md).

The plan for taking it to production scale is in [`DESIGN.md`](DESIGN.md).

## Architecture

[![docqa MVP: api, worker and db under docker compose, with the ingestion and question flows](docs/architecture/mvp.png)](docs/architecture/mvp.svg)

`docker compose up` starts three services on one host; `make eval` adds a fourth.

| Service | What it does |
|---|---|
| `api` | FastAPI on port 8000: uploads, documents, questions and answer records. It streams each upload to the `blobs` volume and never opens a PDF. It holds the BGE-M3 int8 embedder in its process, so a question is embedded and searched locally. |
| `worker` | One claim loop per job kind: `extract` (one at a time), `ocr_page` (three at a time) and `index` (one at a time), with lease heartbeats and an hourly cleanup. A PDF is opened only in a sandboxed child process, without secrets and with time, stall and memory limits. The worker holds its own copy of the embedder to index chunks. |
| `db` | Postgres 16 with pgvector and `unaccent`: documents and pages, chunks with their vectors and full text, the job queue, the OCR cache, answer records and spans. |
| `eval` | Runs only under `make eval`: the harness talks to the API like a client, and the judge calls Gemini. |

The PDFs live in the `blobs` volume, shared by `api` and `worker`, under `{tenant_id}/{sha256}.pdf`. Gemini is the only
external service: it transcribes the pages whose text layer is missing or broken, writes the answers, and judges them
in the eval.

Ingestion (blue): the API checks and stores the file, then commits the document row and its `extract` job in one
transaction (I1–I3). The worker claims the job (I4). The sandboxed child reads each page's text layer and flags the
unusable pages (I5), and the worker writes the page rows and queues the next jobs (I6). Each flagged page is rendered,
transcribed by Gemini and written back, cached by file content (I7–I9). `index` cuts each page into chunks, embeds
them and stores them; a page is searchable as soon as it is indexed (I10–I11).

Question (green): the API embeds the question (Q2) and runs the hybrid search (Q3). It sends the 8 best passages to
Gemini and verifies the answer, with at most one retry (Q4), then writes the record before it responds (Q5).

## Setup

Requirements: Docker with Compose v2 and `make`. No Python is needed on the host.

```sh
cp .env.example .env        # then set GEMINI_API_KEY in .env, or export GEMINI_API_KEY in the shell
make up                     # builds the image, starts db, api and worker, waits until all three are healthy
make tenant NAME=my-team    # creates a tenant and prints its tenant_id and api_key
```

The first build downloads the BGE-M3 ONNX export (about 2.3 GB) from Hugging Face and quantizes it to int8, so the
container never downloads anything at start. Measured on the build machine: 10 to 13 minutes, almost all of it the
download, and a peak of about 4 GB of RAM during the 5-second quantization step (the fp32 weights stay on disk until
each one is quantized). On Docker Desktop, give the VM at least 6 GB of memory for the first build; later builds reuse
the cached layer. At run time the api holds about 1 GB (the embedding model) and the worker about 1.3 GB while it
indexes, plus up to 0.5 GB for the PDF extraction child (measured on the sample manuals).

The API listens on `http://localhost:8000`; the interactive documentation is at `http://localhost:8000/docs`
(use the **Authorize** button with a tenant key, or the admin key for the admin route).

## Commands

| Command | What it does |
|---|---|
| `make up` | Checks that `GEMINI_API_KEY` is set (shell or `.env`), builds if needed, starts the stack and waits for health |
| `make down` | Stops the stack (data stays in the `pgdata` and `blobs` volumes) |
| `make logs` | Follows the api and worker logs (JSON lines) |
| `make tenant NAME=<name>` | Creates a tenant through the admin endpoint and prints its key |
| `make lint` | `ruff` plus `tools/check_style.py` (no `#` comments, a docstring on every function and class) |
| `make test [ARGS="-k queue"]` | Runs `pytest` in a container against the `docqa_test` database; `ARGS` go to `pytest` |
| `make eval [PDFS=<dir>] [QUESTIONS="<file> ..."] [ARGS="--no-judge --repeat 2"]` | Runs the eval harness (see "Eval"); defaults: `case_files/` and `eval/public_questions.jsonl` |
| `make eval-robustness` | Generates the robustness files into `eval/fixtures/pdfs/` and runs the eval on them in their own tenant |
| `make eval-external-fetch` | Downloads the external eval set listed in `eval/external/manifest.json` into `eval/external/pdfs/` and checks each file's sha256; no model calls |
| `make eval-external [ARGS="--no-judge"]` | Optional, apart from `make eval`: fetches the external set, then runs the eval on it with `eval/external/questions_external.jsonl` (see "Eval") |

Paths given to `make eval` must not contain spaces.

## Environment variables

Only `GEMINI_API_KEY` must be set. `compose.yaml` passes `GEMINI_API_KEY` and `ADMIN_API_KEY` from the shell first, then
from `.env`; the other variables go in `.env`.

| Variable | Default | Read by | Notes |
|---|---|---|---|
| `GEMINI_API_KEY` | required | api, worker, eval | Missing: the process exits with code 2 and a one-line message. Never logged |
| `ADMIN_API_KEY` | `local-dev-admin-key-change-me` | api, `docqa.cli`, eval | Protects `POST /v1/admin/tenants`. Change it outside local use |
| `DATABASE_URL` | set by compose | api, worker | |
| `GEMINI_ANSWER_MODEL` | `gemini-3.8-flash` | api | On the free tier, `gemini-3.5-flash-lite` |
| `GEMINI_OCR_MODEL` | `gemini-3.5-flash-lite` | worker; api (start check) | Transcribes the pages whose text layer is missing or broken |
| `GEMINI_OCR_FALLBACK_MODEL` | `gemini-3.8-flash` | worker; api (start check) | Used once per attempt after a refusal or a cut, and on the last attempt |
| `EMBEDDING_PROVIDER` | `local` | api, worker | `local` (BGE-M3 int8 in the container) or `gemini` (Gemini Embedding 2 at 1024 dimensions; the corpus is sent to the provider at ingestion). Switching reindexes every page at the next worker start, without OCR |
| `BLOB_STORE` | `local` | api, worker | Only `local` (a named volume) exists |
| `BLOB_STORE_PATH` | `/data/blobs` | api, worker | |
| `MAX_UPLOAD_MB` | `200` | api | Larger uploads get 413 |
| `MAX_PDF_PAGES` | `2000` | worker | |
| `EXTRACT_TIMEOUT_S` | `2400` | worker | Total time limit of the PDF extraction child; it is also killed after 300 s without finishing a page. Keep it below `EVAL_INGEST_TIMEOUT_S` |
| `EXTRACT_MEMORY_MB` | `2048` | worker | Resident memory limit of the PDF extraction child (measured peak on the sample manuals: 480 MiB) |
| `OCR_MAX_CONCURRENCY` | `3` | worker | Global cap of page transcriptions running at once |
| `OCR_MIN_INTERVAL_S` | `1.0` | worker | Minimum gap between OCR calls per model; on the free tier, `4.5` |
| `EMBED_BATCH_SIZE` | `1` | api, worker | Texts per ONNX call of the local embedder. The int8 model quantizes each batch with one range, so a batch changes the vectors slightly; any value other than 1 is part of the index version (and reindexes). One text per call was faster on CPU and uses less memory |
| `LOG_LEVEL` | `INFO` | api, worker, eval | |
| `EVAL_API_URL` | `http://api:8000` | eval, `docqa.cli` | |
| `EVAL_INGEST_TIMEOUT_S` | `3600` | eval | |
| `EVAL_JUDGE_MODEL` | `gemini-3.5-flash-lite` | eval | Provisional |

## Requests

Every response carries an `X-Request-ID` header. Every error has the same body:

```json
{"error": {"code": "unauthorized", "message": "Missing or invalid API key.", "request_id": "…", "retryable": false}}
```

Health (no authentication):

```sh
curl -s localhost:8000/healthz
# {"status":"ok","checks":{"database":"ok","embedder":"ok","models":{"gemini-3.8-flash":"ok","gemini-3.5-flash-lite":"ok"}},"reasons":[]}
```

At start the API loads the embedder (and runs it once) before it answers, then checks each configured model once with
a metadata call (no generation, no cost; the embedding model too with `EMBEDDING_PROVIDER=gemini`). A rejected key or
model leaves the status `degraded` with the reason (for example `"gemini-3.8-flash: provider rejected the model"`), but
the service keeps running: search is local. The database unreachable, or the embedder not loaded, answers 503.

Create a tenant (admin key):

```sh
curl -s -X POST localhost:8000/v1/admin/tenants \
  -H "Authorization: Bearer local-dev-admin-key-change-me" -H "Content-Type: application/json" \
  -d '{"name": "plant-7"}'
# {"tenant_id":"…","name":"plant-7","api_key":"dq_…"}
```

The key is shown once; only its sha256 is stored. Tenant routes take it as `Authorization: Bearer dq_…`.

Upload a folder as one batch, then follow it (one file per request; the same bytes again answer 200 with
`"duplicate": true`; a new document answers 202):

```sh
KEY=dq_…
BATCH=$(curl -s -X POST localhost:8000/v1/batches -H "Authorization: Bearer $KEY" | python3 -c 'import json,sys; print(json.load(sys.stdin)["batch_id"])')
for f in case_files/*.pdf; do
  curl -s -X POST "localhost:8000/v1/documents?batch_id=$BATCH" -H "Authorization: Bearer $KEY" -F "file=@$f"; echo
done
# {"document_id":"…","filename":"LB5001.pdf","status":"queued","duplicate":false}
curl -s localhost:8000/v1/batches/$BATCH -H "Authorization: Bearer $KEY"
# {"batch_id":"…","complete":false,"counts":{"items":4,"accepted":4,"duplicate":0,"rejected":0,
#  "documents":{"queued":0,"processing":2,"ready":2,"partial":0,"failed":0}},
#  "items":[{"filename":"LB5001.pdf","outcome":"accepted","error_code":null,"created_at":"…",
#            "document":{"document_id":"…","status":"ready","page_count":2,"pages_indexed":2,"pages_pending":0,
#                        "pages_failed":0,"error_code":null,"ocr_paused":null}}, …]}
```

A batch is `complete` when every document is `ready`, `partial` or `failed`, or was rejected at upload (`not_a_pdf`,
`empty_file`, `file_too_large`). Files the worker can't read end `failed` with a code: `pdf_encrypted`, `pdf_corrupt`,
`pdf_no_pages`, `pdf_too_many_pages`, `extract_timeout`, `extract_memory`, `extract_crashed`, or `no_text` when every
page is blank. When the provider can't be used for a while (daily quota, prepaid balance or spend cap exhausted, key or
model rejected), transcriptions pause for 15 minutes without spending attempts, and `ocr_paused` names the model (`*`
for the whole account), the reason and the end of the pause.

List the documents, and inspect one (pages by state and text source, the reason each page went to OCR, the OCR calls and
tokens):

```sh
curl -s "localhost:8000/v1/documents?limit=100&offset=0" -H "Authorization: Bearer $KEY"
curl -s localhost:8000/v1/documents/<document_id> -H "Authorization: Bearer $KEY"
# {"document_id":"…","filename":"WEG-CESTARI-….pdf","status":"processing","error":null,"page_count":84,
#  "pages_indexed":28,"pages_pending":53,"pages_failed":0,
#  "pages_by_state":{"needs_ocr":53,"text_ready":0,"indexed":28,"blank":3,"failed":0},
#  "pages_by_source":{"layer":28,"ocr":0},
#  "ocr":{"pages_flagged":53,"pages_transcribed":0,"pages_from_cache":0,"recitations":0,"fallback_used":0,
#         "calls":0,"input_tokens":0,"output_tokens":0,"by_model":{},"paused":null},
#  "pages":[{"page":18,"state":"needs_ocr","text_source":null,"page_label":"14",
#            "flag_reason":"garbled:few_vowels,control_chars","error":null}, …]}
```

Flagged pages are rendered at 150 dpi and transcribed by `GEMINI_OCR_MODEL`; a `RECITATION` refusal is retried twice
on the same model, and a page that still fails gets one call on `GEMINI_OCR_FALLBACK_MODEL`. Only a complete answer is
accepted. Transcriptions are cached by file content (sha256, page, OCR version), so the same file uploaded again, by
any tenant, costs no model call (`pages_from_cache`). `output_tokens` include thinking tokens, which are billed as
output. A page that can't be transcribed ends `failed` with a code: `render_failed`, `ocr_recitation`, `ocr_safety`,
`ocr_max_tokens`, `ocr_empty`, `ocr_finish_other`, `ocr_invalid_request` (for example an image over the provider's size
limit), or `ocr_unavailable` (the provider kept failing through every attempt); the document is then `partial`.

## Index and search

Once a page has text, the worker's `index` job cuts it into chunks of whole units (headings, sentences, table rows) of
about 250 tokens, never across pages; a table stays with its caption and footnotes, and each chunk starts with the file
name and the nearest heading. A table too long for one chunk is split by rows: every piece repeats the first row, and
the pieces after the first also carry the table's column header rows among the units the model reads, not in the text
the search reads. Each chunk is embedded (BGE-M3 int8 on CPU by default) and stored with its units, which
answers will cite. A page is searchable as soon as it is indexed, and the document counters (`pages_indexed`,
`pages_pending`, `pages_failed`) change in the same transaction. Reindexing a page moves its old chunks to
`chunks_history` (kept while an answer references them, deleted after 24 hours otherwise).

Search fuses two ranked lists with reciprocal rank fusion (k = 60): the 50 nearest chunks by cosine (pgvector HNSW
with iterative scans, so a scope filter still yields full lists), and the chunks matching the question's rare terms
(lexemes found in 1 to max(3, 0.3% of the tenant's chunks) chunks that are not function words of English, Portuguese,
Spanish, German, French or Italian; Postgres full text with `simple` and `unaccent`). Codes such as part
numbers are found by the second list in any language. A scope (`document_ids`) filters both lists; rarity is still
counted over the whole tenant.

With `EMBEDDING_PROVIDER=gemini`, a rate limit or a provider pause delays indexing without spending attempts; when
indexing gives up, the waiting pages end `failed` with `index_failed`. Every index run records `index.document`
(pages, chunks, peak RSS of the worker) and `embed.chunks` spans; every search records `search.vector`,
`search.lexical` (lexemes, rare terms, the bound `docqa.rare_max_df`) and `search.fuse`.

## Questions and answers

```bash
curl -s -X POST localhost:8000/v1/questions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"question": "What is the relubrication interval for a Baldor ball bearing motor, NEMA frame 256, running at 1800 RPM?"}'
# {"answer_id":"b08ccba8-…","status":"answered",
#  "answer":"… operating at 1800 RPM, the recommended relubrication interval is 9500 Hrs. …",
#  "citations":[{"document_id":"…","filename":"LB5001.pdf","page":2,"page_label":null,"transcribed":false,
#                "quote":"Motors that do not have regrease capability are factory lubricated for the normal life of the bearings.\nRelubrication Intervals (For motors with rearease capability)\nRelubricatio | n Intervals\nNEMA (IEC) Frame Size | Rated Speed | (RPM)\n3600 | 1800 | 1200 | 900\nOver 210 to 280 incl.(180) | 3600Hrs. | 9500Hrs. | 15000Hrs. | 18000Hrs."}],
#  "coverage":{"complete":true,"documents":4,"documents_incomplete":0,"pages_indexed":166,"pages_pending":0,
#              "pages_failed":0,"incomplete":[]}}
```

- `status` is `answered`, `ambiguous` or `not_found`. For `ambiguous`, `answer` says what the answer depends on, and
  `options` lists each case with its `condition`, `answer` and `citations`; the top-level `citations` merges them per
  page. `not_found` has no citations.
- A citation is one page: the units the model cited (sentences, table rows), in page order, verbatim in the page's
  language, plus the header rows of a cited table row, the footnote of a starred row, and an uncited sentence of the same
  page that holds a number of the answer. `transcribed: true` marks a page read from its image.
- `document_ids` (optional, 1–100 ids) limits the search to those documents; an id the tenant doesn't own is a 404
  `document_not_found` with the ids in `details.document_ids`.
- `coverage` tells whether every document in scope is `ready`; `pages_pending` means a `not_found` may change later,
  `pages_failed` that no answer will cover those pages.
- Every number of an answer must appear in the units it cites (any locale reading: `9,500` matches `9500Hrs.`,
  `13,000` matches `13.000 horas`). If it doesn't, the model is asked once more with the problem; if it fails again the
  answer is `not_found` with `reason: "unverified"`, a fixed English `answer` and the most relevant `passages`. The
  rejected text is never returned (it stays in the answer record).

Errors of the answer path (one retry, within a 60 s deadline, on provider 5xx, per-minute rate limits and timeouts;
none on the daily quota, billing or a rejected key):

| HTTP | `code` | `retryable` | When |
|---|---|---|---|
| 503 | `llm_unavailable` | true | Provider down (5xx, connection), after the retry |
| 503 | `llm_rate_limited` | true | Rate limit (`Retry-After` from the provider) or daily quota (`Retry-After: 3600`) |
| 503 | `llm_billing` | false | Prepaid credit exhausted or spend cap reached |
| 503 | `llm_misconfigured` | false | Key or model rejected |
| 502 | `llm_invalid_output` | true | Output outside the schema, or a finish reason other than `STOP` (including the 4,096-token output cap, thinking included) |
| 502 | `llm_refused` | false | Safety or recitation refusal |
| 502 | `llm_invalid_request` | false | The provider rejected the request as malformed |
| 504 | `llm_timeout` | true | No answer within the per-call timeout (29.5 s) after the retry, or the deadline passed |

A 503 body carries `details.passages`: the three best search results (document, page, text), which are search results,
not an answer. Search runs locally, so they are there while the provider is down.

`GET /v1/answers/{answer_id}` (also the `request_id` of an error) returns the record of one question: the question,
scope, versions (model, prompt, passage assembly, index, search), coverage, the 30 fused candidates with their vector and
keyword ranks (`sent_to_llm` for the 8 the model read), each label and the unit it pointed to, every model call with its
raw output, verifier verdict, latency and tokens (`output_tokens` include thinking), the stages of the pipeline as spans,
totals (pipeline latency, tokens) and the body that was returned. It is written before the response is sent.

```bash
curl -s localhost:8000/v1/answers/b08ccba8-… -H "Authorization: Bearer $KEY"
# {"answer_id":"b08ccba8-…","question":"…","search_query":"…","document_ids":null,
#  "outcome":{"http_status":200,"status":"answered","reason":null,"error_code":null},
#  "versions":{"model":"gemini-3.8-flash","prompt":"g5-v3","assembly":"b2","index":"win250-head4|bge-m3-int8@5617a9f6",
#              "search":"rrf60-df0.3pct-stop"},
#  "candidates":[{"rank":1,"filename":"LB5001.pdf","page":2,"chunk_ordinal":0,"vec_rank":1,"lex_rank":1,"sent_to_llm":true,…},…],
#  "labels":{"S1":{"page":2,"pos":0,"kind":"heading","text":"# **Lubrication**",…},…},
#  "calls":[{"n":1,"purpose":"answer","finish_reason":"STOP","raw_output":"{…}","verifier":{"passed":true,"problems":[],
#            "added_labels":[]},"latency_ms":3146.3,"input_tokens":3533,"output_tokens":128,"thinking_tokens":0,…}],
#  "stages":[{"name":"answer.search","duration_ms":11.5,…},{"name":"gen_ai.call","duration_ms":3133.8,…},…],
#  "totals":{"latency_ms":3192.7,"model_calls":1,"input_tokens":3533,"output_tokens":128,"thinking_tokens":0},
#  "coverage":{…},"response":{…}}
```

## On-call queries

Spans (one per request, job and model call) are stored in Postgres. Open a shell with
`docker compose exec db psql -U docqa`:

```sql
SELECT * FROM v_model_calls_per_minute WHERE minute > now() - interval '30 min' ORDER BY minute DESC;
SELECT * FROM v_requests_per_minute WHERE minute > now() - interval '30 min' ORDER BY minute DESC;
SELECT name, duration_ms, status, error_code, attributes FROM spans WHERE request_id = '<X-Request-ID>' ORDER BY started_at;
SELECT kind, page_no, state, attempts, last_error_code FROM jobs WHERE document_id = '<document id>';
SELECT model, paused_until, pause_reason, pause_detail FROM provider_state WHERE paused_until > now();
SELECT index_version, state, count(*) FROM pages GROUP BY 1, 2;
SELECT started_at, duration_ms, attributes->>'docqa.pages' AS pages, attributes->>'docqa.chunks' AS chunks,
       attributes->>'docqa.peak_rss_mb' AS peak_rss_mb FROM spans WHERE name = 'index.document' ORDER BY started_at DESC LIMIT 20;
SELECT http_status, status, reason, error_code, count(*) FROM answers
WHERE created_at > now() - interval '1 hour' GROUP BY 1, 2, 3, 4 ORDER BY 5 DESC;
SELECT a.id, a.created_at, a.http_status, a.status, s.duration_ms, jsonb_array_length(a.calls) AS model_calls
FROM answers a JOIN spans s ON s.request_id = a.id AND s.name = 'answer.question'
WHERE a.created_at > now() - interval '1 hour' ORDER BY s.duration_ms DESC LIMIT 10;
```

The first answers "is the model provider down or slow?" (per stage, `ocr` included), the second "is the API failing or
slow, and on which route?", the fifth "is OCR paused, and why?", the next two "is everything indexed under the current
version, and how heavy was indexing?", and the last two "how are answers ending (declines, provider errors)?" and "which
answers were slow?"; `GET /v1/answers/{id}` then shows one of them in full.
Spans and logs never contain keys, question text, page text or file names.

## Eval

`make eval` brings the stack up, then runs the harness in the compose `eval` service (same image; no Python needed on
the host). The harness only talks to the API, like a client:

1. reads the question files (the fields of `eval/public_questions.jsonl`: `id`, `question`, `expected_status`,
   `expected_answer`, `sources`; optional `lang`) leniently: a UTF-8 BOM, CRLF and blank lines are fine; a line that is
   not valid JSON, lacks `id`, `question` or `expected_status` (or `expected_answer` when the status is not
   `not_found`), has an unknown status or repeats an id is skipped; a question whose `sources` name a file missing from
   the folder is asked anyway, its source shown as `file missing from folder` and its failure stage `ingestion`. Source
   files match the folder by base name, ignoring case and Unicode normalization. Every skipped or partly ignored line is
   listed in the report's **Input problems** section and on the console;
2. creates a fresh tenant, uploads every file of the folder as one batch (non-PDFs included) and waits for a final state
   (`EVAL_INGEST_TIMEOUT_S`; after it, the questions are asked anyway and the report says so);
3. asks each question (`--repeat N`, `--concurrency K`), reads its record (`GET /v1/answers/{id}`) and scores it;
4. writes `results.jsonl` (raw) and `report.md` under `eval/out/<UTC time>/` (or `OUT=<dir>`), and prints the report.

```bash
make eval                                                     # public questions on case_files/
make eval QUESTIONS="eval/public_questions.jsonl eval/questions_design.jsonl eval/questions_holdout.jsonl"
make eval PDFS=/path/to/pdfs QUESTIONS=/path/to/questions.jsonl ARGS="--no-judge"
```

The exit code is not a score: 0 when the run finished, 1 when the harness itself failed (API unreachable), and 2 only
when the run can't start (configuration, a missing folder or one with no readable file, no usable question). The
harness's `validate` command is strict instead: it lists every problem and exits 2 if a line would be skipped or a
source file is missing.

What the report measures, per question and in pipeline order:

| Metric | How |
|---|---|
| Status | 3×3 matrix (plus HTTP errors) |
| Content | Reference-guided judge, only when the status matches and is not `not_found`: the model lists the facts of `expected_answer` (essential or example) and marks each `stated`, `missing` or `contradicted`; the verdict is computed by code. Verdicts are cached by hash in `eval/out/judge_cache/`. `EVAL_JUDGE_MODEL` (default `gemini-3.5-flash-lite`) is provisional |
| Source | Some citation on an expected file and page |
| Quote in the PDF | Each quote looked up in the page's text layer, read by the eval with PyMuPDF (markup and whitespace ignored, rows cell by cell); transcribed pages counted apart |
| Grounded numbers | Every number of the answer found in its quotes or the question, with the verifier's locale readings |
| Retrieval | Rank of an expected page among the 30 candidates and whether it reached the 8 passages sent to the model |
| Failure stage | First failing stage: `ingestion`, `retrieval`, `ranking`, `provider`, `api`, `verification`, `generation` |
| Headline | **Correct** (status, content and source; `not_found` by status) and **unsafe** (answered when the documents don't answer, picked one case of an ambiguous question, or contradicted a fact), with **over-cautious** apart; per set and per `lang` |
| Cost and latency | Tokens from the answer records and `GET /v1/documents/{id}` times a dated price table, product apart from the judge; pipeline latency p50/max (p95 from 20 answers); pages served by the OCR cache |

Question sets: `eval/public_questions.jsonl` (given), `eval/questions_design.jsonl` (18 questions written while
designing the prompt and verifier) and `eval/questions_holdout.jsonl` (48 questions kept out of the prompt and verifier design: translations
of the public ones and questions in other languages; their retrieval was measured while choosing the search). Holdout
reference answers other than the translations are drafts written from the cited pages. `make eval-robustness` builds a scanned page, a locked PDF, a truncated
PDF, an all-blank PDF and non-PDF files with `eval/fixtures/make_fixtures.py` and runs `eval/fixtures/questions_robustness.jsonl` on
them; it runs apart from the scored sets so that a failed file never marks their coverage incomplete. The last report
is `eval/REPORT.md`.

`make eval-external` is optional and separate from `make eval`, which never reads its files. It tests generalization on
ten public manuals from outside `case_files/` (scans without a text layer, a 454-page catalog, manuals in German, Spanish,
French and Italian): `eval/external/fetch.py` downloads each one from its original URL into `eval/external/pdfs/`
(git-ignored; the PDFs are not committed) and checks its sha256 against `eval/external/manifest.json`, then the eval
runs on that folder with `eval/external/questions_external.jsonl` in its own tenant. A file that fails to download or
whose hash changed is listed and left out; the others go on. The first run costs up to about US$0.40, mostly OCR of the
scans. `make eval-external-fetch` only downloads. Details in `eval/external/README.md`.

