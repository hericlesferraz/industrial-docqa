# Decisions

Each entry covers the problem, the options considered, the choice and why, and what would make us change
our mind. Cuts and known limitations are listed at the end.

Entries 1 to 18 were written before the build, from prototypes. Where the build measured something new or
changed a detail, the entry says so in a "Measured in the build" note or in place. Every number is a
measurement unless it is marked as an estimate.

| # | Decision | Choice |
|---|---|---|
| 1 | Ingestion work granularity | `extract` per document, `ocr_page` per flagged page, `index` per document; pages searchable as they finish |
| 2 | Job queue technology | A Postgres table with our own claim and lease code; a job commits with the change that triggers it |
| 3 | Multi-tenancy | Shared tables with `tenant_id`, one API key per tenant, a single data-access layer |
| 4 | File storage | A named volume behind a `BlobStore` interface shaped for S3 |
| 5 | Page model | The page is the record (text layer and OCR side by side, three versions); chunks are a derived index; OCR cached across tenants by content |
| 6 | Unusable text and transcription | Per-font content statistics flag pages; `gemini-3.5-flash-lite` transcribes them, `gemini-3.8-flash` as fallback |
| 7 | Who writes the verbatim quote | The model points at numbered units; the quote is our stored text; a deterministic verifier grounds every number |
| 8 | Page reference in citations | Position in the file always; the printed label only when its neighbours confirm it |
| 9 | Language of answers and quotes | Answer in the question's language, quote verbatim in the page's, no translation |
| 10 | Retrieval | BGE-M3 int8 on CPU plus rare-term full text, fused with RRF; windows of about 250 tokens; no rewriting, no reranker |
| 11 | Follow-up after a clarification | Stateless; an ambiguous answer lists every case with its citation |
| 12 | Answer schema | One shape per status (`answered`, `ambiguous`, `not_found`), enforced by the provider's structured output |
| 13 | Upload format | One file per request, plus an optional batch with an outcome per file |
| 14 | Scoping a question to documents | Optional `document_ids`; rarity still counted over the whole tenant |
| 15 | Errors and provider failures | One error body with `retryable`; one code per provider failure; a 503 carries passages |
| 16 | Coverage in answers | A `coverage` block on every answer; pending and failed pages kept apart |
| 17 | When the verifier rejects an answer | Expand the evidence deterministically, retry once, then decline with passages |
| 18 | Answer model | `gemini-3.8-flash` with thinking `low`; Flash Lite by variable |
| 19 | Evaluation | Per-question checks in pipeline order; a reference-guided judge with the verdict computed by code; correct and unsafe reported apart |
| 20 | Telemetry and observability | Spans and answer records in Postgres, with OpenTelemetry GenAI names; no customer data in third-party tools |
| 21 | Build-time model quantization | ONNX Runtime's quantizer with the weights kept on disk: the same int8 file, peak RAM from 8.5 to 4 GB |
| 22 | Parsing untrusted PDFs | A sandboxed child process per document, with time, stall and memory limits |

## 1. Ingestion work granularity

**Problem.** A single upload can mix 2-page leaflets with 500-page catalogs. In the sample set, one manual
has about 50 of its 84 pages with an unusable text layer, so those pages need OCR: an external call that is
slow, costs money and is rate-limited. If the whole document is one unit of work:
- one failed page restarts the document;
- a large document holds a worker for minutes;
- the OCR rate limit can't be enforced across documents;
- nothing is searchable until the last page is done.

**Options considered.**
- One job per document, running every stage.
- One job per stage per document (extract, then OCR, then index).
- Mixed: extraction per document, OCR per page, indexing per document.
- Every stage per page.

**Choice: mixed.**
- `extract`, one per document: opens the PDF in a sandboxed subprocess with a timeout. It creates a row
  for *every* page up front, with the extracted text and the quality signals, and enqueues `ocr_page` for
  each page the detector flags.
- `ocr_page`, one per flagged page: transcribes the page and writes the text into that page's row.
- `index`, one per document, never more than one queued at a time: indexes every page that is ready and
  not yet indexed, then recomputes the document status from its pages (`ready`, `partial`, `failed`).

**Why.**
- Parallelism goes where the cost is. Text extraction takes milliseconds per page; OCR takes seconds and
  costs money.
- A failed page is retried on its own. The rest of the document stays searchable, and the document is
  reported as `partial` with the pending pages listed.
- The OCR concurrency limit and per-tenant fairness apply per page.
- A page becomes searchable as soon as its text exists, and answers report how much of the corpus was
  available.
- Every job can run twice with the same result, as at-least-once delivery requires. The `index` job
  doesn't need a completion counter: each run recomputes the status, and a page that finishes later
  (or fails for good) enqueues another run.

**Page order is preserved by construction.** Pages finish out of order: text pages in seconds, OCR pages
minutes later. So order never comes from completion time or insertion order.
- Every page row exists from the extraction step, keyed by its page number.
- OCR fills in the text of a row that already exists.
- Chunks carry (page number, position within the page), and every ordered read sorts by that key.

**Costs.**
- More jobs: about 8 per second on average at the production volume, still a small load for any queue.
- While pages are indexed independently, a chunk can't cross a page boundary, so a table that continues
  on the next page gets split. Revisited when chunking is decided.

**What would change our mind.**
- If extraction itself turns out slow on large documents (for example, table detection on 500 pages):
  split extraction into page ranges.
- If one call per page makes OCR inefficient (per-request pricing, or batching pages lowers cost): group
  a few pages per job.

## 2. Job queue technology

**Problem.** Decision 1 needs a queue with:
- leases and retries with backoff;
- concurrency limits per job kind;
- at most one queued `index` job per document;
- rate-limit responses that don't consume an attempt;
- fairness across tenants.

Each job must also be created in the same commit as the change that triggers it. For example, an OCR
result and the `index` job it enqueues must land together.

**Options considered.**
- A Postgres table with our own claim and lease code (`FOR UPDATE SKIP LOCKED`).
- Procrastinate, a Python queue library backed by Postgres.
- Redis 8 as both queue and vector store (Streams plus the query engine), with Postgres kept as the
  system of record.
- Celery, RQ or Dramatiq on Redis or RabbitMQ.
- A managed broker (SQS, Pub/Sub).

**Choice: a Postgres table with our own claim and lease code.** Postgres, with pgvector and full-text
search, also holds the documents, pages, chunks, vectors, answer records and telemetry.

**Why.**
- **The job commits with the state change.** With a separate broker, a crash between the two writes
  leaves a page that is never indexed, and fixing that needs an outbox or a reconciler.
- **One place to investigate (UC3).** Attempts, last error and timings can be joined in SQL with the
  document and its pages.
- **Three services in compose** (api, worker, db). That also means less RAM on the machine that runs it, which
  already loads the embedding model in two processes.
- **Our rules are a few lines on our own table:** the deduplicated `index` job, retrying a rate-limited
  call without spending an attempt, and per-tenant fairness. A library would need each of them verified
  or worked around. The MVP built the first two; jobs are still claimed in FIFO order (Cuts and known
  limitations).
- **The volume fits.** At the production volume this design runs about 8 jobs per second on average and about
  120 in bursts. Postgres queues built on `SKIP LOCKED` run in production at much higher rates: Solid
  Queue (the default in Rails 8), Oban, River, pg-boss.
- **Redis 8 was the strongest alternative.** It does hybrid search in the engine (`FT.HYBRID`, since 8.4)
  and fast in-memory vector search. But it doesn't replace Postgres as the system of record, so it adds a
  service and splits state across two stores. It also keeps vectors in RAM, the most expensive place for
  a long tail of rarely queried manuals (7 to 14 TB of vectors per year at production volume, depending on
  precision, from the chunk volume estimated in DESIGN.md). And retrieval
  isn't the bottleneck of a 4 s budget dominated by the LLM call.

**Risks and mitigations.**
- *Long transactions block vacuum.* The worker claims a job (setting a lease), commits, does the work,
  and records the result in a new transaction. No transaction stays open during OCR.
- *Table bloat.* Finished jobs are deleted or archived, and an index covers only pending jobs. If churn
  grows, the table gets partitioned.
- *At-least-once delivery.* A job whose lease expires runs again, so every job is idempotent (decision 1).
- *Bugs in our own claim code.* Tests cover concurrent claims, expired leases, retries and deduplication.

**What would change our mind.**
- The jobs table slowing regular queries (vacuum pressure, lock waits): first move the queue to its own
  database, then to a managed broker that autoscales on queue depth.
- Fan-out to many consumers, or delivery across regions: a broker.
- A frequently queried corpus small enough to fit in RAM, where retrieval latency matters: revisit Redis
  for the vectors.
- At scale, Redis is still likely to appear as a cache and as the global rate limiter for the LLM
  provider (see DESIGN.md).

## 3. Multi-tenancy

**Problem.** In the production scenario, 2,000 customers share the service. Each customer's manuals and answers must stay
invisible to the others. A technician at one plant must never get a passage from another company's
internal procedure: that is a confidentiality breach, and it describes the wrong equipment. The MVP doesn't
require tenants. But adding them later means changing every table, query, index and test, and migrating
the data already stored.

**Options considered.**
- No tenants in the MVP.
- Shared tables with a `tenant_id` column, one API key per tenant, and filtering in a single data-access
  layer.
- The same, plus Postgres Row-Level Security.
- One schema per tenant, or one database per tenant.
- A multi-tenancy library. The known Python ones (`django-tenants`, `django-multitenant`) are built for
  Django, and we use FastAPI.

**Choice: shared tables with `tenant_id`.**
- Every table that holds customer data carries `tenant_id`. That includes chunks: search queries them
  directly, so the filter needs no join, and the table can later be partitioned by tenant.
  The one exception is the OCR cache, which is keyed by file content (decision 5).
- The API key identifies the tenant. Keys are stored only as hashes.
- Every data-access function takes `tenant_id` as a required parameter, and no query is written outside
  that layer.
- An isolation test: tenant B uploads a document, tenant A asks about it and gets nothing. The same check
  covers listing and fetching documents.
- An admin endpoint protected by `ADMIN_API_KEY` creates tenants. `make eval` creates a fresh tenant for
  each run, so a report depends only on the folder it was given.

**Why.**
- It costs about an hour now. Adding it later would touch every part of the code.
- The same mechanism gives per-tenant fairness in the queue (decision 2) and per-tenant deletion. Later
  it also supports per-tenant cost, quotas and policy, such as a customer that doesn't allow external
  LLMs.
- Shared tables are the cheapest model to operate with 2,000 tenants. One schema or one database per
  tenant multiplies migrations, connections and vector indexes.
- No library: the mechanism is a FastAPI dependency plus required parameters, a few dozen lines. The
  available libraries are tied to Django, and to tenancy models we didn't choose.

**Risks.**
- *A forgotten filter leaks data.* The single access layer and the isolation test mitigate this.
  Row-Level Security is the next layer of defense (DESIGN.md).
- *A tenant-filtered vector search loses results when many tenants share one HNSW index.* The index
  returns the nearest neighbors across all tenants, and the filter then discards most of them. This
  doesn't show with one tenant. At scale, the fix is partitioning by tenant (DESIGN.md).

**What would change our mind.**
- A customer that requires physical isolation, for regulatory or contractual reasons: a dedicated
  database for that customer, with shared tables for everyone else.
- Filtering in the application proving error-prone in review: enable Row-Level Security in the MVP.

## 4. File storage

**Problem.** Uploaded PDFs (up to 500 pages, possibly over 100 MB) must be stored where both the API and
the worker can read them. Storage must not bloat Postgres, and a hostile PDF must never be parsed inside
the API. There also has to be a path to production, where many API and worker instances run behind a
load balancer.

**Options considered.**
- Inside Postgres (`bytea`).
- A named Docker volume shared by the api and the worker, behind a `BlobStore` interface.
- An S3-compatible service in compose, either behind the API or with presigned uploads. MinIO's
  community edition is no longer maintained: it has shipped as source only since October 2025, and its
  repository was archived in April 2026. SeaweedFS and Garage are maintained alternatives.

**Choice: a named volume behind a `BlobStore` interface, with the contract shaped for S3.**
- The key is `{tenant_id}/{sha256}.pdf`. The user's filename is stored only as metadata and is never
  part of a path.
- The database stores the key, never a filesystem path.
- `open(key)` is a context manager that yields a local file path. On the volume it does nothing extra; on
  S3 it downloads to a temp file. The sandboxed extractor needs a file anyway.
- Upload streams to a temp file inside the volume, so the final rename is atomic (same filesystem). While
  streaming, it computes the sha256 and checks the `%PDF-` magic bytes and the size limit, rejecting
  oversized uploads early with 413. It then renames the file and commits the document row and the
  `extract` job in one transaction.
- File before row: a crash leaves at most an orphan file, which a periodic cleanup removes. It never
  leaves a row that points to nothing.
- The API never parses the PDF. The `extract` job detects encryption and corruption, and reports them in
  the document status.
- The api and the worker run with the same UID.
- Storage settings are optional and have defaults: `BLOB_STORE=local`, `BLOB_STORE_PATH`,
  `MAX_UPLOAD_MB`.

**Why.**
- A clean clone has nothing to configure: no extra service, port or credentials. That is the lowest risk
  for `make eval` on a fresh machine.
- The storage choice is neutral for answer quality, document understanding and evaluation, so the time
  goes to those.
- An independent review classified the production work. Everything except the S3 implementation and
  presigned upload is needed whichever option we start from. Starting from the volume adds about three
  days to a production program measured in weeks. The real cost of presigned upload is the API contract
  (the client takes three steps), not the architecture.

**Known limitation.** The api and the worker must run on the same host, since the volume is local to it.
As is, it does not scale horizontally.

**Path to production.**
1. Stateless api and worker instances behind a load balancer, with state only in Postgres and object
   storage. A worker's local disk holds only scratch files for the job at hand.
2. As an optional bridge, a shared network filesystem mounted at the same path lets the current code run
   on several hosts without changes.
3. `S3BlobStore` (or GCS) behind the same interface, selected by `BLOB_STORE`.
4. Presigned uploads: the client uploads directly to the bucket. Hashing, validation and deduplication
   move to the worker, the size limit is enforced by a POST policy, and abandoned uploads expire.
5. The remaining items are described in DESIGN.md: page images as an intermediate artifact, per-tenant
   encryption keys, lifecycle tiers, reconciliation.

**What would change our mind.**
- Answer quality and the eval finished with more than two hours to spare: run an S3-compatible service
  (SeaweedFS, pinned) behind the API so the S3 path actually runs. A cheaper alternative is an
  `S3BlobStore` contract test against `moto`.
- Uploads of several GB, or the api and worker on separate hosts: object storage from the start.

## 5. Page model

**Problem.** Decision 1 creates a row for every page at extraction time, and pages finish out of order.
Each page needs a record of its text and where the text came from, why it was sent to OCR, whether it is
searchable yet, and which version of the pipeline produced it. Three later needs depend on that record:
- reprocessing after a bug fix without paying for OCR again;
- reporting a document as `partial`, with the pages that are missing;
- investigating a specific answer after its document has been reindexed (UC3).

**Options considered.**
- No page table: chunks only, with the page number as metadata. Ruled out by decision 1.
- A page row with a single text, overwritten by OCR.
- A page row that keeps the text layer and the OCR text side by side, plus which one is used.
- A page row plus a table of transcriptions, one row per source and version.

**Choice: the page is the record, and chunks are a derived index.**
- Each page keeps `layer_text` (always), `ocr_text` (when transcribed) and `text_source`. Whatever the
  source, the text uses one canonical format, so the chunker, the verifier and citations never depend on
  where it came from. The exact format (Markdown tables are the candidate) is settled with the extractor.
- Each page also keeps its quality signals and the reason it was flagged (`quality`), its printed label
  and language (filled in by later decisions), and a `text_rev` counter that increases on every new text.
- Page states: `needs_ocr`, `text_ready`, `indexed`, `blank`, `failed`.
- Document status is recomputed from its pages: `queued`, `processing`, `ready`, `partial`, `failed`.
  The last three are final, and that is where the eval stops waiting. Pages already indexed answer
  questions while the document is still `processing`.
- A page that fails for good also enqueues `index`. Otherwise a document whose last pending page fails
  would stay `processing` forever.
- Three versions, recorded on each page: extraction (extractor and detector), OCR (model, prompt, render
  resolution) and index (chunker and embedding model).

**Why.**
- Rebuilding the search index never reopens the PDF or pays for OCR again. The verifier looks for the
  quote in the whole page text, so a quote that spans two chunks is still found.
- Keeping the text layer next to the OCR text costs one column. It keeps the evidence for "why was this
  page sent to OCR?", lets us sample OCR against the text layer to calibrate the detector, and a detector
  change can't overwrite OCR we already paid for. A table of transcriptions would add history and room
  for a second transcription, through joins the MVP doesn't use. It is the next step if we ever
  transcribe pages twice.
- One version per stage, because OCR is the expensive stage: a chunker fix must not pay for 50 pages of
  OCR again.

**OCR results are shared across tenants, by content.**
- A cache keyed by (file sha256, page, OCR version) holds transcriptions. Before calling the model,
  `ocr_page` looks it up and copies the text into the tenant's own page row.
- The cache has no `tenant_id`, a deliberate exception to decision 3. It can only be reached by uploading
  the exact same bytes, and whoever has the bytes already has the content. The hash is always computed by
  us from the bytes received, never taken from the client. In 2011, Dropbox's deduplication trusted
  client-sent hashes, and a tool (Dropship) showed that knowing a file's hash was enough to obtain the
  file.
- The documents table already holds (tenant, sha256) under a unique constraint, so it is the list of
  references: tenant deletion (not built; Cuts and known limitations) can tell which entries no one
  references anymore. The build dropped the separate reference table planned here.
- What leaks is existence: an upload that finishes instantly tells the uploader that someone else has the
  same file. Byte-identical PDFs essentially only occur for publicly distributed files, because
  generation timestamps and internal IDs differ otherwise. That is also where reuse pays off: vendor
  manuals repeated across plants. A per-tenant opt-out is described in DESIGN.md.
- The API makes the leak plainer than timing does: `GET /v1/documents/{id}` reports `pages_from_cache`. We
  kept it in the MVP because the eval report and the on-call use it. At scale the count leaves the tenant's
  API and stays in telemetry (DESIGN.md, security).
- The eval report shows how many pages came from the cache. On a clean clone the cache is empty, so the
  first `make eval` exercises the real OCR path.

**Concurrent writes to a page.** Leases give at-least-once delivery, so an OCR job can run twice, and an
LLM doesn't transcribe the same page identically twice. Suppose `index` reads a page, a late OCR result
then replaces its text, and `index` marks the page `indexed`. The chunks would hold the old text forever,
and the verifier would reject correct quotes. Two guards:
- a job writes its result only if it still holds its lease (a fencing token, checked in the same
  `UPDATE`);
- `index` marks a page `indexed` only if `text_rev` still matches the value it read (optimistic
  concurrency).

Both are covered by tests.

**Chunks used by answers are kept.** Reindexing replaces a page's chunks. An answer record that stores
chunk ids would then point to rows that no longer exist, and an investigation could no longer show what
the model read. So reindexing moves the old chunks to `chunks_history`, with the same id, the same
`tenant_id` and no vector. A periodic cleanup deletes history rows that no answer references, once they
are older than a grace period longer than any answer takes. Checking at reindex time instead would race
with an answer still being generated, which records its chunks only after reindexing has run.
- Search reads only `chunks`. It needs no "active" filter, and the vector index carries no dead entries.
- History grows with reindexing and with what answers actually used, not with the size of the corpus or
  the number of questions. It expires with the answers.
- Also considered: copying the passages into each answer record. That repeats the same text once per
  answer (about 40 GB a day under our production assumptions) and scatters customer text that tenant
  deletion would then have to find.

**What would change our mind.**
- Continuous reindexing, or retention measured in years: store chunks as offsets into versioned page
  texts instead of copying their text.
- Investigations of old answers turning out rare in practice: shorten history retention.
- Transcribing each page twice to cross-check numbers: move texts into a table of transcriptions.
- Tenants with confidential documents that won't accept even the existence leak: opt them out of the
  shared cache, or stop sharing altogether.

## 6. Detecting unusable text and transcribing pages

**Problem.** A page's text layer can be missing (a scan), empty (a blank page) or present but wrong. In the
sample set, 53 of the 84 pages of one manual have a text layer that decodes to garbage: "Temperatura"
comes out as `7HPSHUDWXUD`, digits become control characters, and the oil-change table behind one of the
public questions loses all its numbers. A "does the page have text?" check passes on these pages. Unseen
documents of the same kind may be scanned, so the fix can't depend on this one file. Flagged pages need a
transcription that keeps every digit and the original wording, because answers quote it verbatim.

**What we measured** (CPU for detection; free-tier API keys and a local GPU for transcription):
- **Detector** on all 170 pages of the 4 manuals: per-font text statistics against labels.
- **Transcription** on two hard pages with known truth (an oil-change table in Portuguese, a dense
  relubrication leaflet in English), each as the original render, a synthetic office scan, and a
  synthetic bad scan (100 dpi, 2.5° skew, uneven lighting, heavy noise, JPEG quality 20). Then the same on
  two synthetic pages with the same layouts and random, non-monotonic values plus invented typos, so no
  model can answer from memory.
- Scoring: table cells, every number on the page, printed typos kept verbatim, word-level differences,
  latency, failures. Every model received the same PNG bytes for each page.

**Options considered.**
- Detection: trusting the font metadata (a CID font without a ToUnicode map), a language detector, OCR on
  every page, or content statistics per font.
- Transcription: Tesseract; a dedicated OCR service (Document AI); vision LLMs: Gemini 3.5 Flash Lite,
  Gemini 3.8 Flash, Gemma 4 26B-A4B and 31B through the same API, and Gemma 4 12B self-hosted.
- Pages transcribed by a model, in citations: mark them, or transcribe twice and compare.

**Choice.**

*Detector.* Content statistics per font, from the PDF's own spans:
- a font is garbage if, over at least 30 letters, its words of 4+ letters have under 25% vowels, or over
  20% of its letters are rare in Portuguese, English and Spanish, or over 5% of its characters are control
  characters, or over 5% are U+FFFD or private-use code points;
- a page goes to OCR when at least 20% of its letters come from garbage fonts;
- a page with under 50 letters goes to OCR if it has ink (scan, or text drawn as outlines); with no ink and
  no letters it is `blank`, and with a few letters and no ink it keeps them (a cover line, a header);
- the signals and the reason are stored in the page's `quality`, so "why was this page sent to OCR?" has
  an answer.

*Text-layer pages* are extracted with PyMuPDF's layout-aware Markdown export (`pymupdf4llm`, its own OCR
turned off), confirming Markdown as the canonical page format of decision 5.

*Transcription.*
- Default model `gemini-3.5-flash-lite`, pinned, one page per request, rendered by us at 150 dpi and sent
  with `media_resolution=HIGH` (about 1,100 image tokens).
- A page refused for recitation is first sent again to the same model, at most twice. The refusal is not
  deterministic: in the full run over one manual, the default model refused 2 of 53 pages, and both pages
  then passed on all 14 later attempts (identical requests, temperature 0, half pages).
- A page that still fails is retried once on `gemini-3.8-flash`: refused for recitation or safety, cut by
  the token limit, or out of attempts. The fallback model took 124 s on one of those pages and returned a
  503 on the other before passing.
- Both models are set by environment variables; any model of the same API works, Gemma included.
- The prompt asks for a verbatim Markdown transcription: no translation, no spelling fixes, no LaTeX,
  tables as tables, `[illegible]` instead of guessing.
- Only a `STOP` finish reason is accepted. Every other outcome is a failed page with its own error code.
  500 and 503 responses are retried a bounded number of times. A configuration error can produce a 500
  that is not transient, so retries must run out.

*Citations from transcribed pages are marked*, not double-checked. The citation says the text was
transcribed from the page image and that searching the PDF won't find it.

**Why.**
- **The detector** separated all 170 pages, including a second broken encoding in the same file ("Índice"
  extracted as `1ŶĚŝĐĞ`), which a fix tied to one encoding would have missed. No single signal is
  enough: vowels alone flag a table of flange codes on a healthy page (the closest healthy font scores
  0.28 against 0.23 for the worst broken one), and control characters alone miss pages whose broken font
  only draws underscores. The 4 manuals contain no real scan, so a rasterized page in a PDF is a test.
- **Text-layer extraction** keeps the row × column structure of the relubrication table and the reading
  order of a two-column page, which plain text extraction loses. It takes 0.1–0.4 s per page on CPU.
- **Flash Lite**: on the original and scanned pages, real and synthetic, it read every table cell (172 of
  172) and every number in 3–8 s per page (one outlier at 27 s), at about $0.002 per page. Flash read the
  same and was the most literal, but took 40 s to 6 minutes per page on the free tier and has a
  20-request daily quota there. The cheapest model that is right is the default, and the most literal one is the fallback.
- **Gemma through the API** read as well when it answered, but failed operationally: the 31B model
  answered 3 of 16 requests on the real pages (500 and 503 errors), the API rejects every setting that
  turns its thinking off, and the 26B model refused the English leaflet twice for recitation after about
  three minutes. A recitation refusal is not hypothetical, which is why the fallback exists.
- **Gemma 4 12B self-hosted** (4-bit, one 16 GB consumer GPU, 10.6 GB of VRAM, thinking off, temperature
  0) matched Flash Lite on clean and scanned pages: every cell, every printed typo kept on the real pages,
  10–23 s per page, and no customer data leaves the machine. Its image budget decides everything: capped
  at 512 tokens by the server's default batch size, the same model looped, misread a cell and inverted a
  condition ("at least 80°C" for "less than 80°C"). It is the self-hosted option in DESIGN.md, not the MVP
  default, because the evaluation runs on an API key.
- **Image resolution matters more than model size.** The same 12B model goes from looping and misreading
  to every cell right only by changing its image budget. Word rewrites mostly follow: at 512 tokens it
  rewrote "Hidrocarbons" (sic) as "Hydrocarbons" at temperature 0 and 1, with thinking on or off; at full
  resolution it kept it on the real page, though not on the synthetic one. Temperature made no difference.
- **Marking instead of transcribing twice.** On the bad scan of the real table, all four models that returned
  the full table misread the same cell the same way (2,500 read as 2,000), and none wrote `[illegible]`. Agreement between two
  models would have confirmed the error, at double the cost. On readable pages, the two Gemini models
  agreed on every number.
- **Tesseract** loses table structure and would need its own table pipeline. **Document AI** needs a
  service account and a GCP project, one more credential to set up. DESIGN.md weighs Document AI
  against a self-hosted model for refused pages.

**Measured in the build** (paid key; 55 pages: the 53 flagged pages of the sample manual, a cover and a
synthetic scan):
- **Cost:** $0.0013 per page on average (about 1,320 input and 355 output tokens), $0.0030 at most, for a
  dense table. On scanned manuals from other sources, pages of dense typed text cost $0.0021–0.0024
  (680–780 output tokens). The estimate above ($0.002) came from the free tier.
- **Time:** 2.8 s per call at the median, 6.2 s at most; the 53 pages in 74 s, with 3 calls at a time.
- **No LaTeX.** The prompt measured above didn't forbid it, and 24 LaTeX markers appeared in an earlier
  run. With the line added: 0 in the 53 transcriptions. The numbers match the earlier transcriptions on 49
  of the 53 pages; of the other 4, two had come from the fallback model, and two differ in list numbering.
- **No recitation refusal** in these 55 calls. On a public-domain scan, refusals were systematic instead
  (Cuts and known limitations).
- **Provider errors.** A 400 caused by one page (an image over the provider's size limit, for example) fails
  that page with `ocr_invalid_request` instead of pausing the model for every tenant. A daily quota, an
  exhausted balance or a rejected key pauses transcription for 15 minutes without spending attempts, and
  the document view shows the pause and its reason.

**Known limitations.**
- A badly scanned page with small print is transcribed with confident, plausible wrong numbers. On the
  synthetic bad scan at 7.5 pt, models got 0–10 of 28 cells right, with one `[illegible]` mark across
  five models. A
  deterministic legibility check (effective resolution, character height in pixels, sharpness) is the
  next step; it would mark the page low quality, warn in citations and send it to the strongest model.
- A transcription error passes the citation verifier, which compares quotes with the stored text, not with
  the image.
- Models normalize some words: an invented "lubrificaçâo" became "lubrificação" in every model, and the
  Gemma models also fixed invented typos in clean headings.
- A healthy page whose images contain text (labels in a figure, a table pasted as a picture) is not sent
  to OCR. In the sample set, such images hold only part labels and photos.
- The transcription sample is small: 4 pages in 3 versions. The eval report adds the pages of the 4
  manuals.

**What would change our mind.**
- The eval or production sampling showing digit errors on readable pages: transcribe table pages twice
  with different models, or move to a dedicated OCR service.
- Data-residency requirements: the self-hosted Gemma 12B, after a larger labeled comparison.
- Recitation refusals becoming frequent on the fallback too: a self-hosted model for those pages.
- The detector flagging healthy pages in non-Latin scripts: add a script check before the vowel signal.

## 7. Who writes the verbatim quote

**Problem.** Every answer cites a document, a page and what the page literally says, so a technician can
check it. If the model types the quote, the quote drifts from the page in ways the sample set already
shows:
- the model fixes the original's typos ("rearease", "kpas") without noticing;
- the stored page text is Markdown (`**`, `|`, `<br>`), which isn't what is printed;
- one manual prints a Unicode minus sign in "over−current", which a model retyping it would likely turn
  into a hyphen.

The quote also has to be small enough to check at a glance. A retrieved chunk of about 250 tokens is not a
quote.

**Options considered.**
- The model types the quote, and we accept it only if it appears verbatim in the page text. The cases
  above would reject correct answers.
- The model types the quote, and we locate the closest passage in the page by fuzzy matching, then return
  the stored original.
- The model generates the answer, and we pick the citation afterwards by similarity between the answer
  and the retrieved text.
- The text is split into numbered units, the model points at unit ids, and we return the stored text of
  those units.

**Choice: numbered units.**
- *Units.* A deterministic, versioned splitter divides page text into sentences, list items, headings and
  table rows. Chunks are built from whole units, so chunking (decided later) never cuts one. The splitter
  is part of the index version (decision 5).
- *Prompt.* At question time, the retrieved chunks are split into units, and each unit gets a short label
  valid only for that request (`S1`…`Sn`). A map from label to (chunk, unit position, page, document) is
  kept and stored with the answer record.
- *Model output.* Structured output with `status`, `citations`, `answer` and, for an ambiguous question,
  `options`, each option with its own condition, answer and citations. The citation fields are a closed
  list of the labels in that request, so the model can't produce an id that doesn't exist. `citations` comes
  before `answer`, so the model picks its evidence first and then writes from it.
- *Verifier* (deterministic, no extra model call):
  1. every cited label exists in the request's map;
  2. every number in the answer, and in each option, appears in the units it cites or in the question;
  3. at most 6 units per answer or per option, so citing everything can't pass check 2;
  4. a cited table row that carries a footnote marker (`*`) gets the note with the same marker on that
     page attached.
- *Quote.* The stored text of the cited units, in page order, one citation per page. Markdown markup is
  removed, table cells are joined with ` | `, and whitespace is collapsed; every other character comes
  from the stored text. A unit from a transcribed page is marked as such (decision 6).
- Only a `STOP` finish reason is accepted, and output that fails the schema gets its own error code. What
  happens when verification fails is decision 17.

**Number matching across locales.** "13.000 horas" in a Portuguese table and "13,000 hours" in an English
answer are the same number, but "1.800" can mean 1800 or 1.8. Each numeric token gets every reading its
format allows (thousands separator or decimal mark), and a number is grounded when any of its readings
matches any reading in the sources. This tolerates locale ambiguity and still rejects invented values.
Digits inside codes count ("IP55", "6205-2RS").

**Why.**
- **Verbatim by construction.** The model never types the quote. Typos, markup and unusual characters in
  the original can't make a correct citation fail or a quote differ from the page.
- **Fuzzy matching is unsafe on tables.** Rows of the same table differ only in their digits ("Over 210 to
  280 incl. (180) … 9500Hrs." and "Over 280 to 360 incl. (225) … 7400Hrs."), so the closest match to a
  slightly wrong quote can be a different row, silently.
- **Similarity can't pick the row.** One chunk of the relubrication leaflet holds both the interval table
  and the grease-quantity table. Only the model that reasoned over them knows which row it used.
- **Labels are cheaper to generate than quotes.** Each unit costs a few extra prompt tokens, but the
  output is shorter, and output tokens count for latency.
- **Measured on the public questions** (prototype, page texts from the extractor and from the
  transcriptions of decision 6). For the six questions that cite sources, every number in the expected
  answer is grounded in the units a correct answer would cite, including "9,500" against "9500Hrs." and
  "13,000" against "13.000 horas". The citations are 1 to 6 units out of 12–76 per page: one sentence for
  the submergence depth, a header and one row for the oil-change interval, the speed header, four rows
  and the footnote for the ambiguous regreasing question. Four planted wrong answers were
  all flagged: a converted unit ("48.8 m"), "24 meses", and two values taken from rows that weren't cited.
- Anthropic's citations feature uses the same pattern: documents are split into sentences, and the model
  points at them.

**Costs.**
- A quote is a whole sentence or row, never a fragment of one.
- The splitter will mis-split some abbreviations ("incl.", "Fig."). A bad split makes a quote longer or
  shorter, never wrong.
- Chunking is constrained to unit boundaries.

**Known limitations.**
- A grounded number is not proof of the right cell. "15,000 hours" for frame 256 at 1800 RPM passes,
  because 15000 sits in the same cited row, in another column. The eval measures cell choice.
- The model can cite a real unit that doesn't support its answer. The number check catches part of it.
- Numbers written as words are not checked.
- The verifier compares against stored text, not the page image (decision 6).

**What would change our mind.**
- Long sentences making the relevant phrase hard to spot: let the model also quote a phrase inside a cited
  unit, located by matching within that one unit only, where a near-duplicate can't be confused.
- The eval showing wrong-cell answers: for table answers, have the model name the row and the column
  header, and check the cell value deterministically.
- The cap of 6 units rejecting correct answers that need many rows: raise it, or make it per table.

## 8. Page reference in citations

**Problem.** A citation has to say where the passage is. In most manuals the number printed on a sheet
is not the page's position in the file:
- in one manual, PDF page 24 is printed "20", because the cover and contents are unnumbered;
- in another, PDF page 8 is printed "2-2", numbered by section;
- the two-page leaflet prints no number at all.

The text itself uses printed numbers. The oil-change table says "figure 1, page 2 of this manual". Figure
1 is on PDF page 6, printed "2", and PDF page 2 is blank. A citation can carry three things:
- the position in the file;
- the printed number;
- the passage's coordinates on the page, so a viewer can highlight it.

**Options considered.**
- Position only.
- Position plus the page labels the PDF declares (`/PageLabels`).
- Position plus the printed number, read from the page margins.
- Any of these plus the coordinates of the quoted passage.

**Choice.**
- `page` is the 1-based position in the file. It is always present and is the reference the system relies
  on.
- `page_label` is the printed number when it is confirmed, and `null` otherwise:
  - a candidate is a line in the top or bottom 10% of the page shaped like a page number (`20`, `A3`,
    `2-2`, `ii`);
  - it is accepted only if a page at most two away has a candidate in the same series at the matching
    distance;
  - extraction computes it and stores it in the page's label field (decision 5);
  - on a transcribed page, the same rule runs on the first and last lines of the transcription, which
    keeps page numbers on their own line.
- The detector is the last ingestion item to be built. The field is nullable from the start, so leaving the
  detector out changes nothing else.
- No coordinates in the MVP.

**Why.**
- **The position is always known and always right.** Any PDF that opens has its pages in order, scanned
  or not, and every chunk carries its page from extraction. It is what the eval checks, and what
  `file.pdf#page=N` opens in common PDF viewers.
- **The declared page labels are wrong.** The two manuals that have them give "A20" for the sheet printed
  "20" and "2" for the sheet printed "2-2".
- **A printed number is confirmed by its sequence.** A lone number in a margin can be a figure label or a
  part number; only page numbers step by one from page to page. On the 147 pages whose numbering we know,
  the detector returned 139 right labels, none wrong, and 8 `null` (in the build, with the edge lines of
  transcribed pages added: 142, none wrong, 5 `null`). The neighbour check rejected one
  plausible but wrong label, and picked the page number out of five candidates on a page with a figure
  near its footer. The section-numbered manual gets "1-1", "2-2", "ii"; the leaflet gets `null`.
- **A wrong label is worse than none**, the same principle as answers: the technician would trust it.
- **Coordinates depend on where the text came from, not on whether the page is a scan.** A healthy text
  layer gives the rectangles of a quoted sentence or a table row. A transcribed page doesn't: vision
  transcription returns text without positions, and on the broken pages of the sample (not scans) the
  text layer is garbage, so even searching it finds nothing. The MVP has no user interface, and the API
  never opens the PDF (decision 4), so coordinates would have to be stored at extraction for a feature
  nobody can see yet.

**Known limitations.**
- Page-number formats outside the patterns ("Page 3 of 10", "- 3 -", "3/10") give `null`, as do
  single-page documents, which have no neighbour to confirm against.
- Labels on transcribed pages were checked on only two pages.
- A citation opens the right page but doesn't highlight the passage.

**What would change our mind.**
- New documents using other numbering formats: add patterns. The neighbour check keeps new patterns
  safe.
- Users who never use printed manuals and find two numbers confusing: drop `page_label`.
- A user interface: store word coordinates at extraction for text-layer pages, and serve the page image
  with the passage marked, reusing the page images described in DESIGN.md. Highlighting on transcribed
  pages needs an OCR that returns word positions.

## 9. Language of answers and quotes

**Problem.** A technician asks in their own language, and the manual may be written in another. One
public question asks in English about a manual that exists only in Portuguese, and another manual repeats
every page in three languages. The answer has to be understood, and the quote has to be checkable against
the page.

**Options considered.**
- Answer in the question's language, quote in the document's language, no translation.
- The same, plus a translation of the quote generated in the same model call and marked as a translation.
- The same, plus a translation from a second, separate call.
- Translate the quote and show only the translation.

**Choice.**
- The answer is written in the language of the question.
- The quote is always verbatim, in the language of the page (decision 7).
- No translation of quotes in the MVP.
- No `language` field on citations in the MVP. A reader sees the quote's language by reading it, and no
  code consumes the field. It comes back with a page language detector at extraction, if a client needs to
  label quotes or offer a translation; adding a field later breaks no client.

**Why.**
- **The quote is evidence, and a translation is not what the page says.** Showing only a translation
  would break the verbatim requirement outright.
- **A translation from the same call verifies nothing.** If the model misread "less than 80 °C" as "at
  least 80 °C", the kind of inversion seen in decision 6, its translation repeats the misreading.
  It would look like a second source without being one.
- **The riskiest content needs no translation.** Numbers, codes and units ("IP55", "13.000 horas") read the
  same in any language, and decision 7 already checks every number in the answer against the quote.
- **The answer already carries the meaning** in the technician's language.
- **Latency and cost.** A translation adds output tokens to every cross-language answer, or a second call
  in sequence. Questions dominate the cost at production scale, and the p95 budget is 4 s.

**Known limitations.**
- A technician who doesn't read the document's language can check numbers and codes in the quote, but
  relies on the answer for the meaning of the surrounding words.

**What would change our mind.**
- Technicians unable to relate an answer to a foreign-language quote: translation on demand, from a
  separate endpoint outside the answer path. It would be marked as machine translation and never replace
  the quote, and its numbers would be checked against the quote with the same verifier.

## 10. Retrieval

**Problem.** A question has to reach the page that answers it before the model can answer and cite. In
the sample set, the question and the manual often don't share a language: an English question about a
Portuguese-only brochure, a Portuguese question about an English leaflet. Answers sit in tables whose rows
differ only in their numbers. A part code ("8040MDXYZ", "W3Seal") appears once in the whole corpus. One
manual repeats every page in three languages. A page that isn't retrieved turns into a false "not found",
and the p95 budget of 4 s leaves room for a single model call per question.

**What we measured** (CPU only: Ryzen 7 5700X, 8 threads; free-tier API key):
- **Corpus:** all 170 pages as ingestion stores them. Text-layer pages come from the extractor, and the 54
  flagged pages are transcribed (decision 6). That makes 533 chunks.
- **54 questions,** each with a known expected page; any copy of a repeated page counts:
  - the 6 public questions that have sources, each asked in English, Portuguese, Spanish and German (a
    language no manual uses);
  - 23 questions written by us over the four manuals, each asked in a language other than its page's;
  - 7 questions built around a code.
- **8 needle queries** that are only a code or a product name, such as "W3Seal" or "330k".
- We wrote the 30 extra questions after reading the pages. We paraphrased and translated them to limit
  word overlap with the page, but the bias remains.
- **Metric:** recall@k, the share of questions whose expected page appears among the first k chunks; and
  how many tokens come before the first right chunk.

**Options considered.**
- Chunks: the whole page, or windows of about 250 tokens built from the units of decision 7.
- Embeddings: multilingual-e5 small and base, BGE-M3 and Qwen3-Embedding-0.6B locally; Gemini Embedding 2
  through the API.
- Keyword search: none; every word of the question; only its rare terms.
- Query rewriting: an LLM translates the question into the corpus languages before the search.
- A reranker over the top candidates.

**Choice.**
- **Chunks.** Windows of up to about 250 tokens of whole units, never crossing a page.
  - A table, its caption and its footnotes form one block. The block is split by rows only when it alone
    exceeds the budget, and every piece repeats the table's first row. Since the build, the pieces after the
    first also carry the table's column names among the units the model reads, not in the text the search
    reads (Measured in the build).
  - Each chunk starts with the file name and the nearest heading above it.
- **Vectors.** BGE-M3 (1024 dimensions), run locally on CPU through ONNX Runtime with int8 weights, is the
  default. The weights are BAAI's official ONNX export, quantized by us with ONNX Runtime's dynamic int8
  quantization, so the runtime needs no torch. `EMBEDDING_PROVIDER=gemini` switches to Gemini Embedding 2,
  also at 1024 dimensions, in the same column. Three rules keep this cheap:
  - one provider per deployment, because vectors from different models are not comparable and are never
    searched together;
  - the embedding model is part of the index version (decision 5), so switching providers reindexes the
    pages without paying for OCR again;
  - both providers were measured once on the retrieval set in the build (Results); `make eval` runs the
    default.
- **Keywords, rare terms only.** Postgres full text (`simple` configuration plus `unaccent`) searches only
  the query terms that are rare in the tenant. The rule measured below was "in at most 3 of the tenant's
  chunks". Since the build, a term is rare when it occurs in 1 to max(3, 0.3% of the tenant's chunks) chunks
  and is not a function word of English, Portuguese, Spanish, German, French or Italian (Measured in the
  build). The terms are OR-ed and ranked with `ts_rank`. Each term is counted on the GIN index, up to one
  more than the bound.
- **Fusion.** Reciprocal rank fusion (RRF, k = 60) merges the two lists. Each list gives a chunk
  1 / (60 + its rank), and the scores add up. The top 8 chunks, about 2,000 tokens, go to the model.
- **No query rewriting and no reranker.**

**Results** (top 8 out of 54 questions; needles in the top 8 out of 8):

| Configuration | Questions | Needles |
|---|---|---|
| Postgres `websearch_to_tsquery` (every word required) | 0 | 8 |
| BM25, all words | 36 | 8 |
| e5-small / e5-base | 41 / 47 | 3 / 2 |
| Qwen3-Embedding-0.6B | 52 | 3 |
| Gemini Embedding 2 | 53 | 6 |
| BGE-M3 | 54 | 4 |
| BGE-M3 + BM25, all words | 48 | 8 |
| BGE-M3 + rare terms (plain Postgres) | 54 | 8 |
| **BGE-M3 int8 (ONNX) + rare terms** | **54** | **8** |
| Gemini Embedding 2 + rare terms | 54 | 8 |
| e5-base + BM25 on an LLM-rewritten query | 50 | 8 |

Gemini Embedding 2 was measured at 768 dimensions while choosing. The build measured both providers again
in Postgres, through the product's own search with the 3-chunk rule: BGE-M3 int8 54 of 54 and 8 of 8, with
the right page first on 42 questions; Gemini Embedding 2 at 1024 dimensions 54 of 54 and 8 of 8, first on
41. With the current rule, BGE-M3 puts the right page first on 46; Gemini wasn't measured again.

**Why.**
- **Vectors carry the cross-language search.** BGE-M3 found the page for every question in all four
  languages, German included. The e5 models miss Portuguese and Spanish questions about English manuals.
- **Every embedding model misses codes.** Alone, BGE-M3 ranked "W3Seal" 26th and didn't return "330k" in
  its top 50; keyword search put all 8 needles first.
- **Keywords must be rare terms only.**
  - Searching every word hurts cross-language questions. The common words of a Portuguese question match
    the Portuguese manuals, and RRF gives that noise the same weight as the vector list. BGE-M3 dropped
    from 54 to 48.
  - Rare terms keep their selectivity in any language, so the fusion keeps 54 of 54 and finds every needle
    first.
  - Postgres's `ts_rank` has no inverse document frequency: a common word counts as much as a rare one.
    That is why every-word Postgres search did worse than BM25. With only rare terms that no longer
    matters, and plain Postgres matched BM25. So no BM25 extension and no extra image.
- **Rewriting the question** reached the same 54 of 54 at +1.2 s median latency (6 s at worst) and one
  more model call per question. It only paid off with the weaker e5 (47 to 50). It would also make search
  depend on the provider: with the provider down, the API could no longer return the most relevant passages
  with its 503.
- **A reranker has nothing to fix.** The expected page is already in the top 8 for every question, and a
  reranker only reorders what search returned.
  - A cross-encoder the size of BGE-M3 takes about 0.4 s per chunk on CPU: about 3 s for 8 chunks and
    12 s for 30, beyond the p95 budget.
  - A reranker through an API is one more key and one more third party reading customer text.
  - The answering model already picks units among the 8 chunks (decision 7).
- **Windows rather than whole pages.** Recall is the same, but the first right chunk arrives after a
  median of 244 tokens, against 621–788 for whole pages. Within a 2,000-token budget, windows find 54 of
  54 and pages 49. e5's 512-token limit truncated 87 of the 166 pages.
- **int8 rather than full precision.** Quantization moves the vectors a little (cosine 0.977 median,
  0.967 minimum, against the original model) but keeps the ranking: 54 of 54 and 8 of 8, as before. On
  the same CPU, compared with the original model under torch:
  - 17 ms per query instead of 78, and 263 ms per chunk instead of 420 (in batches of 8; see Costs for one
    text per call);
  - 0.9 GB of RAM for a process that only embeds questions, instead of 1.9 GB;
  - 569 MB of weights instead of 2.27 GB, and no torch in the image.

  BAAI's full-precision ONNX export gives vectors identical to the original (cosine 1.0000), so it can
  replace the int8 file in the same runtime.
- **BGE-M3 rather than Gemini at equal recall:**
  - the corpus doesn't leave at ingestion; only the passages retrieved for each question do;
  - search keeps working when the provider is down;
  - no daily quota (1,000 requests on the free tier);
  - no reindexing forced by the retirement of a hosted model;
  - 17 ms per query, against 412–697 ms on the free tier (431 ms at the median on the paid key, in the
    build).

**Costs** (measured in the build, same CPU).
- **One text per call.** ONNX Runtime's dynamic int8 quantizes the input of each matrix product with one
  range for the whole batch, padding included. A text's vector therefore depends on its batch neighbours:
  cosine 0.976–0.984 between a batch of 1 and a batch of 8. The worker embeds one chunk per call, as the API
  does with a question, so a passage and a question are embedded the same way. On this CPU it was also
  faster and lighter: 130 ms per chunk against 229 ms. A batch size other than 1 becomes part of the index
  version.
- 133 ms per chunk at ingestion on average (about 70 s for the 535 chunks of the sample set), 17 ms per
  query (31 ms on 2 threads). A search, question embedding included, takes 23–32 ms at the median (two
  runs).
- RAM: 0.93 GB in the API; 1.2–1.4 GB in the worker while it indexes, and a 1.7 GB peak for the worker's
  container with a PDF extraction running beside it. The 2.25 GB peak measured while choosing came from
  batches of 8.
- Part of the cosine of 0.977 against the original model may be the same batch effect, since those
  vectors were computed in batches. Not measured.
- The image grows by 569 MB of weights plus ONNX Runtime.

**Measured in the build: the rare-term rule and split tables.** On 10 external manuals (3,095 chunks, 15
questions), the 3-chunk rule failed in two ways. Useful terms were too frequent to count: "M16" occurs in 9
chunks. And function words counted as rare when the corpus barely uses their language: Portuguese "em"
occurs once, in an English glossary, which then took a prompt slot in five Portuguese questions. We
measured six rules, search only (no model call), on the 4 sample manuals (535 chunks, the 54 questions and 8
needles above) and on the external set (rank of the expected page among the 30 fused candidates):

| Rule | Samples: top 8 / first / MRR | Needles in top 8 | External: top 8 / first / MRR |
|---|---|---|---|
| df ≤ 3 (before) | 54 / 42 / 0.881 | 8 | 11 / 4 / 0.454 |
| df ≤ max(3, 0.3% of chunks) | 54 / 42 / 0.881 | 8 | 12 / 5 / 0.517 |
| df ≤ max(3, 0.5% of chunks) | 54 / 42 / 0.881 | 8 | 12 / 3 / 0.444 |
| function words + df ≤ 3 | 54 / 46 / 0.918 | 8 | 11 / 5 / 0.498 |
| **function words + df ≤ max(3, 0.3%)** | **54 / 46 / 0.918** | **8** | **12 / 5 / 0.517** |
| function words + df ≤ max(3, 0.5%) | 54 / 46 / 0.918 | 8 | 12 / 4 / 0.478 |

- **Adopted: function words plus 0.3%.** The bound is 3 at 535 chunks, 9 at 3,095 and 300 at 100,000. On
  the external set, the torque question went from 13th to 1st and the taper question from 5th to 1st, and
  the glossary left four of the five Portuguese questions (the fifth is about that manual). On the samples,
  only four German questions moved, from 2nd to 1st: no manual is in German, so "wie" and "und" were rare.
- **The external set is no longer held out.** The rule was written after reading its failures, and 0.3%,
  proposed in a review before measuring, was kept over 0.5% on its numbers. Its generalization figures are
  the ones before: MRR 0.43 in the eval run, 0.45 in this search-only rerun. The 0.52 after is tuned on it.
- **Cost:** one count of the tenant's chunks per question (an index-only scan), and a per-term count that
  grows with the bound. Search alone, after a vacuum: 7.6 ms at the median on the samples, 11.8 ms on the
  external set.
- Every answer record names its rule (`rrf60-df0.3pct-stop`; records written before say `rrf60-df3`).
- **Split tables.** When a table's first row is a title the extractor split, the later pieces reached the
  model without column names, and a question on a bearing clearance table came back `not_found`. Repeating
  the table's head (its header rows, plus the rows before the first mostly numeric row when that row comes
  within the first 4; only when the head fits in half the budget) in each piece's text and units dropped the
  right chunk from 1st to 22nd: the pieces looked alike to the embedding, and the column words stopped being
  rare. Kept: the head only in the units the model reads, with the searched text unchanged, so search
  results stayed the same (samples 54 / 46 / 0.918, external 12 / 5 / 0.52). The split title is still the
  only table context the search reads in those pieces. Chunker version `win250-head4`.

**Known limitations.**
- **Translations of the same page fill part of the window.** For questions about the trilingual manual,
  about 3 of the top 8 chunks repeat content already listed, and up to 5. We tried collapsing chunks with
  near-identical vectors and the same numbers, which only brought this down to 1.65, because chunks don't
  align across languages. Recall was unaffected. Grouping equivalent pages at ingestion is in DESIGN.md.
- **The rare-term bound was validated only between 535 and 3,095 chunks,** and its 0.3% was chosen on the
  external set. A code in more than 0.3% of a tenant's chunks (a catalog of one product family) doesn't
  count as rare, and only the vector search can find it.
- **The sample is small:** 62 queries, 30 of them written by us.
- A table split across two pages becomes two chunks.
- int8 speed was measured on one x86 CPU with AVX2. Under emulated `linux/arm64` (QEMU), the same int8 file
  gives slightly different vectors (cosine 0.972 to 0.987 against x86; the same top-1 and the same five nearest
  neighbours on the 7 public questions), so an index built on one architecture shouldn't be served on the other;
  the index version doesn't record the architecture. Speed on Apple Silicon wasn't measured.
- int8 and full-precision vectors are close but not identical. They count as different models in the
  index version and are never mixed.
- `sentence-transformers` loaded Qwen3 in bfloat16, which is slow on CPU (4 s per chunk). We didn't measure
  it in float32. It ranked below BGE-M3, so that didn't change the choice.

**What would change our mind.**
- RAM still too high in a typical Docker Desktop VM: Gemini through the flag (the worker already embeds one
  text at a time).
- The eval showing int8 misses that full precision gets right: switch to the official full-precision
  export, in the same runtime, and reindex.
- The eval showing the right page retrieved, but outside the top 8: first raise k, then consider a
  reranker. Each answer record keeps the rank of every chunk, so the report separates "not retrieved" from
  "retrieved too low".
- Cross-language misses in the eval: rewrite the question only when the best score falls below a
  threshold, not on every question.
- Codes in more than 0.3% of a tenant's chunks going unfound: keep the question's k rarest terms instead of
  a bound.
- Questions in a language outside the six lists bringing function-word noise back: a longer list, or
  language detection.
- Tenants with millions of chunks: keep the chunk count per tenant instead of counting on every question.

## 11. Follow-up after a clarification

**Problem.** When an answer depends on something the technician didn't say, the system returns
`ambiguous` instead of picking one case. The natural next step is the technician supplying the missing
value ("frame 256, 1800 RPM"), and the API has to say how that step works. A follow-up such as "and at
3600?" retrieves nothing on its own, so a conversation would have to turn each turn into a standalone
question before the search.

**What the requirements imply.**
- They ask the system to "make that clear instead of picking one", and list "needs clarification" as one
  of three outcomes of an answer, next to answered and not found.
- The expected answer for the ambiguous public question accepts asking for *or* enumerating the frame size
  and speed, and it quotes the range of values.
- The eval runs one question per line with no history, and any new question set uses the same format.

**Options considered.**
- Stateless: every question stands alone. The ambiguous answer names what is missing and lists every case
  with its own citation; to narrow it down, the client asks the full question again.
- Follow-up by reference: the question carries `follow_up_of` (a previous answer id), and the server joins
  the stored original question with the reply before the search.
- A conversation: `conversation_id`, history, any number of turns.

**Choice: stateless.**
- An `ambiguous` answer says what the answer depends on and lists the cases, each with its condition, its
  answer and its citations (decision 12).
- The server never combines two questions.

Four properties keep a conversation cheap to add later:
1. every response returns an `answer_id` (the request id);
2. the answer record stores the question, the ids and ranks of the retrieved chunks, and the cited units;
3. the request body is a JSON object, so new fields don't break clients;
4. the answer record keeps `question` (what was typed) apart from `search_query` (what the search
   received). They are equal today.

**Why.**
- An ambiguous answer that lists every case with its citation is useful in a single round: the technician
  usually finds their row in it. It also covers both readings of the expected answer, asking for the
  parameters and enumerating them.
- There is no conversation state to store, expire and isolate per tenant, and any API instance can answer
  any question.
- The real cost of a conversation is in retrieval, not in the API. A turn like "and at 3600?" must be
  rewritten into a standalone question, which is the query rewriting measured in decision 10 (+1.2 s
  median, one more model call). That cost is the same now or later, and the eval wouldn't measure it.
- Adding a conversation later only adds things: an optional request field, a nullable
  `parent_answer_id` column, the previous turn in the prompt, the verifier accepting numbers from any
  question in the chain, and a `gen_ai.conversation.id` attribute on spans. With the fourth property, a
  follow-up merged wrongly shows up in the record as its own failure stage, instead of passing for a
  retrieval miss.

**What would change our mind.**
- Technicians replying with fragments ("256, 1800"): add `follow_up_of`. For a clarification, joining the
  original question with the reply and reusing the chunks the first answer retrieved would probably avoid
  a rewrite call (not measured).
- A chat interface: a conversation with history, rewriting a turn only when it doesn't stand alone.

## 12. Answer schema

**Problem.** The model's output has to carry three things:
- one of three outcomes: answered, couldn't find, needs clarification;
- the evidence for it, as unit labels (decision 7);
- for an ambiguous question, every case with its own evidence (decision 11).

Two errors pull against each other. One is picking a case silently, which the requirements forbid. The other
is asking back about a question the documents already answer, which leaves a technician next to a machine
without an answer. A field present in every response may push the model toward filling it, and so toward
the second error.

**Options considered.**
- A flat schema with every field in every response: status, citations, answer, a list of what is missing,
  and options.
- The same, with each missing item typed (parameter, interpretation, equipment, conflict, other).
- One shape per status, a union keyed by `status`, with no list of missing items.
- The union, with the list of missing items in the ambiguous shape.

**Choice: one shape per status.** The status values are those of the eval file.
- `answered`: `citations` (1 to 6 labels), then `answer`.
- `ambiguous`: `depends_on` (one sentence: what the answer depends on), then `options`, 2 to 5, each with
  `condition`, `citations` (1 to 6) and `answer`.
- `not_found`: `answer`, saying what the documents don't state.

The rules that come with it:
- **Verification.** The verifier checks each option like an answer. `depends_on` has no citations of its
  own, so its numbers are checked against the citations of all the options together.
- **An answer with a condition is a prompt rule, not a field.** When the document gives a default and an
  exception that the question doesn't rule out, the answer states the default, adds the exception in one
  sentence, and cites both.
- **No derived values.** The model writes "half that interval", never the halved number, because the
  verifier rejects a number that no unit contains.
- **Documents that disagree** are `ambiguous`, with one option per document. There is no fourth status,
  because the eval knows three.
- **The schema sent to Gemini** is an `anyOf` of three objects. In each one, `status` is a single-value
  enum, and citation items are an enum of the request's labels. The output is validated again on our side.
- **The API response is a separate object.** It adds the answer id, the expanded citations and the
  coverage. It returns `depends_on` in its `answer` field, so clients always read the same field.

**Why.**
- **A shape that can't be produced can't be wrong.** Generation rejects an answered response with options,
  or an ambiguous one with a single option. No coherence code is needed.
- **The answered shape has no optional field to fill.** That removes the schema's push toward asking back.
  The rest is the prompt's rule, which the eval measures by reporting wrong statuses in each direction
  separately.
- **A list of missing parameters has no consumer**, typed or not. There is no user interface and no
  follow-up mechanism (decision 11), and the eval compares text. The sentence in `depends_on` gives the
  reader the same information. The typed list was also fragile: checking it against the sample manuals
  found a missing kind at once ("frame 180", read as NEMA or IEC).
- **The summary comes before the options.** The model states the axis first, and the options stay on it:
  one per frame row, not a mix of axes. The name also steers the model: a field called `answer` right after
  "ambiguous" invites it to pick a case.
- **At most 5 options.** For the ambiguous public question, one option per frame row, with the speeds
  inside, costs 600 to 900 output tokens in the eval; one per cell would cost more. Output tokens drive
  latency.
- **Status comes first, before the evidence.** Decision 7's order, evidence before text, still holds
  inside each shape, and the status is decided from the prompt either way. The cost: once "answered" is
  written, the model must cite something and can't fall back to not found.
- **An independent review reached the same shape.** A review given only the requirements, the eval file
  and decisions 7 to 11 proposed the summary-first order and the cap.

**Measured.** One call to `gemini-3.5-flash-lite` on 30 Oct accepted the union at the root of the schema.
It returned a valid `ambiguous` output in 2.0 s, with 2,378 input and 446 output tokens.

**Known limitations.**
- A two-part question with one part unanswered comes back `answered`, with the gap stated in the text.
- Whether a case is a default with an exception or an ambiguity is the model's judgment. The schema
  doesn't decide it.

**What would change our mind.**
- The union failing on another model or on another provider key: a flat schema, with a coherence check in
  the verifier. The API response wouldn't change.
- Questions with a default and an exception coming back ambiguous: an explicit caveat field in the
  answered shape.
- Statuses measurably more accurate with citations before the status: reorder the fields.
- A user interface that acts differently per kind of ambiguity: add a typed list of what is missing.

## 13. Upload format

**Problem.** A reliability engineer onboards a site in one go: dozens of manuals, from a 2-page leaflet to
a 500-page catalog of over 100 MB, some of them corrupted, password-protected or not PDFs at all. One
broken file must not ruin the rest. The engineer needs to see how the whole onboarding went, and the eval
needs to know when ingestion is done.

**Options considered.**
- One multipart request carrying every file, with a result per file.
- One file per request, with no grouping.
- One file per request, plus an optional batch that groups them.
- Presigned upload straight to object storage, the production path of decision 4.

**Choice: one file per request, plus an optional batch.**
- `POST /v1/batches` creates a batch. `POST /v1/documents` takes one file, optionally with a `batch_id`.
- Each file sent with a `batch_id` becomes a batch item, with its filename and an outcome:
  - `accepted`: a new document;
  - `duplicate`: points to the document that already existed;
  - `rejected`: not a PDF, or too large, with an error code.
- `GET /v1/batches/{id}` returns every item with its document's status, and the counts.
- **Duplicates** are found per file, by the sha256 we compute while the file streams in, whether it comes
  alone or in a batch.
  - A unique constraint on (tenant, sha256) settles two concurrent uploads of the same file.
  - A duplicate returns the existing document as it is, with `200` and `duplicate: true`. A new document
    returns `202`.
  - The same bytes from another tenant become a separate document (decision 3), but OCR isn't paid
    again, because the transcription cache is keyed by content (decision 5).

**Why.**
- **A dropped connection loses one file, not the onboarding.** Sending the file again is safe, because
  the upload is idempotent by hash. With a single multipart request, a drop halfway through a 2 GB body
  also loses the response, so the client doesn't know which files were stored.
- **The size limit is checked before reading the body.** With one file per request, `Content-Length` is
  the file's size, so an oversized file gets a 413 at once (decision 4). In a multipart request with many
  files it is the sum, and a file over the limit is only rejected after the whole body has arrived.
- **Rejected files stay visible.** A file that isn't a PDF never becomes a document, so without batch
  items it would vanish from the onboarding view.
- **Same shape as production.** Presigned uploads also give one URL per file.
- **The eval waits on one resource.** `make eval` creates a batch, uploads the folder, and polls the batch
  until every document reaches a final state or a timeout.
- **The cost is a loop on the client side.** `make eval` already has it, and the README shows it as three
  lines of shell.

**Known limitations.**
- The same manual with different bytes, exported again or in another revision, is a new document. If the
  two disagree, answers come back ambiguous with one option per document (decision 12). Detecting
  near-duplicates from page text is in DESIGN.md.
- Sending the same bytes again doesn't reprocess a failed document. Reprocessing after a fix goes through
  the pipeline versions of decision 5.

**What would change our mind.**
- Clients that can only send one request: also accept several files in one multipart request, as a
  convenience that creates the batch items the same way.
- Uploads of several GB, or the API and the worker on separate hosts: presigned uploads (decision 4).

## 14. Scoping a question to documents

**Problem.** A technician standing at one machine wants the answer from that machine's manual, not from
every manual the plant uploaded. Without a scope, a question can retrieve another product's passage that
looks like an answer. In the sample set, a question about the warranty of WEG electric motors retrieves
the warranty terms of WEG-CESTARI gear units. The eval doesn't scope its questions, so answers must stay
correct without a scope.

**Options considered.**
- No filter: search every document of the tenant.
- An optional list of document ids in the question.
- Inferring the scope from the question text ("the manual MN414").

**Choice: an optional `document_ids` field in the question.**
- Without it, the search covers every document of the tenant.
- The ids come from the upload response, the batch view or `GET /v1/documents`. The client chooses them.
- **Validation.** One query checks that every id belongs to the tenant. If any doesn't, the answer is
  `404 document_not_found` with those ids, whether the id belongs to another tenant or doesn't exist. An
  empty list gets a `422`.
- **Search.** Both searches, by vector and by rare terms, get the filter. Rarity is still counted over the
  whole tenant, so whether a term counts as rare doesn't depend on the scope.
- The response doesn't echo the scope: its `coverage` block (decision 16) counts the documents in scope and
  lists the ones that aren't ready, with their status.

**Why.**
- **It removes ambiguity between documents at the source.** Scoped to the motor guide, the gear units'
  warranty never reaches the prompt.
- **It costs one condition in each search query.** Chunks already carry their document id.
- **Rarity inside the scope would break keyword search on small scopes.** In a 2-page leaflet nearly
  every word occurs in at most 3 chunks. The rare-term search would turn into the every-word search that
  hurt cross-language questions in decision 10.
- **The scope is not inferred from the text.** A wrong guess would silently exclude the right document. A
  code in the question, such as "MN414", already pulls its document through the rare-term search.
- **The same 404 for both cases** doesn't reveal that another tenant's document exists.

**Risk.** An HNSW index with a selective filter can return fewer than 8 chunks, because the filter
applies after the index scan. pgvector's iterative scans (0.8 and later) keep scanning until enough rows
pass, and for a single document an exact scan is cheap. The tenant filter has the same risk (decision 3).
A test checks that a scoped question returns only that document's chunks.

**Next step (DESIGN.md).** In a conversational interface, the filter becomes a tool. The assistant gets
`list_documents` and `ask(question, document_ids)`. It resolves "the gear unit on line 3" once and keeps
the scope for the next questions, or asks "which equipment?" instead of listing one option per document.
That needs:
- conversation state (decision 11);
- document metadata, such as equipment, line and model, given at upload or extracted at ingestion;
- authorization kept on the server: the model passes ids and the server checks them, and the tenant
  always comes from the API key, never from the model;
- a typed list of what is missing, which then has a consumer (decision 12);
- one more model call per tool round trip, measured against the p95 budget.

An MCP server over the API is one way to expose these tools to any compatible assistant.

**What would change our mind.**
- Scoped questions losing recall in the eval: fetch more candidates before filtering, or use exact search
  for small scopes.
- Questions that name their manual being frequent and the rare-term search missing them: suggest a scope
  to the client, without applying it on our own.

## 15. Errors and provider failures

**Problem.** When something fails, the client and the on-call engineer need to know what failed and
whether a retry helps (UC3). The answer path depends on an external model, which can be down, out of
quota, slow, or return something unusable. Our runs on the free tier saw all of these:
- "high demand" 503s on both models, 4 of 48 calls in one run on the default model;
- 500s;
- answers that took minutes.

**Choice.**
- **One error body for the whole API:** `{"error": {"code", "message", "request_id", "retryable"}}`, with
  details when they help. Client errors use the usual codes: 401 for the key, 404, 413, 422. Ingestion
  failures are asynchronous, so they appear in the document's status, not as HTTP errors.
- **Model failures on the answer path:**

| Failure | HTTP | `code` | `retryable` |
|---|---|---|---|
| Provider down (5xx, connection), after the retry | 503 | `llm_unavailable` | true |
| Rate limit or daily quota | 503 with `Retry-After` | `llm_rate_limited` | true |
| Prepaid balance exhausted or spend cap reached | 503 | `llm_billing` | false |
| Key or model rejected | 503 | `llm_misconfigured` | false |
| Call past its deadline | 504 | `llm_timeout` | true |
| Output outside the schema, or a finish reason other than `STOP` | 502 | `llm_invalid_output` | true |
| Refused by the provider (safety, recitation) | 502 | `llm_refused` | false |
| Request rejected by the provider as malformed (400) | 502 | `llm_invalid_request` | false |

- **A 503 body carries the most relevant passages,** marked as not an answer, with document and page.
- **One retry, only where it can help:** a provider 5xx or connection error, a per-minute rate limit whose
  `Retry-After` fits in the deadline, and a call that timed out. Never on the daily quota, billing, a
  rejected key or a malformed request: the same request would fail the same way.
- **Deadlines.** A question has 60 s. Each call gets at most 29.5 s, half of what remains after the backoff,
  so a call that times out still leaves room for its retry.
- **A verifier rejection is not an error.** It is an outcome of the answer.
- **The ingestion path uses the same classes.** A rate limit reschedules the job without spending an
  attempt; a daily quota, billing or a rejected key pauses the model (or the whole account) for 15 minutes;
  a 400 fails only the page that caused it (decision 6).

**Why.**
- **`retryable` and `Retry-After` tell the client what to do** without parsing the message.
- **503, not 429, for the provider's quota.** A 429 tells the client it is sending too many requests. Here
  it is our quota at the provider that ran out.
- **502 and 504 keep their meaning for a gateway:** the upstream answered with something invalid, or
  didn't answer in time.
- **`request_id` in every error** links the client's report to the logs and the answer record.
- **Passages keep the service useful during an outage.** Search is local and keeps working when the
  provider is down (decision 10), so the technician can still read the source.
- **The retry is bounded.** In the run above, 3 of the 4 calls that got a 503 passed on a second try. The
  deadline keeps the retry from blowing the latency budget.
- **`retryable: false` for billing and configuration.** Retrying can't help until someone acts, and the
  client and the on-call see that at once, instead of a generic 503.

**Known limitation.** The table covers the failures seen so far. The build added three codes: one from
the provider's billing documentation (a prepaid balance at zero answers 402; a spend cap pauses requests),
one for a rejected key or model, and one for a 400 that a single OCR page can trigger.

**What would change our mind.**
- Clients that can't handle passages in an error: drop them from the 503.
- Provider outages long or frequent enough to matter: a second provider as fallback, at the cost of one
  more key.

## 16. Coverage in answers

**Problem.** Pages become searchable as they finish (decision 1), so a question can arrive while part of a
document is still being transcribed, or after some pages failed for good. A "not found" then means "not
in the pages searched so far", which is not the same as "the documents don't say". The technician needs
to tell the two apart. Otherwise a pending page turns into a confident "the manual doesn't answer this".

**Choice.**
- **Every answer carries a `coverage` block** for its scope, meaning the tenant's documents or the ones in
  `document_ids` (decision 14):
  - `complete`: every document in scope is `ready`, with every page indexed or blank;
  - counts of documents, and of pages indexed, pending and failed;
  - `incomplete`: the documents that aren't ready, each with its status and its pending and failed pages,
    capped at about 20 entries.
- **Pending and failed are kept apart.** A pending page means the answer may change, so asking again later
  helps. A failed page means no answer will cover it.
- **The counts are kept on the document row.** Every transaction that changes a page's state (extraction,
  indexing, a page failing for good) recomputes them and the document's status, with the document row
  locked. A question reads the document rows in scope, never the pages table.
- **Documents with no pages yet** (`queued`, before extraction) count as incomplete, with unknown page
  counts. Documents that failed extraction (encrypted, corrupted) count as failed.
- **When a document in scope is still queued or processing, the prompt gets one extra line:** part of the
  documents is still being processed, and if the answer isn't found, say that the documents searched so far
  don't contain it. The model writes that caveat in the question's language, without numbers. A failed
  document doesn't add the line, which would tell the model something false.
- **The eval waits for every document to reach a final state** before asking. The report stores the
  coverage of each question and flags any incomplete one, so that a pending page isn't counted as a model
  error. The eval's broken files (locked, truncated, blank) run in their own tenant, so their failures
  don't mark the scored answers' coverage as incomplete.

**Why.**
- **The counts stay exact for what matters,** which is whether a page is searchable. They change in the
  same transaction as the page states, and a page that fails for good also triggers an `index` run
  (decision 5). Between OCR and indexing, a page counts as pending, which is true: it can't be found yet.
- **Reading document rows keeps the cost flat.** A tenant with tens of thousands of pages pays the same
  as one with a leaflet.
- **The system can't write the caveat itself.** Without language detection (decision 9), a fixed notice
  would come out in English next to a Portuguese answer. The structured block lets clients build their own
  message. The prompt line costs a few tokens, and only when something is pending.
- **No numbers in the caveat,** because the verifier would reject a page count that no cited unit
  contains.

**What would change our mind.**
- Clients needing page-level detail (which pages are missing): list page numbers per document. It's a
  read from the pages table, only for incomplete documents.
- Answers given on incomplete coverage turning out often wrong in practice: hold `not_found` answers until
  the scope is complete, or mark them as provisional.

## 17. When the verifier rejects an answer

**Problem.** The verifier of decision 7 rejects an answer when one of its numbers isn't in the units it
cites. We replayed it on 170 answers from the default model. It rejected 10, and only 1 was wrong: "15000
Hrs. at 12000 RPM", a speed that doesn't exist. The other 9 were right, with an incomplete citation:
- the speeds sat in a header row the model didn't cite;
- "divide by 2" came from a sentence it didn't cite;
- a value was cited through its footnote, without the starred row;
- a column of four rows hit the cap of 6 units.

Turning every rejection into "not found" would throw away correct answers. Returning a rejected answer
would ship unverified numbers.

**Options considered.**
- Return `not_found` on any rejection.
- Return the answer, flagged as unverified.
- Call the model again once, with the verifier's message.
- Expand the cited evidence deterministically before judging.

**Choice: expand, then retry once, then decline.**
1. **Three deterministic expansions**, with no model call:
   - a cited table row brings its table's header rows;
   - a cited footnote brings only the starred cells it explains, and a cited starred row brings its
     footnote;
   - a number missing from the citations, but present in an uncited sentence of a cited page, adds that
     sentence. Never a table row, so a value taken from another row of the same table still fails.
2. **One retry.** If the answer still fails, the model is called once more with the verifier's message
   (which numbers have no source), within the question's deadline.
3. **Decline.** If it fails again, the answer is `not_found` with the reason `unverified`. It carries the
   most relevant passages, as in a provider outage (decision 15). The unverified answer is never returned.
4. **The prompt says headers and footnotes are attached automatically**, so the model doesn't spend
   citations on them.

**Why.**
- **The expansions cut rejections from 10 to 4,** from 5.9% to 2.4%, replayed offline on the same 170
  answers. The 4 left are the real error, a column of rows over the cap, a value in an uncited table row,
  and a number in the summary line.
- **Replayed again in the build,** with the product's verifier, on 287 answers (every prompt version and
  both models, the paid-key runs included): 15 rejections without the expansions, 6 with them (5.2% to
  2.1%). The 6 left: one real error (a value from another row), two columns of four rows plus headers over
  the cap, two answers whose only ungrounded number is "0", and a "2" in a summary line. In the
  73-question eval, 1 of 74 answer calls was a verifier retry, and no answer was declined.
- **The expansions add context that can't change which cell is cited.** Headers name columns, a footnote
  explains the cells it marks, and a sentence can't hold another row's value.
- **A retry on about 2% of questions leaves the p95 where it is.** Only the p99 grows, by one call.
- **Declining with the passages** keeps a wrong answer from going out, and still gives the technician the
  source.

**Known limitations.**
- Small numbers find an anchor by coincidence: "divide by 2" was grounded by the "2 pole" of another
  sentence.
- The verifier still doesn't check the column inside a cited row (decision 7). "12000Hrs. at 18000 RPM"
  passed, because 18000 sits in the same row.
- While testing, the bold markup of a header (`**3600**`) matched the footnote marker pattern and attached
  a footnote to unmarked rows. It is fixed and covered by a unit test. The build found and fixed two more
  defects, each with a test: the starred-cell pattern took the rest of the row, which grounded the values
  of other cells; and an option's condition and answer were joined into one string, so "up to 210" and
  "5,500 h" read as a single number and a correct option was rejected.

**What would change our mind.**
- Retries often failing again on correct answers: a cap per table instead of per answer, or adding a row
  the model names but didn't cite.
- Rejections above about 5% on the model in use: the retry starts to move the p95, so revisit the
  expansions first.

## 18. Answer model

**Problem.** One model call answers each question. It has to:
- pick the right status;
- cite the right units;
- answer in the question's language, from passages that may be in another one.

At scale, the p95 budget is 4 s, and every question pays for this call. That makes it the largest cost at
production volume.

**What we measured** (free-tier key, 30 Oct; retrieval of decision 10, schema of decision 12, verifier of
decision 17):
- **A design set of 18 questions:**
  - the 7 public questions;
  - a question with a default and an exception;
  - 5 ambiguous questions: a condition left unstated, a value matching two labels, a question fitting two
    subjects, two synthetic oils, an oil type left unstated;
  - 5 near-miss "not found" questions: another product's warranty, a warranty that is only mentioned, an
    IP rating that only another brand's brochure gives, and a temperature and a torque the manuals don't
    state.
- **48 held-out questions**, not used to tune the prompt: the public questions in Portuguese, Spanish and
  German, and the 30 cross-language and code questions of decision 10.
- **Two prompts.** The second added four generic rules for the first one's failures, without naming any
  manual.
- **4 planted prompt injections.**

**Options considered.** `gemini-3.5-flash-lite` and `gemini-3.8-flash`, with thinking `minimal` and
`low`. Gemma 4 through the API was left out: its hosted latency was already unusable in decision 6, and a
self-hosted model is a question for production scale.

**Results.**

| | Flash Lite | Flash |
|---|---|---|
| Design set, right status | 16 of 18, in every run | 14 of 14 run (2 skipped by plan, 2 lost to 503s and the daily quota) |
| A value matching two labels ("frame 180": NEMA or IEC) | answered with one reading, 5 of 5 runs | gave both readings, 2 of 2 |
| Another product's warranty, asked about "WEG electric motors" | answered "12 months", 5 of 5 | not found, naming whose warranty it is, 2 of 2 |
| Held-out set | 45 of 48; 3 false "not found" on questions in Spanish, German and Portuguese about English manuals | not measured |
| Verifier rejections, after expansions | 1 of 66 | 0 of 15 |
| Answer-call latency | p50 1.2 s, p95 2.3 s, max 7.2 s (66 calls) | not measurable: 11–280 s, unrelated to output size |
| Median tokens, input / output | 3,014 / 79 | 2,848 / 110 |
| Cost per question, list price | about $0.0011 | about $0.0032 |
| Planted injections obeyed | 0 of 4 in the prototype, 0 of 3 through the API | not run |

- **Thinking made no difference on Flash Lite.** With `low`, it produced thinking tokens on 2 of 18
  questions and changed no status.
- **The second prompt fixed none of Flash Lite's status errors.** Flash got those questions right with
  both prompts.

**Measured in the build** (paid key; the API's final prompt and verifier):

| | Flash Lite | Flash |
|---|---|---|
| Design set, right status | 16 of 18, the same two errors | 18 of 18 |
| Answer call only, design set (prototype) | p50 1.1 s, p90 2.4 s, max 3.2 s | p50 3.1 s, p90 5.0 s, max 19.7 s |
| Whole pipeline through the API, design set | p50 1.3 s, p90 1.8 s, max 4.0 s | p50 2.7 s, p90 4.1 s, max 7.1 s |
| Whole pipeline, 73-question eval | not run | p50 2.9 s, p95 8.7 s, max 19.7 s; 18 of 73 over 4 s |
| Held-out set, right status | 45 of 48 (free tier, before the build) | 45 of 48 (eval) |
| Cost per question | $0.0013 | $0.0032 |

- **The tail is the provider's queue more than generation.** In the eval, two answers of 78 and 98 output
  tokens took 18–20 s, and four of under 60 tokens took 6–8 s. Ambiguous answers are the long ones: 520
  output tokens and 5.8 s at the median. Without them, the p95 is still 8.3 s.
- **On the held-out set the two models tie on status,** and both miss on the cautious side (`not_found`, or
  a clarification). Flash Lite's two design-set errors are the unsafe kind, and they repeat in every run.

**Choice: `gemini-3.8-flash` as the default answer model, pinned, with thinking `low`.**
`GEMINI_ANSWER_MODEL` selects any model of the same API. Flash Lite is the documented option for free-tier
keys and for lower cost.

**Why.**
- **Flash Lite fails exactly where the requirements look hardest:** on the design set it picks one case
  silently and answers with another product's passage, in every run. Across languages it does no worse: on
  the 48 held-out questions both models get 45 statuses right (Flash Lite before the build, Flash in the
  eval), and both miss on the cautious side. A wrong answer is worse than none.
- **The prompt didn't fix those errors; the model did.**
- **Flash's free-tier latency is queueing, not generation.** An answer of 54 tokens took 280 s, and one
  of 644 tokens took 79 s, while Flash Lite answered the same prompts in 1–2 s. We hold a serving problem
  against a model no more than we did with Gemma in decision 6.
- **The MVP stands or falls on its answers.** The 4 s p95 belongs to the production scenario.

**Costs and open risks.**
- **Flash's p95 on a paid key is above our own threshold.** Measured in the build at 8.7 s in the eval, it
  fired the first condition under "What would change our mind". We kept Flash for the MVP: its misses fall
  on the cautious side and Flash Lite's two design-set errors do not, and the MVP stands or falls on its answers.
  Meeting a 4 s p95 is planned in DESIGN.md: dedicated provider capacity, hedged requests, shorter ambiguous
  answers, routing.
- **About 2.5 times the cost per question:** $3.2k against $1.3k per million questions at list prices,
  measured in the build. At production volume, that is the largest line of the bill.
- **`make eval` can't run on Flash with a free-tier key.** The free tier allows 20 requests a day, and a
  503 counts against the quota. The README says so and points free-tier keys to Flash Lite.
- **The evidence for the choice is narrow:** two questions of our own, right 0 times in 14 runs on Flash
  Lite and 10 times in 10 on Flash, counting the build's runs. On the 48 held-out questions, mostly
  cross-language, the two models tie (45 of 48).

**Found along the way.** A bug in our prompt builder dropped the speed row of a table. Once rendered, two
rows looked identical, and one was removed as a duplicate. It made two questions fail on both models, and
was fixed before the final runs. Prompt construction now has its own version. Reading a failure starts by
checking what the model actually received.

**What would change our mind.**
- Flash's p95 on a paid key above about 3.5 s: Flash Lite as the default, with the errors above as known
  limitations, or a smaller output budget. This fired in the build (8.7 s), and we kept Flash for the reason
  under "Costs and open risks".
- An eval on new questions showing Flash Lite as accurate as Flash: Flash Lite, for latency and
  cost.
- A signal that tells hard questions apart before answering: route the easy ones to Flash Lite and the
  hard ones to Flash (DESIGN.md).

## 19. Evaluation

**Problem.** The requirements ask for a harness that takes a folder of PDFs and a JSONL file, ingests through
the API, runs the questions and reports metrics we choose and compute "reliably". Real use brings new
questions and other documents, so the harness must work on files it has never seen. Free-text answers in
several languages can't be compared with the reference by string matching. And the headline has to follow
the product's rule: a wrong answer is worse than none.

**Options considered.**
- Content: word or number overlap with the reference; an LLM judge that gives a score; an LLM judge guided
  by the reference, fact by fact; human labels only.
- Headline: one accuracy number; accuracy plus a separate count of unsafe answers.
- Harness: calling our Python code in-process; calling the API like any client.

**Choice.**
- **The harness is an API client.** It creates a fresh tenant per run, uploads the whole folder as one batch
  (non-PDFs included), waits for every document to reach a final state, asks each question and reads its
  answer record (`GET /v1/answers/{id}`). A report depends only on the folder and the JSONL.
- **One bad input doesn't stop the run.** A line it can't score (not JSON, a required field missing, an
  unknown status, a repeated id) is skipped and listed under "Input problems" in the report. A question
  whose source file isn't in the folder is still asked, with its source reported as missing. The run stops
  only when nothing can be asked.
- **Per question, in pipeline order:**
  - status, as a 3×3 matrix plus HTTP errors;
  - content, by the judge;
  - source: some citation on an expected document and page;
  - quote in the PDF: each quoted line looked up in the page's text layer, read again by the eval with
    PyMuPDF; transcribed pages are counted apart;
  - grounded numbers, re-checked from the API's response alone;
  - retrieval: the rank of the expected page among the 30 candidates, and whether it reached the 8
    passages sent to the model;
  - the first failing stage: `ingestion`, `retrieval`, `ranking`, `provider`, `api`, `verification`,
    `generation`;
  - latency, tokens and cost, from the answer records.
- **Headline: correct and unsafe, side by side**, per question set and per language.
  - Correct: the expected status, the content right, and a citation on an expected page. For `not_found`,
    the status alone.
  - Unsafe: an answer when the documents don't answer, one case picked from an ambiguous question, or a
    contradicted fact.
  - Over-cautious answers (`not_found` or a clarification when an answer was expected) are counted apart.
- **Judge: guided by the reference, with the verdict computed by code.** The judge model lists the facts
  of the reference answer, marks each one essential or example, and says whether the answer states,
  misses or contradicts it. Code turns that into correct, partial or wrong. It runs only when the status
  matches and isn't `not_found`, at temperature 0, and its verdicts are cached by a hash of their inputs.
  A numeric cross-check flags verdicts that disagree with the numbers (correct with none of the reference's
  numbers in the answer, partial or contradicted with all of them). `--no-judge` skips it and reports
  bounds instead.
- **Question sets, reported apart:** public (7); design (18, used to write the prompt and the verifier);
  held-out (48, not used for the prompt, but used to choose the search); and a robustness run on generated
  files (a scanned page, a locked, a truncated and a blank PDF, non-PDFs), in its own tenant.

**Why.**
- **Through the API, because that is what users call and what the on-call reads.** One record explains both
  an eval miss and a production complaint, and the harness tests the operational surface too.
- **Skip and report, rather than fail fast.** A new set brings files we have never seen; a renamed
  file or one malformed line must not turn the whole report into nothing.
- **Two numbers, because the errors aren't symmetric.** An average hides whether the misses are guesses or
  refusals. In the committed run: 62 of 73 correct, 1 unsafe, 3 over-cautious. The unsafe one (x-11) is a
  judge false positive, and one over-cautious (c-04) falls between two prompt rules (below).
- **Facts instead of a score.** A 3 out of 5 can't be audited; "missing: services, 6 months" can, and the
  report prints it next to the reference.
- **The rule in code, the reading in the model.** The model does what it is good at; the threshold is
  fixed, versioned and tested.
- **Independent checks catch the judge and the references.** The first judge prompt marked an answer
  partial for not repeating a condition the question itself gave. With the second, x-11 came out
  contradicted: the answer gives the manual's "+5% … above or below" for frequency, and our reference "±5%",
  the same range. The numeric cross-check flagged both. Reading the PDF independently found 3 of 44
  text-layer quotes that aren't literally in the PDF, all artifacts of the extractor: two split headers
  and a figure marker.
- **The failure stage says where to look.** All 11 misses of the committed run are `generation`. On 10
  external manuals, one was `ingestion` and one `ranking`, which pointed at retrieval.
- **A fresh tenant per run, with the OCR cache.** No answer depends on an earlier run's documents. OCR is
  cached by file content across tenants (decision 5), so later runs pay no transcription; a clean clone
  pays it once (about $0.07 for the 54 flagged pages of the sample).

**Known limitations.**
- **The judge is provisional.** `gemini-3.5-flash-lite` with prompt `j2` has not been measured against
  human labels. 46 cases with draft labels exist, but the meta-evaluation hasn't been run: agreement, and
  above all false "correct" verdicts, the direction that would hide an unsafe answer. Correct and unsafe
  inherit the judge's errors.
- 30 held-out reference answers are drafts written from the cited page, not yet reviewed. Two are disputed:
  x-11, where the judge, not the reference, is wrong (above), and c-04, which expects the 60 Hz value. Its page
  gives a 60 Hz table and a note for 50 Hz, a default and an exception that rule 5 of the prompt answers as
  `answered`; the model read the frequency as an open condition. The report scores both as written. The design set is not independent. Each question ran once.
- The expected position is measured per page, not per chunk: when the right page reaches the model through
  the wrong chunk, the miss is classified as `generation`.
- The language of the answer isn't measured.

**What would change our mind.**
- The meta-evaluation showing more than a few false "correct" verdicts: a stronger judge model, or two
  judges that must agree.
- Agreement of 95% or more, stable across repeats: keep the cheap judge.
- A decision resting on one or two questions: `--repeat` before acting on it.

## 20. Telemetry and observability

**Problem.** UC3: when a document fails, an answer is wrong, the provider is down or latency spikes, the
on-call engineer must find out what happened to that request or document without guessing. What explains
an answer (the question, the passages, the model's raw output) is customer data, and some customers won't
accept it in a third party's system.

**Principle.** Customer data doesn't leave our infrastructure for observability. The model provider
receives only what answering requires: the passages retrieved for a question and the page images flagged
for transcription.

**Options considered.**
- A hosted LLM-observability service. Ruled out by the principle.
- Langfuse, self-hosted: traces, prompt versions and evals in one interface. Its current version needs
  ClickHouse, Redis and object storage: three more services in the compose file anyone runs to try it.
- The OpenTelemetry SDK exporting to our own collector and backend (Tempo, Jaeger or ClickHouse). It is
  the standard, but it also adds services, and the traces would live apart from the jobs, pages and answers
  they explain.
- A `spans` table in Postgres behind a small `Telemetry` interface, using OpenTelemetry's attribute names.

**Choice: spans and answer records in Postgres, with OpenTelemetry GenAI names.**
- One span per HTTP request, job, pipeline stage and model call, carrying `request_id`, `job_id`,
  `document_id` and `tenant_id`. Model calls use the GenAI semantic conventions (`gen_ai.request.model`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`), with thinking tokens counted as output, as
  they are billed.
- JSON logs with the same ids. Every response carries `X-Request-ID`, and every error body its
  `request_id`.
- **The answer record** (`GET /v1/answers/{id}`), written before the response is sent: the question, the
  scope, the versions (model, prompt, passage assembly, index, search), the coverage, the 30 candidates
  with their vector and keyword ranks, every label and the unit it pointed to, each model call with its raw
  output, verifier verdict, latency and tokens, and the pipeline stages.
- **The document view** (`GET /v1/documents/{id}`): pages by state and text source, why each page went to
  OCR, the OCR calls and tokens, and any provider pause with its reason.
- Two SQL views (model calls and requests per minute) and the on-call queries in the README.
- Never in spans or logs: keys, question text, page text, file names.

**Why.**
- **One place to join.** "Why is this document `partial`?" is a join between its pages, its jobs and their
  spans. "Why was this answer wrong?" is one record, from the candidates to the raw output.
- **Three services in compose,** as decision 2 wanted.
- **Standard names make an export a mapping, not a rewrite.** Moving to an OpenTelemetry collector changes
  where spans go, not the instrumentation.
- **It is checked against the bill.** In the build, the tokens of the OCR spans matched the spend log
  exactly (55 calls, 72,832 input and 19,533 output tokens), and each answer record matched its model-call
  span.

**Known limitations.**
- Telemetry shares the database with the data: about 12 spans per question.
- No retention job yet for spans and answer records. Finished jobs and unreferenced chunk history are
  cleaned.
- No metrics, alerts, dashboards or trace viewer: the on-call uses SQL and the two endpoints.
- The `http.request` and `answer.record` spans are written after the response, so a
  `GET /v1/answers/{id}` sent right away may miss them. The stages and totals are already there.

**What would change our mind.**
- Production volume, about 3,600 spans per second at the question peak: export to a self-hosted OpenTelemetry
  collector with sampling and retention, and keep only the answer record in Postgres (DESIGN.md).
- Prompt and judge iteration becoming the bottleneck: Langfuse, self-hosted, which fits the principle.

## 21. Build-time model quantization

**Problem.** The int8 model of decision 10 must be inside the image, so that no container downloads
anything when it starts. The first version quantized it in the Dockerfile with ONNX Runtime's
`quantize_dynamic`, which peaked at 8.5 GB of RAM. Docker Desktop often gives its VM 8 GB, and in an 8 GB
container the build was killed for lack of memory. The first `docker compose up` from a clean clone, on
someone else's machine, is where the whole project can fail.

**Options considered.**
- Keep it and ask for 10 GB of memory in the README.
- Ship BAAI's full-precision ONNX export without quantizing: no build peak, but 1.9 GB per process and
  78 ms per query under torch, against 0.9 GB and 17 ms (decision 10).
- Download an int8 file quantized by someone else: an artifact we would have to trust and measure again.
- Write our own streaming quantizer: different bytes, so every retrieval measurement would have to be redone.
- Keep ONNX Runtime's quantizer, but stop it from holding the full-precision weights three times.

**Choice: the last one.** The quantizer loads the model for shape inference, deep-copies it and reloads
it, so three copies of the 2.27 GB of weights are alive at once. Shape inference needs only the graph
(725 KB). The build replaces three internal ONNX Runtime functions so that the weights stay on disk until
each one is quantized.

**Why.**
- **The same file, byte for byte.** The output has the same sha256 as before, in 10 runs, so the retrieval
  results of decision 10 stand without measuring again.
- **Peak RAM from 8.5 to 4.0 GB, and 11.6 s to 4.5 s.** The build now passes in a 4 GB container. The
  README asks for 6 GB for the first build.
- What remains of the peak is the word-embedding matrix (977 MB), which the quantizer copies four times.

**Risks.**
- The patch depends on internal function names of ONNX Runtime. The version is pinned. If a new version
  changes them, the build fails at once with an error; it can't produce a different file silently.
- The build downloads 2.3 GB from Hugging Face at a pinned revision: 10 to 13 minutes measured, and a
  dependency on that service being up.

**What would change our mind.**
- Docker VMs under 4 GB: read the weights through a memory map (about 3 GB, measured), or publish our int8
  file as a release artifact pinned by its sha256.
- An ONNX Runtime release with a low-memory option: drop the patch.

## 22. Parsing untrusted PDFs

**Problem.** Uploads come from customers and their vendors. Some are corrupted, password-protected or not
PDFs at all, and any of them can be hostile: decompression bombs, recursive objects, pages that take
forever to render. The parser is C code running on those bytes. One bad file must not crash the worker,
exhaust the machine's memory or block the rest of the batch (UC1).

**Options considered.**
- Parse inside the worker process.
- A child process with operating-system limits on address space (`RLIMIT_AS`) and CPU time.
- A child process with a time limit, a stall watchdog and polling of its resident memory.
- A container or micro-VM per document.

**Choice: a sandboxed child process per document, and one per page render for OCR.**
- The API never parses the PDF. While the file streams in, it checks only the `%PDF-` magic bytes and the
  size (decision 4).
- The child gets a minimal environment (no keys, no database URL) and no core dumps. It has a total time
  limit and a watchdog that kills it when no page finishes for a while. The parent reads its resident
  memory every 0.2 s and kills it above `EXTRACT_MEMORY_MB`.
- Each outcome has its own code on the document: `pdf_encrypted`, `pdf_corrupt`, `pdf_no_pages`,
  `pdf_too_many_pages`, `extract_timeout`, `extract_memory`, `extract_crashed`, or `no_text` when every page
  is blank. A page whose Markdown export fails goes to OCR on its own (`layer_error`); the rest of the
  document is unaffected.

**Why.**
- **An address-space limit fails here.** PyMuPDF's layout mode loads ONNX Runtime, whose threads reserve
  2.4 GB of address space with 140 MB resident on a 16-CPU machine. Under a 2 GB limit the child died at
  import, before reading any PDF. Resident memory is what actually hurts the host.
- **Measured:** the child peaked at 337–479 MB on the four sample manuals and 917 MB on a 454-page handbook.
  Locked and truncated files failed in 0.6–0.8 s, and non-PDFs never reach the worker (422 at upload). In
  the eval's robustness run, every broken file reached a final state with its code, and the scanned page
  uploaded next to them was answered.

**Known limitations.**
- A memory spike between two readings can pass the container's limit. The kernel then kills the child, and
  the job retries and ends as `extract_crashed`.
- No network or system-call isolation: the child shares the worker's kernel and host.
- Extraction is the only stage not split per page, so its time limit applies to the whole document. The
  limit is 2,400 s: a 500-page catalog at 1.5 s per page, the slowest rate measured, takes 750 s, under a
  third of it.

**What would change our mind.** Production traffic: a separate extraction pool with no network, under
gVisor or Firecracker, and page ranges for large catalogs (DESIGN.md).

## Cuts and known limitations

What we left out on purpose, and what we know is weak, with what we would do. DESIGN.md puts the
production items in order.

### Deliberately cut

- **User interface.** The delivery is the API, with `/docs` to try it in a browser. A useful interface
  would also open the cited page with the passage highlighted, which needs page images and word
  coordinates (decision 8).
- **Reranker.** The expected page was already among the 8 passages for every question on the sample
  manuals, and a CPU cross-encoder costs about 3 s (decision 10). Revisit when the eval shows pages
  retrieved but ranked too low; on 10 external manuals, that happened once.
- **Query rewriting.** The same recall for +1.2 s and one more model call (decision 10).
- **Conversation.** Questions are stateless (decision 11); `follow_up_of` is the next step.
- **Tesseract and Document AI.** Tesseract loses table structure; Document AI needs a service account and a
  GCP project, one more credential to set up (decision 6). Both stay candidates for pages that a
  vision model refuses.
- **Answer cache.** Its hit rate is unknown without traffic, and a semantic cache is unsafe here: "frame 256
  at 1800 RPM" and "at 3600 RPM" have nearly identical embeddings and different answers. An exact cache is
  in DESIGN.md.
- **Translation of quotes** (decision 9), **tenant and document deletion**, and **cancellation** (below).

### Ingestion and OCR

- **Recitation refusals can be systematic.** On a public-domain scan digitized by Google, both models
  refused 36 of 60 pages, all the prose, leaving the manual 40% searchable; the document ends `partial`.
  There was none on the four sample manuals in the build. Next: an OCR without that filter for refused
  pages (a self-hosted model, or Document AI).
- **Failures aren't cached.** Every new upload of that file pays the same refusals again: 148 calls and
  $0.08 in our run. Next: a negative cache keyed like the OCR cache, with an expiry.
- **Bad scans get confident wrong digits** (decision 6). On a synthetic bad scan at 7.5 pt, five models
  read 0–10 of 28 cells right, with one `[illegible]` mark among them; on a real scan, a value landed in the
  wrong table row. Next: a deterministic legibility check (effective resolution, character height in
  pixels, sharpness) that marks the page, warns in citations and sends it to the strongest model.
- **Drawing pages go to OCR.** A page with ink and under 50 letters is flagged: 27 drawing pages in one
  external manual (two language editions), 24 pages of photos and dividers in another, and 4 pages of part
  codes flagged as garbled. It cost $0.04 and the text is useful, but it raises the OCR volume at production scale.
- **The queue is FIFO, and extraction and indexing run one document at a time.** A batch of 10 external
  manuals (1,229 pages) took 861 s, the last file waiting for all the others. There is no per-tenant
  fairness yet (decisions 1, 2). Next: claims balanced by tenant, then more threads, after measuring RAM.
- **No cancellation and no deletion.** A document or a batch can't be cancelled, and stopping the worker
  only pauses it: the next start resumes the paid pages. Deleting a tenant (documents, history, answer
  records, cache references) isn't built.
- **Near-duplicates.** The same manual exported again is a new document; if the two disagree, answers come
  back ambiguous with one option per document (decision 13).

### Extraction

- **`pymupdf4llm` can damage numbers and headers in the text layer.** In one handbook a decimal comma moved
  to the next line ("0,700" became "0700"); merged table headers split mid-word ("Relubricatio | n
  Intervals"); a figure marker (`<!-- End of picture text -->`) ends up in a quote. The verifier checks
  against the stored text, so a damaged number passes it. The eval found 3 of 44 text-layer quotes not
  literally in the PDF, all from this. Next: compare each table cell's numbers with the plain text layer
  inside the cell's box, and clean display markup from quotes without changing the stored text.
- **A table that continues on the next page** becomes two chunks (decision 10).
- **License.** PyMuPDF and `pymupdf4llm` are AGPL. Production needs a commercial license or another
  extractor.

### Answers and verification

- **The verifier doesn't check the column inside a cited row** (decision 7): "15,000 hours" for 1800 RPM
  passes, because 15000 sits in the same row.
- **Leading-dot decimals and numbers in words aren't checked.** `.030"` reads as 30, and "fifteen" isn't a
  number. American manuals write clearances this way. Next: accept a leading dot; numbers in words stay a
  limitation.
- **The answer's language.** With only Portuguese passages available, Flash Lite once answered an English
  question in Portuguese (one sample), and the eval doesn't measure language. Next: measure it per `lang`
  before touching the prompt.
- **An answer can stop at the first category the page lists.** The public warranty question asks about gear
  units; the page gives 12 months for products and 6 for services, and the answer leaves services out in every
  language, with that line in the prompt (decision 12). The eval report comments on it under "Unfixed failures".

### Platform and operations

- **The api and the worker must run on the same host,** sharing a local volume (decision 4). Next: object
  storage and stateless instances.
- **One model provider.** An outage returns 503 with passages, and there is no second provider.
- **Apple Silicon checked only under emulation; free-tier key not tested.** Under emulated `linux/arm64`, the
  image builds unchanged from a clean checkout (same int8 model hash), the stack starts healthy, and ingestion,
  OCR, indexing and an answer work. The full eval wasn't run (emulation is 20 to 30 times slower), and speed and
  memory on a real Mac weren't measured. One unit test compares query vectors with x86 values at cosine 0.999
  and would fail on arm64, where they differ at 0.97 to 0.99. On a free-tier key, Flash allows 20 requests a day,
  so `make eval` needs `GEMINI_ANSWER_MODEL=gemini-3.5-flash-lite` (README).
- **The eval's judge is provisional** (decision 19).
