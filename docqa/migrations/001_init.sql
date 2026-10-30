CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE OR REPLACE FUNCTION docqa_unaccent(text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
AS $$ SELECT public.unaccent('public.unaccent'::regdictionary, $1) $$;

CREATE TABLE tenants (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name       text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE api_keys (
  key_hash   text PRIMARY KEY,
  tenant_id  uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE documents (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     uuid NOT NULL REFERENCES tenants(id),
  sha256        text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  filename      text NOT NULL,
  blob_key      text NOT NULL,
  status        text NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued','processing','ready','partial','failed')),
  error_code    text,
  error_message text,
  page_count    int,
  pages_indexed int NOT NULL DEFAULT 0,
  pages_pending int NOT NULL DEFAULT 0,
  pages_failed  int NOT NULL DEFAULT 0,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, sha256)
);
CREATE INDEX documents_tenant_created ON documents (tenant_id, created_at DESC, id);

CREATE TABLE batches (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id  uuid NOT NULL REFERENCES tenants(id),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE batch_items (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   uuid NOT NULL REFERENCES tenants(id),
  batch_id    uuid NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  filename    text NOT NULL,
  outcome     text NOT NULL CHECK (outcome IN ('accepted','duplicate','rejected')),
  document_id uuid REFERENCES documents(id) ON DELETE SET NULL,
  error_code  text,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX batch_items_batch ON batch_items (batch_id, created_at, id);

CREATE TABLE pages (
  tenant_id       uuid NOT NULL REFERENCES tenants(id),
  document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page_no         int  NOT NULL CHECK (page_no >= 1),
  state           text NOT NULL CHECK (state IN ('needs_ocr','text_ready','indexed','blank','failed')),
  text_source     text CHECK (text_source IN ('layer','ocr')),
  layer_text      text NOT NULL DEFAULT '',
  ocr_text        text,
  quality         jsonb NOT NULL DEFAULT '{}',
  page_label      text,
  text_rev        int  NOT NULL DEFAULT 0,
  extract_version text NOT NULL,
  ocr_version     text,
  index_version   text,
  error_code      text,
  error_message   text,
  updated_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (document_id, page_no),
  CHECK (state NOT IN ('text_ready','indexed') OR text_source IS NOT NULL)
);
CREATE INDEX pages_tenant_doc_state ON pages (tenant_id, document_id, state);

CREATE TABLE ocr_cache (
  sha256      text NOT NULL,
  page_no     int  NOT NULL,
  ocr_version text NOT NULL,
  text        text NOT NULL,
  model       text NOT NULL,
  PRIMARY KEY (sha256, page_no, ocr_version)
);

CREATE TABLE chunks (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     uuid NOT NULL REFERENCES tenants(id),
  document_id   uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page_no       int  NOT NULL,
  ordinal       int  NOT NULL,
  text_rev      int  NOT NULL,
  text_source   text NOT NULL CHECK (text_source IN ('layer','ocr')),
  index_version text NOT NULL,
  text          text NOT NULL,
  units         jsonb NOT NULL,
  embedding     vector(1024) NOT NULL,
  tsv           tsvector GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, docqa_unaccent(text))) STORED
);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX chunks_tenant_version ON chunks (tenant_id, index_version);
CREATE INDEX chunks_doc_page ON chunks (document_id, page_no, ordinal);

CREATE TABLE chunks_history (
  id            uuid PRIMARY KEY,
  tenant_id     uuid NOT NULL REFERENCES tenants(id),
  document_id   uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page_no       int  NOT NULL,
  ordinal       int  NOT NULL,
  text_rev      int  NOT NULL,
  text_source   text NOT NULL,
  index_version text NOT NULL,
  text          text NOT NULL,
  units         jsonb NOT NULL,
  archived_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX chunks_history_archived ON chunks_history (archived_at);

CREATE TABLE jobs (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id        uuid NOT NULL REFERENCES tenants(id),
  kind             text NOT NULL,
  document_id      uuid REFERENCES documents(id) ON DELETE CASCADE,
  page_no          int,
  state            text NOT NULL DEFAULT 'queued' CHECK (state IN ('queued','running','done','failed')),
  attempts         int  NOT NULL DEFAULT 0,
  max_attempts     int  NOT NULL,
  run_after        timestamptz NOT NULL DEFAULT now(),
  lease_token      uuid,
  lease_owner      text,
  lease_expires_at timestamptz,
  last_error_code  text,
  last_error       text,
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),
  finished_at      timestamptz
);
CREATE INDEX jobs_queued   ON jobs (kind, run_after, id) WHERE state = 'queued';
CREATE INDEX jobs_running  ON jobs (kind, lease_expires_at) WHERE state = 'running';
CREATE INDEX jobs_document ON jobs (document_id);
CREATE UNIQUE INDEX jobs_one_queued_index ON jobs (document_id) WHERE kind = 'index' AND state = 'queued';
CREATE UNIQUE INDEX jobs_one_ocr_per_page ON jobs (document_id, page_no)
  WHERE kind = 'ocr_page' AND state IN ('queued','running');

CREATE TABLE provider_state (
  model         text PRIMARY KEY,
  next_call_at  timestamptz NOT NULL DEFAULT now(),
  paused_until  timestamptz,
  pause_reason  text CHECK (pause_reason IN ('quota_exhausted','billing','misconfigured')),
  pause_detail  text
);

CREATE TABLE answers (
  id               uuid PRIMARY KEY,
  tenant_id        uuid NOT NULL REFERENCES tenants(id),
  question         text NOT NULL,
  search_query     text NOT NULL,
  document_ids     uuid[],
  http_status      int  NOT NULL,
  status           text CHECK (status IN ('answered','ambiguous','not_found')),
  reason           text CHECK (reason IN ('unverified')),
  error_code       text,
  model            text NOT NULL,
  prompt_version   text NOT NULL,
  assembly_version text NOT NULL,
  index_version    text NOT NULL,
  coverage         jsonb NOT NULL,
  labels           jsonb NOT NULL DEFAULT '{}',
  calls            jsonb NOT NULL DEFAULT '[]',
  response         jsonb NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX answers_tenant_created ON answers (tenant_id, created_at DESC);

CREATE TABLE answer_candidates (
  answer_id     uuid NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
  tenant_id     uuid NOT NULL REFERENCES tenants(id),
  rank          int  NOT NULL,
  chunk_id      uuid NOT NULL,
  document_id   uuid NOT NULL,
  page_no       int  NOT NULL,
  chunk_ordinal int  NOT NULL,
  vec_rank      int,
  lex_rank      int,
  sent_to_llm   boolean NOT NULL,
  PRIMARY KEY (answer_id, rank)
);
CREATE INDEX answer_candidates_chunk ON answer_candidates (chunk_id);

CREATE TABLE spans (
  span_id        uuid PRIMARY KEY,
  name           text NOT NULL,
  started_at     timestamptz NOT NULL,
  duration_ms    double precision NOT NULL,
  status         text NOT NULL CHECK (status IN ('ok','error')),
  error_code     text,
  tenant_id      uuid,
  request_id     uuid,
  job_id         bigint,
  document_id    uuid,
  page_no        int,
  attributes     jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX spans_request  ON spans (request_id) WHERE request_id IS NOT NULL;
CREATE INDEX spans_job      ON spans (job_id) WHERE job_id IS NOT NULL;
CREATE INDEX spans_document ON spans (document_id, started_at) WHERE document_id IS NOT NULL;
CREATE INDEX spans_name_time ON spans (name, started_at);

CREATE VIEW v_model_calls_per_minute AS
SELECT date_trunc('minute', started_at)                       AS minute,
       attributes->>'docqa.stage'                             AS stage,
       attributes->>'gen_ai.request.model'                    AS model,
       count(*)                                               AS calls,
       count(*) FILTER (WHERE status = 'error')               AS errors,
       count(*) FILTER (WHERE error_code IN ('rate_limited','quota_exhausted')) AS rate_limited,
       round(percentile_cont(0.5)  WITHIN GROUP (ORDER BY duration_ms)::numeric) AS p50_ms,
       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)::numeric) AS p95_ms,
       sum((attributes->>'gen_ai.usage.input_tokens')::bigint)  AS input_tokens,
       sum((attributes->>'gen_ai.usage.output_tokens')::bigint) AS output_tokens
FROM spans WHERE name = 'gen_ai.call'
GROUP BY 1, 2, 3;

CREATE VIEW v_requests_per_minute AS
SELECT date_trunc('minute', started_at)                       AS minute,
       attributes->>'http.route'                              AS route,
       count(*)                                               AS requests,
       count(*) FILTER (WHERE (attributes->>'http.response.status_code')::int >= 500) AS errors_5xx,
       round(percentile_cont(0.5)  WITHIN GROUP (ORDER BY duration_ms)::numeric) AS p50_ms,
       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)::numeric) AS p95_ms
FROM spans WHERE name = 'http.request'
GROUP BY 1, 2;
