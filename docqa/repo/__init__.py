"""The data-access layer: every SQL statement on customer data lives here, filtered by `tenant_id`.

Declared exceptions to the tenant filter: `ocr_cache` (keyed by file content), `maintenance` (cleanup across
tenants) and `tenants.tenant_for_key` (resolves the tenant). The queue's `claim` and `requeue_expired` serve
every tenant too.
"""
