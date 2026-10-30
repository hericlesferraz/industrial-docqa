-- The search rule each answer was retrieved with; rows recorded before the column searched with rare terms df <= 3.
ALTER TABLE answers ADD COLUMN search_version text NOT NULL DEFAULT 'rrf60-df3';
ALTER TABLE answers ALTER COLUMN search_version DROP DEFAULT;
