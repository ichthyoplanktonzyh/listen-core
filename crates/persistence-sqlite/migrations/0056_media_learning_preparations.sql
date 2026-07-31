CREATE TABLE media_learning_preparations (
  id TEXT PRIMARY KEY,
  target_key TEXT NOT NULL,
  input_fingerprint TEXT NOT NULL,
  status TEXT NOT NULL CHECK (
    status IN ('queued','running','cancelling','completed','failed','cancelled')
  ),
  revision INTEGER NOT NULL CHECK (revision >= 0),
  retry_of_id TEXT REFERENCES media_learning_preparations(id),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  preparation_json TEXT NOT NULL
);

CREATE INDEX media_learning_preparations_target_history
  ON media_learning_preparations(target_key, created_at_ms DESC, id);

CREATE UNIQUE INDEX media_learning_preparations_active_target
  ON media_learning_preparations(target_key)
  WHERE status IN ('queued','running','cancelling');
