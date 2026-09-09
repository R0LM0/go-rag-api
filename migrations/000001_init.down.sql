-- 000001_init.down.sql: reverse of 000001_init.up.sql.
-- The vector extension is intentionally NOT dropped: it is a cluster-wide
-- object and other databases or future migrations may still depend on it.
DROP TABLE IF EXISTS chunks;
