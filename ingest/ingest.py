-- schema.sql
CREATE TABLE IF NOT EXISTS floats (
  float_id TEXT PRIMARY KEY,
  platform_number TEXT,
  deploy_date TIMESTAMP,
  properties JSONB
);

CREATE TABLE IF NOT EXISTS profiles (
  profile_id TEXT PRIMARY KEY,
  float_id TEXT REFERENCES floats(float_id),
  profile_time TIMESTAMP,
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  pressure DOUBLE PRECISION,   -- store representative pressure (e.g., surface or mean)
  depth DOUBLE PRECISION,      -- optional representative depth
  variable_name TEXT,          -- e.g., TEMP / PSAL
  variable_value DOUBLE PRECISION,
  level INTEGER,               -- level index in profile
  raw_profile JSONB            -- optional: store full level arrays as JSONB if needed
);

-- For faster queries
CREATE INDEX IF NOT EXISTS idx_profiles_time ON profiles (profile_time);
CREATE INDEX IF NOT EXISTS idx_profiles_lat ON profiles (lat);
CREATE INDEX IF NOT EXISTS idx_profiles_lon ON profiles (lon);
CREATE INDEX IF NOT EXISTS idx_profiles_float ON profiles (float_id);