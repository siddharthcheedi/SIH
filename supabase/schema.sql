-- ============================================================
-- Women Safety Analytics MVP — Supabase Database Schema
-- ============================================================
-- Run this in your Supabase project's SQL Editor.
-- Dashboard → SQL Editor → New Query → Paste → Run

-- Sessions table: tracks each application run
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    total_alerts INTEGER    DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Alerts table: stores confirmed SOS events
CREATE TABLE IF NOT EXISTS alerts (
    id            UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id    UUID        REFERENCES sessions(id) ON DELETE SET NULL,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alert_type    TEXT        NOT NULL DEFAULT 'SOS',
    severity      TEXT        NOT NULL DEFAULT 'CRITICAL',
    total_people  INTEGER     DEFAULT 0,
    male_count    INTEGER     DEFAULT 0,
    female_count  INTEGER     DEFAULT 0,
    unknown_count INTEGER     DEFAULT 0,
    confidence    FLOAT       DEFAULT 0.0,
    status        TEXT        NOT NULL DEFAULT 'CONFIRMED',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Row Level Security (MVP — allow all for anon key)
-- Tighten these policies before any production deployment.
-- ============================================================
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts   ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all operations on sessions" ON sessions
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations on alerts" ON alerts
    FOR ALL USING (true) WITH CHECK (true);

-- ============================================================
-- Indexes for common queries
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_alerts_session   ON alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity  ON alerts(severity);
