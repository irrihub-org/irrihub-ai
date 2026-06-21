CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS kb_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    category TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    is_active BOOLEAN NOT NULL DEFAULT true,
    uploader_id TEXT,
    uploaded_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES kb_sources(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(384),
    category TEXT,
    language TEXT DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sheng_map (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slang TEXT NOT NULL UNIQUE,
    standard TEXT NOT NULL,
    language TEXT DEFAULT 'sw',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id TEXT UNIQUE NOT NULL,
    is_human_managed BOOLEAN DEFAULT false,
    language TEXT DEFAULT 'en',
    last_active TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_gaps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    channel TEXT,
    confidence FLOAT,
    failure_reason TEXT,
    escalated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL,
    message TEXT,
    channel TEXT,
    user_ref TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS consent_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_ref TEXT NOT NULL,
    channel TEXT NOT NULL,
    consent_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS analytics_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id TEXT,
    intent TEXT,
    confidence FLOAT,
    route TEXT,
    channel TEXT,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kb_gaps_date ON kb_gaps (escalated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_conv ON sessions (conversation_id);
CREATE INDEX IF NOT EXISTS idx_analytics_date ON analytics_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_date ON security_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_kb_sources_active ON kb_sources (is_active) WHERE is_active = true;
