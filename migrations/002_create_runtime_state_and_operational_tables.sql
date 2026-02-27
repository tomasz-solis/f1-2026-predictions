-- Migration: Runtime state + lock + operational observability tables
-- Run via Supabase SQL Editor after migration 001.

-- Reusable updated_at trigger helper (idempotent)
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- Runtime state blobs keyed by namespace + state_key.
CREATE TABLE IF NOT EXISTS public.runtime_state (
    namespace TEXT NOT NULL,
    state_key TEXT NOT NULL,
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    version BIGINT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace, state_key)
);

CREATE INDEX IF NOT EXISTS idx_runtime_state_namespace_updated
    ON public.runtime_state(namespace, updated_at DESC);

DROP TRIGGER IF EXISTS update_runtime_state_updated_at ON public.runtime_state;
CREATE TRIGGER update_runtime_state_updated_at
    BEFORE UPDATE ON public.runtime_state
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

CREATE OR REPLACE FUNCTION public.increment_runtime_state_version()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS increment_runtime_state_version_trigger ON public.runtime_state;
CREATE TRIGGER increment_runtime_state_version_trigger
    BEFORE UPDATE ON public.runtime_state
    FOR EACH ROW
    EXECUTE FUNCTION public.increment_runtime_state_version();

-- Processing locks for concurrent-safe backlog updates.
CREATE TABLE IF NOT EXISTS public.runtime_processing_locks (
    lock_key TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_runtime_processing_locks_expires
    ON public.runtime_processing_locks(expires_at);

DROP TRIGGER IF EXISTS update_runtime_processing_locks_updated_at ON public.runtime_processing_locks;
CREATE TRIGGER update_runtime_processing_locks_updated_at
    BEFORE UPDATE ON public.runtime_processing_locks
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

-- Operational telemetry stream (counter + alert events).
CREATE TABLE IF NOT EXISTS public.operational_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL,
    event_name TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL DEFAULT '',
    labels JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_operational_events_name_created
    ON public.operational_events(event_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_operational_events_created
    ON public.operational_events(created_at DESC);

-- Secure defaults: backend writes should use service_role key.
ALTER TABLE public.runtime_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.runtime_processing_locks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.operational_events ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.runtime_state FROM anon;
REVOKE ALL ON public.runtime_state FROM authenticated;
GRANT ALL ON public.runtime_state TO service_role;

REVOKE ALL ON public.runtime_processing_locks FROM anon;
REVOKE ALL ON public.runtime_processing_locks FROM authenticated;
GRANT ALL ON public.runtime_processing_locks TO service_role;

REVOKE ALL ON public.operational_events FROM anon;
REVOKE ALL ON public.operational_events FROM authenticated;
GRANT ALL ON public.operational_events TO service_role;
