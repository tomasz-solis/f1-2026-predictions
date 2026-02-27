-- Migration: Create artifacts table for JSONB artifact store
-- Run this via Supabase SQL Editor (Dashboard -> SQL Editor -> New Query -> Paste -> Run)

-- Create artifacts table
CREATE TABLE IF NOT EXISTS public.artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_type VARCHAR(50) NOT NULL,
    artifact_key VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    run_id UUID,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint: one version per artifact
    UNIQUE(artifact_type, artifact_key, version)
);

-- Create indexes for query performance
CREATE INDEX IF NOT EXISTS idx_artifacts_type_key
    ON public.artifacts(artifact_type, artifact_key);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_id
    ON public.artifacts(run_id)
    WHERE run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_artifacts_created_at
    ON public.artifacts(created_at DESC);

-- Optional: GIN index for JSONB queries (add later if needed)
-- CREATE INDEX IF NOT EXISTS idx_artifacts_data_gin
--     ON artifacts USING GIN (data);

-- Add trigger to auto-update updated_at timestamp
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

DROP TRIGGER IF EXISTS update_artifacts_updated_at ON public.artifacts;
CREATE TRIGGER update_artifacts_updated_at
    BEFORE UPDATE ON public.artifacts
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

-- Secure defaults: backend uses service_role key, not anon/authenticated clients.
ALTER TABLE public.artifacts ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.artifacts FROM anon;
REVOKE ALL ON public.artifacts FROM authenticated;
GRANT ALL ON public.artifacts TO service_role;

-- Verify table was created
SELECT
    'artifacts table created successfully' AS status,
    COUNT(*) AS row_count
FROM public.artifacts;
