-- Migration: Harden RLS policies and grants for Supabase runtime tables
-- Run via Supabase SQL Editor after migrations 001 and 002.
--
-- Purpose:
-- 1) enforce RLS/forced RLS on runtime tables
-- 2) explicitly restrict table access to service_role
-- 3) add explicit service_role RLS policies (helps with security-advisor noise)

-- Ensure RLS is enabled and forced.
ALTER TABLE IF EXISTS public.artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.artifacts FORCE ROW LEVEL SECURITY;

ALTER TABLE IF EXISTS public.runtime_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.runtime_state FORCE ROW LEVEL SECURITY;

ALTER TABLE IF EXISTS public.runtime_processing_locks ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.runtime_processing_locks FORCE ROW LEVEL SECURITY;

ALTER TABLE IF EXISTS public.operational_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.operational_events FORCE ROW LEVEL SECURITY;

-- Restrict table privileges to service_role only.
REVOKE ALL ON TABLE public.artifacts FROM PUBLIC;
REVOKE ALL ON TABLE public.runtime_state FROM PUBLIC;
REVOKE ALL ON TABLE public.runtime_processing_locks FROM PUBLIC;
REVOKE ALL ON TABLE public.operational_events FROM PUBLIC;

REVOKE ALL ON TABLE public.artifacts FROM anon;
REVOKE ALL ON TABLE public.runtime_state FROM anon;
REVOKE ALL ON TABLE public.runtime_processing_locks FROM anon;
REVOKE ALL ON TABLE public.operational_events FROM anon;

REVOKE ALL ON TABLE public.artifacts FROM authenticated;
REVOKE ALL ON TABLE public.runtime_state FROM authenticated;
REVOKE ALL ON TABLE public.runtime_processing_locks FROM authenticated;
REVOKE ALL ON TABLE public.operational_events FROM authenticated;

GRANT ALL ON TABLE public.artifacts TO service_role;
GRANT ALL ON TABLE public.runtime_state TO service_role;
GRANT ALL ON TABLE public.runtime_processing_locks TO service_role;
GRANT ALL ON TABLE public.operational_events TO service_role;

-- Explicit RLS policies for service_role (idempotent).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'artifacts'
    ) AND NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'artifacts' AND policyname = 'service_role_all_artifacts'
    ) THEN
        CREATE POLICY service_role_all_artifacts
            ON public.artifacts
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'runtime_state'
    ) AND NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'runtime_state'
          AND policyname = 'service_role_all_runtime_state'
    ) THEN
        CREATE POLICY service_role_all_runtime_state
            ON public.runtime_state
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'runtime_processing_locks'
    ) AND NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'runtime_processing_locks'
          AND policyname = 'service_role_all_runtime_processing_locks'
    ) THEN
        CREATE POLICY service_role_all_runtime_processing_locks
            ON public.runtime_processing_locks
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'operational_events'
    ) AND NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'operational_events'
          AND policyname = 'service_role_all_operational_events'
    ) THEN
        CREATE POLICY service_role_all_operational_events
            ON public.operational_events
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;

-- Restrict helper function execution to service_role.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_proc
        WHERE pronamespace = 'public'::regnamespace
          AND proname = 'update_updated_at_column'
          AND pronargs = 0
    ) THEN
        REVOKE ALL ON FUNCTION public.update_updated_at_column() FROM PUBLIC;
        REVOKE ALL ON FUNCTION public.update_updated_at_column() FROM anon;
        REVOKE ALL ON FUNCTION public.update_updated_at_column() FROM authenticated;
        GRANT EXECUTE ON FUNCTION public.update_updated_at_column() TO service_role;
    END IF;
END
$$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_proc
        WHERE pronamespace = 'public'::regnamespace
          AND proname = 'increment_runtime_state_version'
          AND pronargs = 0
    ) THEN
        REVOKE ALL ON FUNCTION public.increment_runtime_state_version() FROM PUBLIC;
        REVOKE ALL ON FUNCTION public.increment_runtime_state_version() FROM anon;
        REVOKE ALL ON FUNCTION public.increment_runtime_state_version() FROM authenticated;
        GRANT EXECUTE ON FUNCTION public.increment_runtime_state_version() TO service_role;
    END IF;
END
$$;

-- Optional verification snippets:
-- SELECT tablename, policyname, roles, cmd
-- FROM pg_policies
-- WHERE schemaname = 'public'
--   AND tablename IN ('artifacts', 'runtime_state', 'runtime_processing_locks', 'operational_events')
-- ORDER BY tablename, policyname;
--
-- SELECT table_name, grantee, privilege_type
-- FROM information_schema.role_table_grants
-- WHERE table_schema = 'public'
--   AND table_name IN ('artifacts', 'runtime_state', 'runtime_processing_locks', 'operational_events')
-- ORDER BY table_name, grantee, privilege_type;
