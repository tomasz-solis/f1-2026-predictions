-- Harden app usage telemetry to match the runtime-table RLS posture.
-- app_events stores only coarse client-channel telemetry, not raw user-agent.

ALTER TABLE IF EXISTS public.app_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.app_events FORCE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.app_events FROM PUBLIC;
REVOKE ALL ON TABLE public.app_events FROM anon;
REVOKE ALL ON TABLE public.app_events FROM authenticated;

GRANT ALL ON TABLE public.app_events TO service_role;

DO $$
BEGIN
    IF to_regclass('public.app_events_id_seq') IS NOT NULL THEN
        REVOKE ALL ON SEQUENCE public.app_events_id_seq FROM PUBLIC;
        REVOKE ALL ON SEQUENCE public.app_events_id_seq FROM anon;
        REVOKE ALL ON SEQUENCE public.app_events_id_seq FROM authenticated;
        GRANT ALL ON SEQUENCE public.app_events_id_seq TO service_role;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'app_events'
    )
    AND NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'app_events'
          AND policyname = 'service_role_all_app_events'
    ) THEN
        CREATE POLICY service_role_all_app_events
            ON public.app_events
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END $$;

