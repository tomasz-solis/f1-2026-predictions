-- Migration: Normalize prediction + accuracy snapshot artifact keys
-- Run via Supabase SQL Editor after migrations 001-003.
--
-- Purpose:
-- 1) repair historical checkpoint rows whose keys drifted because of whitespace/casing
-- 2) collapse duplicate checkpoint states, even when they were stored as multiple versions
-- 3) canonicalize future writes before the existing unique constraint is checked
-- 4) enforce singleton prediction/accuracy rows per artifact key going forward

CREATE OR REPLACE FUNCTION public.normalize_artifact_race_name(raw_value TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT trim(regexp_replace(COALESCE(raw_value, ''), '\s+', ' ', 'g'));
$$;


CREATE OR REPLACE FUNCTION public.normalize_prediction_artifact_key(raw_key TEXT)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
SET search_path = public
AS $$
DECLARE
    parts TEXT[];
    year_part TEXT;
    race_part TEXT;
    session_part TEXT;
BEGIN
    parts := string_to_array(COALESCE(raw_key, ''), '::');
    IF array_length(parts, 1) <> 3 THEN
        RETURN COALESCE(raw_key, '');
    END IF;

    year_part := trim(parts[1]);
    race_part := public.normalize_artifact_race_name(parts[2]);
    session_part := upper(trim(parts[3]));

    RETURN year_part || '::' || race_part || '::' || session_part;
END;
$$;


CREATE OR REPLACE FUNCTION public.normalize_accuracy_snapshot_artifact_key(raw_key TEXT)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
SET search_path = public
AS $$
DECLARE
    parts TEXT[];
    year_part TEXT;
    race_part TEXT;
    checkpoint_part TEXT;
    target_part TEXT;
BEGIN
    parts := string_to_array(COALESCE(raw_key, ''), '::');
    IF array_length(parts, 1) <> 4 THEN
        RETURN COALESCE(raw_key, '');
    END IF;

    year_part := trim(parts[1]);
    race_part := public.normalize_artifact_race_name(parts[2]);
    checkpoint_part := upper(trim(parts[3]));
    target_part := lower(trim(parts[4]));

    RETURN year_part || '::' || race_part || '::' || checkpoint_part || '::' || target_part;
END;
$$;


CREATE OR REPLACE FUNCTION public.prediction_artifact_completeness(raw_data JSONB)
RETURNS INTEGER
LANGUAGE sql
IMMUTABLE
AS $$
    WITH actual_targets AS (
        SELECT COUNT(*)::INTEGER AS target_count
        FROM jsonb_each(
            CASE
                WHEN jsonb_typeof(COALESCE(raw_data->'actuals'->'targets', '{}'::jsonb)) = 'object'
                    THEN COALESCE(raw_data->'actuals'->'targets', '{}'::jsonb)
                ELSE '{}'::jsonb
            END
        )
        WHERE value IS NOT NULL
          AND value <> 'null'::jsonb
    )
    SELECT
        CASE
            WHEN jsonb_typeof(COALESCE(raw_data->'actuals'->'qualifying', 'null'::jsonb)) = 'array'
                THEN 1
            ELSE 0
        END
        + CASE
            WHEN jsonb_typeof(COALESCE(raw_data->'actuals'->'race', 'null'::jsonb)) = 'array'
                THEN 1
            ELSE 0
        END
        + COALESCE((SELECT target_count FROM actual_targets), 0);
$$;


CREATE OR REPLACE FUNCTION public.normalize_prediction_artifact_data(
    raw_data JSONB,
    normalized_key TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
SET search_path = public
AS $$
DECLARE
    payload JSONB;
    metadata JSONB;
    parts TEXT[];
    year_part TEXT;
    race_part TEXT;
    session_part TEXT;
BEGIN
    parts := string_to_array(COALESCE(normalized_key, ''), '::');
    IF array_length(parts, 1) <> 3 THEN
        RETURN COALESCE(raw_data, '{}'::jsonb);
    END IF;

    year_part := trim(parts[1]);
    race_part := public.normalize_artifact_race_name(parts[2]);
    session_part := upper(trim(parts[3]));

    payload := COALESCE(raw_data, '{}'::jsonb);
    IF jsonb_typeof(payload) <> 'object' THEN
        payload := '{}'::jsonb;
    END IF;

    metadata := COALESCE(payload->'metadata', '{}'::jsonb);
    IF jsonb_typeof(metadata) <> 'object' THEN
        metadata := '{}'::jsonb;
    END IF;

    IF year_part ~ '^\d+$' THEN
        metadata := jsonb_set(metadata, '{year}', to_jsonb(year_part::INTEGER), true);
    END IF;
    metadata := jsonb_set(metadata, '{race_name}', to_jsonb(race_part), true);
    metadata := jsonb_set(metadata, '{session_name}', to_jsonb(session_part), true);

    RETURN jsonb_set(payload, '{metadata}', metadata, true);
END;
$$;


CREATE OR REPLACE FUNCTION public.normalize_accuracy_snapshot_artifact_data(
    raw_data JSONB,
    normalized_key TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
SET search_path = public
AS $$
DECLARE
    payload JSONB;
    metadata JSONB;
    parts TEXT[];
    year_part TEXT;
    race_part TEXT;
    checkpoint_part TEXT;
    target_part TEXT;
BEGIN
    parts := string_to_array(COALESCE(normalized_key, ''), '::');
    IF array_length(parts, 1) <> 4 THEN
        RETURN COALESCE(raw_data, '{}'::jsonb);
    END IF;

    year_part := trim(parts[1]);
    race_part := public.normalize_artifact_race_name(parts[2]);
    checkpoint_part := upper(trim(parts[3]));
    target_part := lower(trim(parts[4]));

    payload := COALESCE(raw_data, '{}'::jsonb);
    IF jsonb_typeof(payload) <> 'object' THEN
        payload := '{}'::jsonb;
    END IF;

    metadata := COALESCE(payload->'metadata', '{}'::jsonb);
    IF jsonb_typeof(metadata) <> 'object' THEN
        metadata := '{}'::jsonb;
    END IF;

    IF year_part ~ '^\d+$' THEN
        metadata := jsonb_set(metadata, '{year}', to_jsonb(year_part::INTEGER), true);
    END IF;
    metadata := jsonb_set(metadata, '{race_name}', to_jsonb(race_part), true);
    metadata := jsonb_set(metadata, '{checkpoint_session}', to_jsonb(checkpoint_part), true);
    metadata := jsonb_set(metadata, '{target_key}', to_jsonb(target_part), true);

    RETURN jsonb_set(payload, '{metadata}', metadata, true);
END;
$$;


-- Repair prediction artifacts first. The winner rule prefers richer actuals, then newer rows.
WITH normalized_prediction_rows AS (
    SELECT
        id,
        run_id,
        updated_at,
        created_at,
        public.normalize_prediction_artifact_key(artifact_key) AS normalized_key,
        public.normalize_prediction_artifact_data(
            data,
            public.normalize_prediction_artifact_key(artifact_key)
        ) AS normalized_data,
        public.prediction_artifact_completeness(data) AS completeness_score
    FROM public.artifacts
    WHERE artifact_type = 'prediction'
),
ranked_prediction_rows AS (
    SELECT
        id,
        normalized_key,
        normalized_data,
        ROW_NUMBER() OVER (
            PARTITION BY normalized_key
            ORDER BY
                completeness_score DESC,
                CASE WHEN run_id IS NULL THEN 0 ELSE 1 END DESC,
                updated_at DESC,
                created_at DESC,
                id DESC
        ) AS row_rank
    FROM normalized_prediction_rows
),
deleted_prediction_rows AS (
    DELETE FROM public.artifacts artifacts
    USING ranked_prediction_rows ranked
    WHERE artifacts.id = ranked.id
      AND ranked.row_rank > 1
    RETURNING artifacts.id
)
UPDATE public.artifacts artifacts
SET
    artifact_key = ranked.normalized_key,
    version = 1,
    data = ranked.normalized_data
FROM ranked_prediction_rows ranked
WHERE artifacts.id = ranked.id
  AND ranked.row_rank = 1
  AND (
      artifacts.artifact_key IS DISTINCT FROM ranked.normalized_key
      OR artifacts.data IS DISTINCT FROM ranked.normalized_data
  );


-- Repair accuracy snapshot artifacts as well so dashboard aggregates use one canonical key.
WITH normalized_snapshot_rows AS (
    SELECT
        id,
        run_id,
        updated_at,
        created_at,
        public.normalize_accuracy_snapshot_artifact_key(artifact_key) AS normalized_key,
        public.normalize_accuracy_snapshot_artifact_data(
            data,
            public.normalize_accuracy_snapshot_artifact_key(artifact_key)
        ) AS normalized_data
    FROM public.artifacts
    WHERE artifact_type = 'accuracy_snapshot'
),
ranked_snapshot_rows AS (
    SELECT
        id,
        normalized_key,
        normalized_data,
        ROW_NUMBER() OVER (
            PARTITION BY normalized_key
            ORDER BY
                CASE WHEN run_id IS NULL THEN 0 ELSE 1 END DESC,
                updated_at DESC,
                created_at DESC,
                id DESC
        ) AS row_rank
    FROM normalized_snapshot_rows
),
deleted_snapshot_rows AS (
    DELETE FROM public.artifacts artifacts
    USING ranked_snapshot_rows ranked
    WHERE artifacts.id = ranked.id
      AND ranked.row_rank > 1
    RETURNING artifacts.id
)
UPDATE public.artifacts artifacts
SET
    artifact_key = ranked.normalized_key,
    version = 1,
    data = ranked.normalized_data
FROM ranked_snapshot_rows ranked
WHERE artifacts.id = ranked.id
  AND ranked.row_rank = 1
  AND (
      artifacts.artifact_key IS DISTINCT FROM ranked.normalized_key
      OR artifacts.version IS DISTINCT FROM 1
      OR artifacts.data IS DISTINCT FROM ranked.normalized_data
  );


CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_dashboard_singleton_key
    ON public.artifacts(artifact_type, artifact_key)
    WHERE artifact_type IN ('prediction', 'accuracy_snapshot');


CREATE OR REPLACE FUNCTION public.normalize_dashboard_artifact_row()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
    IF NEW.artifact_type = 'prediction' THEN
        NEW.artifact_key := public.normalize_prediction_artifact_key(NEW.artifact_key);
        NEW.data := public.normalize_prediction_artifact_data(NEW.data, NEW.artifact_key);
    ELSIF NEW.artifact_type = 'accuracy_snapshot' THEN
        NEW.artifact_key := public.normalize_accuracy_snapshot_artifact_key(NEW.artifact_key);
        NEW.data := public.normalize_accuracy_snapshot_artifact_data(NEW.data, NEW.artifact_key);
    END IF;

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS normalize_dashboard_artifact_row_trigger ON public.artifacts;
CREATE TRIGGER normalize_dashboard_artifact_row_trigger
    BEFORE INSERT OR UPDATE ON public.artifacts
    FOR EACH ROW
    EXECUTE FUNCTION public.normalize_dashboard_artifact_row();


-- Restrict helper function execution to service_role.
REVOKE ALL ON FUNCTION public.normalize_artifact_race_name(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.normalize_artifact_race_name(TEXT) FROM anon;
REVOKE ALL ON FUNCTION public.normalize_artifact_race_name(TEXT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.normalize_artifact_race_name(TEXT) TO service_role;

REVOKE ALL ON FUNCTION public.normalize_prediction_artifact_key(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.normalize_prediction_artifact_key(TEXT) FROM anon;
REVOKE ALL ON FUNCTION public.normalize_prediction_artifact_key(TEXT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.normalize_prediction_artifact_key(TEXT) TO service_role;

REVOKE ALL ON FUNCTION public.normalize_accuracy_snapshot_artifact_key(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.normalize_accuracy_snapshot_artifact_key(TEXT) FROM anon;
REVOKE ALL ON FUNCTION public.normalize_accuracy_snapshot_artifact_key(TEXT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.normalize_accuracy_snapshot_artifact_key(TEXT) TO service_role;

REVOKE ALL ON FUNCTION public.prediction_artifact_completeness(JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.prediction_artifact_completeness(JSONB) FROM anon;
REVOKE ALL ON FUNCTION public.prediction_artifact_completeness(JSONB) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.prediction_artifact_completeness(JSONB) TO service_role;

REVOKE ALL ON FUNCTION public.normalize_prediction_artifact_data(JSONB, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.normalize_prediction_artifact_data(JSONB, TEXT) FROM anon;
REVOKE ALL ON FUNCTION public.normalize_prediction_artifact_data(JSONB, TEXT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.normalize_prediction_artifact_data(JSONB, TEXT) TO service_role;

REVOKE ALL ON FUNCTION public.normalize_accuracy_snapshot_artifact_data(JSONB, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.normalize_accuracy_snapshot_artifact_data(JSONB, TEXT) FROM anon;
REVOKE ALL ON FUNCTION public.normalize_accuracy_snapshot_artifact_data(JSONB, TEXT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.normalize_accuracy_snapshot_artifact_data(JSONB, TEXT) TO service_role;

REVOKE ALL ON FUNCTION public.normalize_dashboard_artifact_row() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.normalize_dashboard_artifact_row() FROM anon;
REVOKE ALL ON FUNCTION public.normalize_dashboard_artifact_row() FROM authenticated;
GRANT EXECUTE ON FUNCTION public.normalize_dashboard_artifact_row() TO service_role;


-- Optional verification snippets:
-- SELECT artifact_type, artifact_key, COUNT(*)
-- FROM public.artifacts
-- WHERE artifact_type IN ('prediction', 'accuracy_snapshot')
-- GROUP BY artifact_type, artifact_key
-- HAVING COUNT(*) > 1;
