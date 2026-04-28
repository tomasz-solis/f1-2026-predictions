-- ============================================================
-- Per-session breakdown with duration and engagement classification.
-- Single-event sessions read as 0s; flagged as 'bounce'.
-- ============================================================
select
    session_id,
    count(*) as event_count,
    min(occurred_at) as session_started,
    max(occurred_at) as session_ended,
    extract(epoch from (max(occurred_at) - min(occurred_at)))::int as duration_seconds,
    array_agg(distinct event_type) as event_types,
    array_agg(distinct page) filter (where page is not null) as pages_visited,
    case
        when count(*) = 1 then 'bounce'
        when extract(epoch from (max(occurred_at) - min(occurred_at))) < 30 then 'short'
        when extract(epoch from (max(occurred_at) - min(occurred_at))) < 300 then 'medium'
        else 'engaged'
    end as session_class
from public.app_events
where occurred_at >= now() - interval '30 days'
group by session_id
order by session_started desc;


-- ============================================================
-- Aggregate stats: total sessions, bounce rate, median duration.
-- ============================================================
with sessions as (
    select
        session_id,
        count(*) as event_count,
        extract(epoch from (max(occurred_at) - min(occurred_at)))::int as duration_seconds
    from public.app_events
    where occurred_at >= now() - interval '7 days'
    group by session_id
)
select
    count(*) as total_sessions,
    count(*) filter (where event_count = 1) as bounces,
    round(100.0 * count(*) filter (where event_count = 1) / nullif(count(*), 0), 1) as bounce_rate_pct,
    percentile_cont(0.5) within group (order by duration_seconds)
        filter (where event_count > 1) as median_duration_s,
    avg(duration_seconds) filter (where event_count > 1)::int as mean_duration_s
from sessions;


-- ============================================================
-- Most-clicked predictions in the last 30 days.
-- ============================================================
select
    payload->>'race' as race,
    payload->>'weather' as weather,
    (payload->>'is_sprint')::boolean as is_sprint,
    count(*) as clicks
from public.app_events
where event_type = 'predict_clicked'
  and occurred_at >= now() - interval '30 days'
group by 1, 2, 3
order by clicks desc;


-- ============================================================
-- Page popularity in the last 30 days.
-- ============================================================
select
    page,
    count(*) as views,
    count(distinct session_id) as unique_sessions
from public.app_events
where event_type = 'page_view'
  and occurred_at >= now() - interval '30 days'
  and page is not null
group by page
order by views desc;


-- ============================================================
-- ATTRIBUTION: traffic source breakdown.
-- Direct = no utm_source captured (typed URL, bookmark, etc.)
-- ============================================================
select
    coalesce(payload->>'utm_source', 'direct') as source,
    coalesce(payload->>'utm_medium', '-') as medium,
    count(distinct session_id) as sessions,
    count(*) as events
from public.app_events
where occurred_at >= now() - interval '30 days'
group by 1, 2
order by sessions desc;


-- ============================================================
-- ATTRIBUTION: predict-click conversion by source.
-- "Did users from Instagram actually engage with predictions?"
-- ============================================================
with session_source as (
    -- One source per session: pick the first event's UTM. Sessions with
    -- no UTM are 'direct'.
    select distinct on (session_id)
        session_id,
        coalesce(payload->>'utm_source', 'direct') as source
    from public.app_events
    where occurred_at >= now() - interval '30 days'
    order by session_id, occurred_at asc
),
session_engagement as (
    select
        s.session_id,
        s.source,
        count(*) filter (where e.event_type = 'predict_clicked') as predict_clicks,
        count(*) filter (where e.event_type = 'page_view') as page_views
    from session_source s
    join public.app_events e using (session_id)
    where e.occurred_at >= now() - interval '30 days'
    group by s.session_id, s.source
)
select
    source,
    count(*) as sessions,
    sum(predict_clicks) as total_predict_clicks,
    count(*) filter (where predict_clicks > 0) as sessions_with_predict,
    round(
        100.0 * count(*) filter (where predict_clicks > 0) / nullif(count(*), 0),
        1
    ) as predict_conversion_pct,
    round(avg(page_views)::numeric, 1) as avg_pages_per_session
from session_engagement
group by source
order by sessions desc;


-- ============================================================
-- ATTRIBUTION: sanity-check UTMs against User-Agent.
-- Useful to detect if Instagram in-app browser visits actually
-- match utm_source=instagram (catches broken share links).
-- ============================================================
select
    coalesce(payload->>'utm_source', 'direct') as utm_source,
    case
        when user_agent ilike '%instagram%' then 'IG in-app'
        when user_agent ilike '%threads%' then 'Threads in-app'
        when user_agent ilike '%fbav%' then 'FB in-app'
        else 'browser'
    end as ua_bucket,
    count(distinct session_id) as sessions
from public.app_events
where occurred_at >= now() - interval '30 days'
group by 1, 2
order by sessions desc;
