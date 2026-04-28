-- Per-session breakdown with duration and engagement classification.
-- Single-event sessions read as 0s by definition; flagged as 'bounce'.
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


-- Aggregate stats: total sessions, bounce rate, median session length.
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


-- Most-clicked predictions in the last 30 days.
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


-- Page popularity in the last 30 days.
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
