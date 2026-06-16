-- App usage telemetry. No PII; session_id is a short hashed UUID generated
-- per Streamlit session_state, scoped to the browser tab. user_agent stores a
-- coarse client-channel label only, not the raw browser user-agent string.
create table if not exists public.app_events (
    id            bigserial primary key,
    occurred_at   timestamptz not null default now(),
    session_id    text        not null,
    event_type    text        not null,           -- 'page_view', 'predict_clicked', etc.
    page          text        null,               -- nav page name when relevant
    payload       jsonb       not null default '{}'::jsonb,
    model_version text        null,
    user_agent    text        null
);

create index if not exists app_events_occurred_at_idx
    on public.app_events (occurred_at desc);
create index if not exists app_events_event_type_idx
    on public.app_events (event_type);
create index if not exists app_events_session_idx
    on public.app_events (session_id);

alter table public.app_events enable row level security;

create policy "service_role_writes_app_events"
    on public.app_events for insert
    to service_role
    with check (true);

create policy "service_role_reads_app_events"
    on public.app_events for select
    to service_role
    using (true);
