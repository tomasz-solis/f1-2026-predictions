"""Build a seconds-native fallback for debut-season drivers without prior nodes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OBSERVATIONS = Path(
    "data/processed/teammate_network_observations/latest/aggregated_observations.csv"
)
DEFAULT_PRIOR = Path("data/processed/teammate_network_prior/latest.json")
DEFAULT_DRIVER_DEBUTS = Path("data/driver_debuts.json")
DEFAULT_OUTPUT_JSON = Path("data/processed/driver_seconds_rookie_fallback/latest.json")
DEFAULT_OUTPUT_MD = Path("data/processed/driver_seconds_rookie_fallback/latest.md")
_REQUIRED_OBSERVATION_COLUMNS = {
    "reference_driver_code",
    "comparison_driver_code",
    "year",
    "session_kind",
    "matched_gap_median_s",
    "n_matched_pairs",
    "weather_bucket",
    "skip_reason",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface for the rookie fallback artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=DEFAULT_OBSERVATIONS,
        help="Aggregate matched-lap observations used by the teammate-network prior.",
    )
    parser.add_argument(
        "--prior-file",
        type=Path,
        default=DEFAULT_PRIOR,
        help="Teammate-network prior used to anchor teammate seconds values.",
    )
    parser.add_argument(
        "--driver-debuts",
        type=Path,
        default=DEFAULT_DRIVER_DEBUTS,
        help="Driver debut artifact containing driver_debuts.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Generated rookie fallback JSON artifact.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Human-readable rookie fallback summary.",
    )
    return parser


def build_driver_seconds_rookie_fallback(
    *,
    observations: pd.DataFrame,
    teammate_prior: Mapping[str, Any],
    driver_debuts: Mapping[str, Any],
    built_at: str | None = None,
) -> dict[str, Any]:
    """Build a fallback from debut-season observations in the prior construct.

    One debut-season matched-lap row implies a rookie seconds rating when the
    teammate's fitted teammate-network rating is used as the anchor. The
    fallback mean is the median of per-rookie medians, so rookies with longer
    schedules do not dominate drivers with shorter debut samples.
    """
    scope = _historical_scope(teammate_prior)
    min_observations = _min_driver_observations(teammate_prior)
    normalized_debuts = _normalize_driver_debuts(driver_debuts)
    valid_rows = _valid_observation_rows(observations)
    race = build_session_fallback(
        observations=valid_rows,
        network=_network(teammate_prior, "race_network"),
        driver_debuts=normalized_debuts,
        session_kind="race",
        cohort_years=scope,
    )
    qualifying = build_session_fallback(
        observations=valid_rows,
        network=_network(teammate_prior, "quali_network"),
        driver_debuts=normalized_debuts,
        session_kind="qualifying",
        cohort_years=scope,
    )
    return {
        "artifact_type": "driver_seconds_rookie_fallback",
        "schema_version": 1,
        "built_at": built_at or datetime.now(UTC).isoformat(),
        "sign_convention": "positive_mu_s_means_faster_than_field",
        "cohort_years": list(scope),
        "method": {
            "mean_policy": "median_of_debut_season_rookie_medians",
            "sigma_policy": "max_of_rookie_median_robust_spread_and_prior_sigma_median",
            "construct": (
                "dry aggregate matched-lap teammate residuals anchored by the "
                "teammate-network seconds prior"
            ),
        },
        "promotion_policy": {
            "evidence_gate": "construct_aligned_driver_observations_by_session_kind",
            "min_observations": min_observations,
            "note": (
                "Keep the fallback for one session kind until live seconds-state "
                "updates have at least this many construct-aligned driver observations."
            ),
        },
        "race": race,
        "qualifying": qualifying,
    }


def build_session_fallback(
    *,
    observations: pd.DataFrame,
    network: Mapping[str, Any],
    driver_debuts: Mapping[str, int],
    session_kind: str,
    cohort_years: tuple[int, ...],
) -> dict[str, Any]:
    """Build one race or qualifying rookie fallback."""
    cohort_rows = observations[
        observations["session_kind"].eq(session_kind) & observations["year"].isin(cohort_years)
    ].copy()
    implied_rows = implied_rookie_seconds_rows(
        observations=cohort_rows,
        network=network,
        driver_debuts=driver_debuts,
    )
    if implied_rows.empty:
        raise ValueError(f"No debut-season rookie rows available for {session_kind}")

    cohort = _rookie_cohort(implied_rows, network)
    if not cohort:
        raise ValueError(f"No rookie cohort estimates available for {session_kind}")

    rookie_medians = [float(row["median_mu_s"]) for row in cohort]
    prior_sigmas = [
        float(row["teammate_network_sigma_s"])
        for row in cohort
        if row["teammate_network_sigma_s"] is not None
    ]
    robust_spread = _robust_sigma(rookie_medians)
    prior_sigma_median = float(np.median(prior_sigmas)) if prior_sigmas else 0.0
    sigma_s = max(robust_spread, prior_sigma_median)
    if not np.isfinite(sigma_s) or sigma_s <= 0.0:
        raise ValueError(f"Could not derive a positive fallback sigma for {session_kind}")

    return {
        "mu_s": float(np.median(rookie_medians)),
        "sigma_s": float(sigma_s),
        "n_rookies": len(cohort),
        "n_implied_observations": int(len(implied_rows)),
        "robust_rookie_spread_s": robust_spread,
        "teammate_network_sigma_median_s": prior_sigma_median,
        "cohort": cohort,
    }


def implied_rookie_seconds_rows(
    *,
    observations: pd.DataFrame,
    network: Mapping[str, Any],
    driver_debuts: Mapping[str, int],
) -> pd.DataFrame:
    """Return row-level debut-season rookie ratings implied by teammate rows."""
    drivers = network.get("drivers")
    if not isinstance(drivers, Mapping):
        raise ValueError("Teammate network is missing a 'drivers' mapping")

    implied: list[dict[str, Any]] = []
    for row in observations.itertuples(index=False):
        year = int(row.year)
        reference = str(row.reference_driver_code)
        comparison = str(row.comparison_driver_code)
        gap_s = float(row.matched_gap_median_s)

        comparison_mu = _network_value(drivers, comparison, "mu_s")
        if driver_debuts.get(reference) == year and comparison_mu is not None:
            implied.append(
                {
                    "driver_code": reference,
                    "debut_year": year,
                    "teammate_code": comparison,
                    "implied_mu_s": gap_s + comparison_mu,
                }
            )

        reference_mu = _network_value(drivers, reference, "mu_s")
        if driver_debuts.get(comparison) == year and reference_mu is not None:
            implied.append(
                {
                    "driver_code": comparison,
                    "debut_year": year,
                    "teammate_code": reference,
                    "implied_mu_s": reference_mu - gap_s,
                }
            )
    return pd.DataFrame(
        implied,
        columns=["driver_code", "debut_year", "teammate_code", "implied_mu_s"],
    )


def format_rookie_fallback_summary(artifact: Mapping[str, Any]) -> str:
    """Format a compact Markdown summary for the fallback artifact."""
    years = ", ".join(str(year) for year in artifact["cohort_years"])
    policy = artifact["promotion_policy"]
    lines = [
        "# Driver Seconds Rookie Fallback",
        "",
        f"Built at: `{artifact['built_at']}`",
        f"Cohort years: {years}",
        "",
        (
            "This artifact gives unseen debut-season drivers a data-derived seconds "
            "state when the teammate-network prior has no driver node."
        ),
        "",
        "## Fallback values",
        "",
        "| Session | Mean (s) | Sigma (s) | Rookies | Implied observations |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("Race", "race"), ("Qualifying", "qualifying")):
        row = artifact[key]
        lines.append(
            f"| {label} | {row['mu_s']:.6f} | {row['sigma_s']:.6f} | "
            f"{row['n_rookies']} | {row['n_implied_observations']} |"
        )

    lines.extend(
        [
            "",
            "## Cohort",
            "",
            "| Session | Driver | Debut | Rows | Median implied mean (s) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for label, key in (("Race", "race"), ("Qualifying", "qualifying")):
        for row in artifact[key]["cohort"]:
            lines.append(
                f"| {label} | `{row['driver_code']}` | {row['debut_year']} | "
                f"{row['n_implied_observations']} | {row['median_mu_s']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Promotion policy",
            "",
            (
                f"Replace the fallback per session kind after at least "
                f"`{policy['min_observations']}` construct-aligned driver observations."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Build and write the rookie fallback artifacts."""
    args = build_parser().parse_args()
    prior = _read_json_object(args.prior_file)
    debuts = _read_driver_debuts(args.driver_debuts)
    artifact = build_driver_seconds_rookie_fallback(
        observations=pd.read_csv(args.observations),
        teammate_prior=prior,
        driver_debuts=debuts,
    )
    artifact["sources"] = {
        "observations": str(args.observations),
        "teammate_network_prior": str(args.prior_file),
        "driver_debuts": str(args.driver_debuts),
    }
    _write_json_object(args.output_json, artifact)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(format_rookie_fallback_summary(artifact), encoding="utf-8")
    print(format_rookie_fallback_summary(artifact))


def _valid_observation_rows(observations: pd.DataFrame) -> pd.DataFrame:
    """Return usable dry aggregate rows for fallback construction."""
    missing = sorted(_REQUIRED_OBSERVATION_COLUMNS.difference(observations.columns))
    if missing:
        raise ValueError(f"Aggregate observations are missing columns: {missing}")

    rows = observations.copy()
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce")
    rows["matched_gap_median_s"] = pd.to_numeric(
        rows["matched_gap_median_s"],
        errors="coerce",
    )
    rows["n_matched_pairs"] = pd.to_numeric(rows["n_matched_pairs"], errors="coerce")
    mask = (
        rows["skip_reason"].isna()
        & rows["weather_bucket"].eq("dry")
        & rows["session_kind"].isin(["race", "qualifying"])
        & rows["year"].notna()
        & rows["matched_gap_median_s"].notna()
        & rows["n_matched_pairs"].gt(0)
    )
    valid = rows[mask].copy()
    valid["year"] = valid["year"].astype(int)
    return valid


def _rookie_cohort(
    implied_rows: pd.DataFrame,
    network: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate row-level rookie estimates into one record per rookie."""
    drivers = network.get("drivers")
    if not isinstance(drivers, Mapping):
        raise ValueError("Teammate network is missing a 'drivers' mapping")

    cohort: list[dict[str, Any]] = []
    grouped = implied_rows.groupby(["driver_code", "debut_year"], sort=True)
    for (driver_code, debut_year), rows in grouped:
        sigma_s = _network_value(drivers, str(driver_code), "sigma_s")
        cohort.append(
            {
                "driver_code": str(driver_code),
                "debut_year": int(debut_year),
                "n_implied_observations": int(len(rows)),
                "teammate_codes": sorted({str(code) for code in rows["teammate_code"]}),
                "median_mu_s": float(np.median(rows["implied_mu_s"].astype(float))),
                "teammate_network_sigma_s": sigma_s,
            }
        )
    return cohort


def _network(prior: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one prior network payload."""
    network = prior.get(key)
    if not isinstance(network, Mapping):
        raise ValueError(f"Teammate prior is missing '{key}'")
    return network


def _network_value(
    drivers: Mapping[str, Any],
    driver_code: str,
    key: str,
) -> float | None:
    """Read a finite numeric teammate-network driver value."""
    state = drivers.get(driver_code)
    if not isinstance(state, Mapping):
        return None
    value = _finite_float(state.get(key))
    return value


def _historical_scope(prior: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the prior historical scope as inclusive cohort years."""
    config = prior.get("config")
    historical_scope = config.get("historical_scope") if isinstance(config, Mapping) else None
    if not isinstance(historical_scope, Mapping):
        raise ValueError("Teammate prior is missing config.historical_scope")
    start_raw = historical_scope.get("start")
    end_raw = historical_scope.get("end")
    if start_raw is None or end_raw is None:
        raise ValueError("Teammate prior historical scope is missing a bound")
    start = int(start_raw)
    end = int(end_raw)
    if end < start:
        raise ValueError("Teammate prior historical scope ends before it starts")
    return tuple(range(start, end + 1))


def _min_driver_observations(prior: Mapping[str, Any]) -> int:
    """Read the prior evidence threshold used for low-observation drivers."""
    config = prior.get("config")
    raw_value = config.get("min_driver_observations") if isinstance(config, Mapping) else None
    if raw_value is None:
        raise ValueError("Teammate prior is missing config.min_driver_observations")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Teammate prior is missing config.min_driver_observations") from exc
    if value <= 0:
        raise ValueError("Teammate prior min_driver_observations must be positive")
    return value


def _normalize_driver_debuts(driver_debuts: Mapping[str, Any]) -> dict[str, int]:
    """Normalize driver debut years from an artifact mapping."""
    normalized: dict[str, int] = {}
    for driver_code, debut_year in driver_debuts.items():
        code = str(driver_code).strip().upper()
        if not code:
            continue
        try:
            normalized[code] = int(debut_year)
        except (TypeError, ValueError):
            continue
    if not normalized:
        raise ValueError("Driver debut payload contains no usable debut years")
    return normalized


def _read_driver_debuts(path: Path) -> dict[str, Any]:
    """Read the driver debut mapping from its JSON artifact."""
    payload = _read_json_object(path)
    driver_debuts = payload.get("driver_debuts")
    if not isinstance(driver_debuts, dict):
        raise ValueError(f"{path} is missing a 'driver_debuts' mapping")
    return driver_debuts


def _robust_sigma(values: list[float]) -> float:
    """Return MAD-derived robust sigma for a short numeric cohort."""
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return float(1.4826 * mad)


def _finite_float(value: Any) -> float | None:
    """Convert one finite numeric value."""
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    with open(path, encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json_object(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
