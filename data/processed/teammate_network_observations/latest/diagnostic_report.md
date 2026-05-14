# Matched-Lap Bulk Extraction Diagnostics

- Built at: 2026-05-13T15:39:13.191654+00:00
- Target sessions: 184
- Loaded sessions: 184
- Error sessions: 0
- Raw rows: 28336
- Matched-pair rows: 27965
- Skipped-pair rows: 371
- Valid aggregate rows: 1490

## Matched-Pair Distribution

```json
{
  "overall": {
    "count": 1490,
    "min": 3.0,
    "p25": 4.0,
    "median": 15.0,
    "p75": 32.0,
    "max": 67.0
  },
  "by_session_kind_weather": [
    {
      "session_kind": "qualifying",
      "weather_bucket": "dry",
      "count": 627,
      "min": 3.0,
      "p25": 3.0,
      "median": 3.0,
      "p75": 4.0,
      "max": 6.0
    },
    {
      "session_kind": "qualifying",
      "weather_bucket": "wet",
      "count": 25,
      "min": 3.0,
      "p25": 3.0,
      "median": 4.0,
      "p75": 4.0,
      "max": 7.0
    },
    {
      "session_kind": "race",
      "weather_bucket": "dry",
      "count": 796,
      "min": 8.0,
      "p25": 23.0,
      "median": 31.0,
      "p75": 39.0,
      "max": 67.0
    },
    {
      "session_kind": "race",
      "weather_bucket": "wet",
      "count": 42,
      "min": 8.0,
      "p25": 9.0,
      "median": 10.5,
      "p75": 14.75,
      "max": 28.0
    }
  ]
}
```

## Matched-Gap SE Distribution

```json
{
  "count": 1490,
  "min": 0.02,
  "p25": 0.09255418338947337,
  "median": 0.15705455350159786,
  "p75": 0.27190597361246194,
  "max": 2.279812242717818
}
```

## Skip Reasons

```json
[
  {
    "session_kind": "qualifying",
    "skip_reason": "insufficient_matched_pairs",
    "count": 248
  },
  {
    "session_kind": "qualifying",
    "skip_reason": "missing_lap_time_data",
    "count": 8
  },
  {
    "session_kind": "qualifying",
    "skip_reason": "single_car_session",
    "count": 1
  },
  {
    "session_kind": "race",
    "skip_reason": "insufficient_matched_pairs",
    "count": 50
  },
  {
    "session_kind": "race",
    "skip_reason": "missing_lap_time_data",
    "count": 48
  },
  {
    "session_kind": "race",
    "skip_reason": "no_compound_overlap",
    "count": 5
  },
  {
    "session_kind": "race",
    "skip_reason": "single_car_session",
    "count": 2
  },
  {
    "session_kind": "race",
    "skip_reason": "teammate_dnf_no_matched_laps",
    "count": 9
  }
]
```

## Zero-Observation Sessions

```json
[]
```

## Weather Buckets

```json
[
  {
    "session_kind": "qualifying",
    "weather_bucket": "dry",
    "count": 2296
  },
  {
    "session_kind": "qualifying",
    "weather_bucket": "wet",
    "count": 141
  },
  {
    "session_kind": "race",
    "weather_bucket": "dry",
    "count": 24847
  },
  {
    "session_kind": "race",
    "weather_bucket": "wet",
    "count": 681
  }
]
```

## Connected Components

```json
[
  {
    "session_kind": "qualifying",
    "weather_bucket": "dry",
    "n_components": 2,
    "component_sizes": [
      29,
      2
    ],
    "component_observation_counts": [
      584,
      43
    ],
    "component_observation_shares": [
      0.9314194577352473,
      0.0685805422647528
    ],
    "components": [
      [
        "ALB",
        "ALO",
        "ANT",
        "BEA",
        "BOR",
        "COL",
        "DEV",
        "DOO",
        "GAS",
        "HAD",
        "HAM",
        "HUL",
        "LAT",
        "LAW",
        "LEC",
        "MAG",
        "MSC",
        "NOR",
        "OCO",
        "PER",
        "PIA",
        "RIC",
        "RUS",
        "SAI",
        "SAR",
        "STR",
        "TSU",
        "VER",
        "VET"
      ],
      [
        "BOT",
        "ZHO"
      ]
    ]
  },
  {
    "session_kind": "qualifying",
    "weather_bucket": "wet",
    "n_components": 8,
    "component_sizes": [
      6,
      3,
      3,
      3,
      2,
      2,
      2,
      2
    ],
    "component_observation_counts": [
      8,
      2,
      2,
      2,
      3,
      2,
      4,
      2
    ],
    "component_observation_shares": [
      0.32,
      0.08,
      0.08,
      0.08,
      0.12,
      0.08,
      0.16,
      0.08
    ],
    "components": [
      [
        "GAS",
        "LAW",
        "NOR",
        "PIA",
        "RIC",
        "TSU"
      ],
      [
        "ALB",
        "LAT",
        "SAR"
      ],
      [
        "ALO",
        "STR",
        "VET"
      ],
      [
        "BEA",
        "HUL",
        "MAG"
      ],
      [
        "BOT",
        "ZHO"
      ],
      [
        "HAM",
        "RUS"
      ],
      [
        "LEC",
        "SAI"
      ],
      [
        "PER",
        "VER"
      ]
    ]
  },
  {
    "session_kind": "race",
    "weather_bucket": "dry",
    "n_components": 2,
    "component_sizes": [
      29,
      2
    ],
    "component_observation_counts": [
      738,
      58
    ],
    "component_observation_shares": [
      0.9271356783919598,
      0.0728643216080402
    ],
    "components": [
      [
        "ALB",
        "ALO",
        "ANT",
        "BEA",
        "BOR",
        "COL",
        "DEV",
        "DOO",
        "GAS",
        "HAD",
        "HAM",
        "HUL",
        "LAT",
        "LAW",
        "LEC",
        "MAG",
        "MSC",
        "NOR",
        "OCO",
        "PER",
        "PIA",
        "RIC",
        "RUS",
        "SAI",
        "SAR",
        "STR",
        "TSU",
        "VER",
        "VET"
      ],
      [
        "BOT",
        "ZHO"
      ]
    ]
  },
  {
    "session_kind": "race",
    "weather_bucket": "wet",
    "n_components": 4,
    "component_sizes": [
      17,
      5,
      2,
      2
    ],
    "component_observation_counts": [
      28,
      10,
      1,
      3
    ],
    "component_observation_shares": [
      0.6666666666666666,
      0.23809523809523808,
      0.023809523809523808,
      0.07142857142857142
    ],
    "components": [
      [
        "ALO",
        "BEA",
        "BOR",
        "DEV",
        "GAS",
        "HUL",
        "LAW",
        "MAG",
        "MSC",
        "NOR",
        "OCO",
        "PER",
        "PIA",
        "RIC",
        "STR",
        "TSU",
        "VER"
      ],
      [
        "ANT",
        "HAM",
        "LEC",
        "RUS",
        "SAI"
      ],
      [
        "ALB",
        "SAR"
      ],
      [
        "BOT",
        "ZHO"
      ]
    ]
  }
]
```

## Extraction Errors

```json
[]
```
