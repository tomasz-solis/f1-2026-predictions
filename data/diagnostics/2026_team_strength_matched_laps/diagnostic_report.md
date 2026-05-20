# Matched-Lap Bulk Extraction Diagnostics

- Built at: 2026-05-20T07:21:11.669948+00:00
- Target sessions: 8
- Loaded sessions: 8
- Error sessions: 0
- Raw rows: 1021
- Matched-pair rows: 999
- Skipped-pair rows: 22
- Valid aggregate rows: 66

## Matched-Pair Distribution

```json
{
  "overall": {
    "count": 66,
    "min": 3.0,
    "p25": 4.0,
    "median": 5.5,
    "p75": 28.75,
    "max": 40.0
  },
  "by_session_kind_weather": [
    {
      "session_kind": "qualifying",
      "weather_bucket": "dry",
      "count": 34,
      "min": 3.0,
      "p25": 3.0,
      "median": 4.0,
      "p75": 4.0,
      "max": 6.0
    },
    {
      "session_kind": "race",
      "weather_bucket": "dry",
      "count": 32,
      "min": 9.0,
      "p25": 20.0,
      "median": 29.0,
      "p75": 33.5,
      "max": 40.0
    }
  ]
}
```

## Matched-Gap SE Distribution

```json
{
  "count": 66,
  "min": 0.03980444629339392,
  "p25": 0.08713707703952292,
  "median": 0.13343299279256393,
  "p75": 0.2602519137265036,
  "max": 1.419160783897572
}
```

## Skip Reasons

```json
[
  {
    "session_kind": "qualifying",
    "skip_reason": "insufficient_matched_pairs",
    "count": 8
  },
  {
    "session_kind": "qualifying",
    "skip_reason": "missing_lap_time_data",
    "count": 2
  },
  {
    "session_kind": "race",
    "skip_reason": "insufficient_matched_pairs",
    "count": 7
  },
  {
    "session_kind": "race",
    "skip_reason": "missing_lap_time_data",
    "count": 5
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
    "count": 128
  },
  {
    "session_kind": "race",
    "weather_bucket": "dry",
    "count": 871
  }
]
```

## Connected Components

```json
[
  {
    "session_kind": "qualifying",
    "weather_bucket": "dry",
    "n_components": 10,
    "component_sizes": [
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2
    ],
    "component_observation_counts": [
      3,
      2,
      4,
      4,
      3,
      4,
      3,
      4,
      3,
      4
    ],
    "component_observation_shares": [
      0.08823529411764706,
      0.058823529411764705,
      0.11764705882352941,
      0.11764705882352941,
      0.08823529411764706,
      0.11764705882352941,
      0.08823529411764706,
      0.11764705882352941,
      0.08823529411764706,
      0.11764705882352941
    ],
    "components": [
      [
        "ALB",
        "SAI"
      ],
      [
        "ALO",
        "STR"
      ],
      [
        "ANT",
        "RUS"
      ],
      [
        "BEA",
        "OCO"
      ],
      [
        "BOR",
        "HUL"
      ],
      [
        "COL",
        "GAS"
      ],
      [
        "HAD",
        "VER"
      ],
      [
        "HAM",
        "LEC"
      ],
      [
        "LAW",
        "LIN"
      ],
      [
        "NOR",
        "PIA"
      ]
    ]
  },
  {
    "session_kind": "race",
    "weather_bucket": "dry",
    "n_components": 11,
    "component_sizes": [
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2
    ],
    "component_observation_counts": [
      3,
      2,
      4,
      4,
      1,
      4,
      3,
      2,
      4,
      3,
      2
    ],
    "component_observation_shares": [
      0.09375,
      0.0625,
      0.125,
      0.125,
      0.03125,
      0.125,
      0.09375,
      0.0625,
      0.125,
      0.09375,
      0.0625
    ],
    "components": [
      [
        "ALB",
        "SAI"
      ],
      [
        "ALO",
        "STR"
      ],
      [
        "ANT",
        "RUS"
      ],
      [
        "BEA",
        "OCO"
      ],
      [
        "BOR",
        "HUL"
      ],
      [
        "BOT",
        "PER"
      ],
      [
        "COL",
        "GAS"
      ],
      [
        "HAD",
        "VER"
      ],
      [
        "HAM",
        "LEC"
      ],
      [
        "LAW",
        "LIN"
      ],
      [
        "NOR",
        "PIA"
      ]
    ]
  }
]
```

## Extraction Errors

```json
[]
```
