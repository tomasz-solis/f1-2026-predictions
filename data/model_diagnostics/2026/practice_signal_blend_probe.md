# Practice Signal Blend Probe - 2026

- Transform: `rerank by w * checkpoint_rank + (1 - w) * PRE_rank`
- `w=1.00` reproduces the stored practice-informed checkpoint forecast; `w=0.00` reproduces the pre-weekend (PRE) forecast.
- Event/checkpoint pairs scored: **25**

## Pooled qualifying MAE by checkpoint and blend weight

| Checkpoint | w=0.00 | w=0.25 | w=0.50 | w=0.75 | w=1.00 | Best |
|---|---:|---:|---:|---:|---:|---:|
| FP1 | 2.818182 | 2.712121 | 2.772727 | 2.901515 | 3.204546 | `0.25` |
| FP2 | 2.890909 | 2.818182 | 2.909091 | 3.327273 | 3.618182 | `0.25` |
| FP3 | 2.886363 | 2.75 | 3.0 | 3.045454 | 3.227273 | `0.25` |
| SQ | 2.863636 | 2.772727 | 2.590909 | 2.75 | 2.863636 | `0.50` |

Notes: blending happens in output (rank) space, so it bounds — but is not identical to — the `stored_checkpoint_blend_weight_*` strength-space knobs. A best weight of 0.00 means the practice-informed reordering adds no value over PRE at that checkpoint; small best weights argue for reducing the stored-checkpoint blend caps.
