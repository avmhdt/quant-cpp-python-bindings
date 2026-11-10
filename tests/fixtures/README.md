# Synthetic fixtures

Seeded, synthetic series used by the tests. Never real or licensed data (DQ-09, DQ-10).

Regenerate with `uv run python -m tools.qkdev fixtures make`; CI runs `--check`, which fails
unless every file is byte-identical to its generator's output. Definitions and seeds are in
`tools/qkdev/fixtures.py`.

| File | Contents |
|---|---|
| `series_gauss_10k.npz` | N(0, 1), n = 10,000 |
| `series_offset_1e9_10k.npz` | 1e9 + N(0, 1): catastrophic cancellation for naive variance |
| `series_nan_runs_10k.npz` | N(0, 1) with 5% isolated NaN plus 8 NaN runs of length 1–300 |
| `series_inf_spikes_5k.npz` | N(0, 1) with 10 `+inf` and 10 `-inf` spikes and one adjacent pair |
| `series_constant_segments_5k.npz` | Constant runs (length 1–500) mixed with noise segments |
| `series_student_t3_10k.npz` | Student-t, 3 degrees of freedom |
| `series_pairs_corr_10k.npz` | `x`, `y = beta x + 0.3 noise`, beta 0.8 then 1.5 from index 6,000 |

Schema: `x` (float64), optional `y` (float64), and `meta`, a 0-d string array holding JSON
`{"generator", "seed", "params", "qkern_fixture_version"}`. The BBO and book_l2 fixtures
arrive with the order-flow kernels (FR-07).
