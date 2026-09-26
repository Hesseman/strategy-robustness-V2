# WFC on the parameter grid: sign-flip null and region lift

The MultiWalk section's WFC card asks whether a strategy's optimisation found something real:
did the parameter combinations that looked best in-sample also do better out-of-sample? Since
2026-09-25 the card's **gate is the region lift**: the combinations in the top 20% of the
pooled in-sample surface are tested against the grid average out-of-sample, under a **block
sign-flip null**. Tinsley's correlation (SSRN 6324079), the card's original statistic, is still
computed and shown for continuity, but it no longer decides.

This file is the single source of truth for the method: the definitions, the decisions and why
they were taken, the evidence, and the known gaps. Code docstrings point here.

## Why not Tinsley's correlation alone

- **Dilution.** A rank correlation over every combination weights a grid's flat majority as much
  as the ridge the trader actually trades. A ridge that persists but covers a fifth of a flat
  grid barely moves it.
- **Attenuation.** Each combination's out-of-sample number is noisy. With few trades per
  combination, a perfectly persistent surface still reads a low correlation.
- **Common mode.** The paper's second rule (most positive-in-sample combinations stay positive
  out-of-sample) passes whenever a good out-of-sample year lifts every combination at once. It is
  retired.
- **No null.** The paper compares the correlation with eyeballed values. V2 first used torus
  shifts of the out-of-sample surface on the grid. Neighbouring combinations share most of their
  trades, and that null passed 10-30% of grids with nothing planted at a nominal 5%.

## The null: block sign-flip (`robustness/null_signflip.py`)

- **H0:** no combination is expected to differ from the grid average out-of-sample.
- **Split each day.** Each out-of-sample day's vector of combination P&L is split into the grid
  mean (the strategy's common path, kept as is) and each combination's deviation from it.
- **Flip each block.** Every block of 21 trading days gets one random sign on its deviations. The
  metric is then recomputed on the rebuilt daily matrix.
- **What a draw keeps:** every combination's noise covariance, the grid edges, and identical
  variants.
- **When it is exact:** when each block's deviation vector is as likely as its negation (symmetric
  block noise) and blocks are independent.
- **p-value:** p = (k + 1) / (n + 1), where k counts the draws at least as large as the observed
  statistic.
- **Torus null:** `wfc_test(..., null='torus')` keeps it for comparison only.

## The region lift (`robustness/region_wfc.py`)

**Per window, on net profit.** Net profit is linear, so an ensemble's result is the mean of its
members'. The fixed knobs are in 'Decisions'.

1. `x` = in-sample net profit per combination; `y` = out-of-sample net profit. There is no
   per-combination minimum trade count: the null already prices each combination's noise.
2. Pool the in-sample surface: each combination is replaced by the equal-weight mean of itself
   and its Chebyshev-1 neighbours (every combination one step away on any axis; the grid does
   not wrap).
3. Region R = the top `round(0.2 x n)` combinations of the pooled surface. Ties keep grid order.
4. Lift L = (mean of `y` over R - mean of `y`) / cross-combination SD of `y` (ddof 0), on the raw
   out-of-sample surface.
5. Null: `SignFlipNull(grid, "NP", 21).draws`. The generator is seeded `seed + 1000 x window
   index`. A draw's lift is scaled by the *observed* SD, a fixed unit per window.
6. Pool over the complete windows: the pooled statistic is the mean L. The pooled null averages
   the per-window draws draw by draw; the windows' draws are independent.

**Display only:**
- **Overlap precision:** the share of R inside the top 20% of the pooled out-of-sample surface
  (chance is 0.20), with its own p.
- **Ridge:** the largest Chebyshev-connected set of combinations at or above a pooled surface's
  80th percentile. Shown in and out of sample as a Jaccard overlap and a centroid shift in grid
  steps.
- **Picks:** where four pick rules landed out of sample (best in-sample combination, best pooled
  in-sample combination, the region ensemble, the window's pick).
- Precision and Jaccard are not scored (NaN) when a window has fewer than 2|R| distinct
  out-of-sample daily patterns.

**Deliberately absent:** a correlation of the two pooled surfaces. Pooling shrinks the effective
number of combinations without removing the noise neighbours share. On correlated synthetic
noise it was weaker than the pointwise correlation.

## The verdict (`robustness/multiwalk_battery.py`)

Tinsley's diagnostic matrix, with the pooled lift as the row test (alpha 0.05):

| | region profitable OOS | region not profitable OOS |
|---|---|---|
| **lift significant** | `edge` → **PASS**: structural edge, localised in R | `loser` → **FAIL**: consistent loser |

| | grid average profitable OOS | grid average not profitable OOS |
|---|---|---|
| **lift not significant** | `plateau`: parameter choice immaterial, the strategy-level cards govern (gate not applied) | `noise` → **FAIL** |

- **Structure gate:** when the median number of distinct out-of-sample daily patterns over the
  complete windows is below 2|R|, the grid is not scored and reads `plateau`. The out-of-sample
  results repeat (identical variants), so any top-set statistic would only measure clusters lining
  up with themselves. This is the only structure gate.
- **Insufficient:** no complete window.
- **Continuity:** Tinsley's Spearman ρ on the project's fitness metric (combinations below
  `min_trades` dropped, same sign-flip null) is still shown on the card, never as the gate.
  `WFCResult.passed` keeps the paper's rule for comparison only.

### Printed guards

Every PASS is printed with the facts it has to be read with. They live in `meta["wfc_guards"]` in
the JSON, on the card's **Read with the PASS** line, and in the window table's `region − grid $`
column:

- **N_eff:** the effective number of independent variants, from the selection card. Below 3
  (`few_variants`), the variants are nearly one strategy and a significant lift rests on the
  handful of trades where they differ. This is context, not a gate (see 'Decisions').
- **Dollars:** the region's and the grid's mean out-of-sample net profit.
- **Per window:** the region's margin over the grid in each complete window (`windows_ahead` of
  `windows_complete`). A pooled pass carried by one window reads differently from one that holds
  in every window.

## Decisions

All dated 2026-09-25, owner's call:

1. **Null: torus shift → block sign-flip.** Under correlated neighbour noise the torus null
   passed 10-30% of no-structure grids; the sign-flip passed about 5%.
2. **Gate: correlation → region lift.** The lift answers the trader's question (does the
   in-sample top region beat the grid out of sample?) and measured better power where it matters.
   See 'Evidence'.
3. **N_eff is context, not a gate.**
   - The method draft gated on N_eff < 3.
   - On 24 real grid × window-scheme runs from the author's MultiWalk archive (the data is not
     in this repository), N_eff sat at 1.0-1.7 in 22 runs and 2.9-3.1 in the other two. N_eff
     on raw daily P&L mostly counts the path every variant shares, which the lift subtracts and
     the sign-flip null keeps exactly.
   - The gate would have left one run scoreable and hidden four significant lifts whose regions
     beat the grid out of sample.
   - The threshold also flipped verdicts for one grid between window schemes (N_eff 2.9 vs 3.1).
   - Lowering the threshold after seeing those runs was rejected as fitting the rule to the
     result.
   - Re-measuring N_eff on the deviations was deferred: it would need its own synthetic
     calibration.
4. **Fixed a priori, never tuned per project:** region share q = 0.2, pooling radius r = 1,
   sign-flip block = 21 trading days, alpha = 0.05. Changing one is a method change: re-run the
   synthetic and real-data oracles and add a decision here.

## Evidence

**Synthetic, planted truth.** Built with `synthetic_multiwalk.make_planted_grid`: 750 trading
days split at day 500, and neighbouring combinations sharing their noise at correlation rho_s.

- **Size under the sign-flip null** (false-pass rate at 5%, mean over 24 no-structure
  configurations: 8×8 and 6×5×4 grids × rho_s 0 / 0.6 / 0.9 × 8-127 OOS trades per
  combination; 30 seeds and 299 draws each):
  - region lift 6.1%, worst single configuration 5 of 30, which is binomial noise;
  - overlap precision 2.6%;
  - `tests/test_region_wfc.py` holds 200 seeds at ≤ 7%.
- **Power on a narrow ridge** (8×8 grid, rho_s 0.6 and 0.9, pooled):

  | OOS trades per combination | region lift | pointwise ρ |
  |---|---|---|
  | ~127 | 0.93 | 0.43 |
  | ~50 | 0.50 | 0.33 |
  | ~20 | 0.23 | 0.20 |

  Below about 20 out-of-sample trades per combination no statistic has power. The fix is longer
  out-of-sample blocks (the 2-windows scheme), not pooling.
- **Decay:** a surface whose bump flips sign after the split never passes.

**Real.** The opt-in `tests/test_real_multiwalk.py` runs on the author's LE2601 project through
`SR_SAMPLE_MW_DIR` and reproduces the research prototype to the draw. With 499 draws and seed 0:
L +0.322, p 0.156; precision 0.292, p 0.302; ridge Jaccard 0.21, shift 2.9. A changed number
there means the statistics changed.

## Known gaps

- **Symmetry check missing.** The sign-flip assumes symmetric block deviations. With few
  effective variants those deviations are a handful of trades, where skew is most likely. The
  centred block bootstrap cross-check (resample the same deviations instead of flipping them,
  and check that the two agree) is not built. Until it is, trust a few-variants PASS near p = 0.05
  less than one far from it.
- **Unreadable schedules.** Walk-forward databases scheduled in trading days (period type ids
  1/1) are rejected by `walkforward_db.py`.
- **Thin windows.** Two windows are two windows: the method cannot add data.
- **Equal steps.** Chebyshev distance treats one step on every axis as equal. That is the grid
  designer's choice, not a measured distance.

## Where things are

| Concern | Code | Tests |
|---|---|---|
| sign-flip null | `robustness/null_signflip.py` | `tests/test_null_signflip.py` (literal sign patterns) |
| region lift, pooling, ridge, picks | `robustness/region_wfc.py` | `tests/test_region_wfc.py` (hand traces, literal null, size, power, decay) |
| Tinsley's correlation (continuity) | `robustness/wfc_grid.py` | `tests/test_wfc_grid.py` |
| verdict matrix, structure gate, guards | `robustness/multiwalk_battery.py` | `tests/test_multiwalk_battery.py`, `tests/test_multiwalk_multiwindow.py` |
| surface heatmaps, lift null chart | `robustness/charts.py` | `tests/test_charts_mw.py` |
| card and copy | `app/streamlit_app.py`, `app/copy.py` | `tests/test_app_smoke.py` |
| planted-truth grids | `robustness/synthetic_multiwalk.py` | used by the oracles above |
| real-data oracle (opt-in) | none | `tests/test_real_multiwalk.py` |
