# WFC on the parameter grid: sign-flip null and region lift

The MultiWalk section's WFC card asks whether a strategy's optimisation found something real:
did the parameter combinations that looked best in-sample also do better out-of-sample? Since
2026-09-25 the card's **gate is the region lift**: the combinations in the top 20% of the
pooled in-sample surface are tested against the grid average out-of-sample, under a **block
sign-flip null**. Every PASS also prints the lift's p under a **centred block bootstrap**, the
flip's cross-check, and flags a disagreement. Tinsley's correlation (SSRN 6324079), the card's
original statistic, is still computed and shown for continuity, but it no longer decides.

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

### The cross-check: centred block bootstrap (`BlockBootstrapNull`, same module)

- **Why.** The flip assumes each block's deviations are as likely as their negation. With few
  effective variants those deviations are a handful of divergent trades, where skew lives. The
  bootstrap keeps the skew instead of flipping it away.
- **Same split, same blocks.** The common path is kept. Each combination's deviations are
  centred per day: its mean daily deviation, (its OOS net profit - the grid's) / OOS days, is taken
  off every day, so under the bootstrap every combination's expected draw is the grid average
  (H0).
- **Draw.** Each of the window's 21-day block positions takes one of the window's centred blocks,
  picked uniformly with replacement: a non-overlapping block bootstrap on the flip's own blocks.
- **Net profit only.** The window's last block can be short, so a resampled block may not fit the
  position it fills. A path metric (NP/AvgDD) would have no rebuilt path to read.
- **Its own bias.** The plug-in variance of B resampled blocks is (B - 1) / B of the flip's under
  H0, so the bootstrap is slightly liberal on short windows: 7.6% against the flip's 6.1% at 12
  blocks per window, 3.5% against 3.2% at 36 (see 'Evidence').
- **Its own generator stream.** It is seeded (seed + 1000 x window index, 1), so the flip's draws
  are unchanged and results published before the bootstrap existed reproduce to the draw.
- **Pooled and p** exactly as the flip: the per-window draws are averaged draw by draw over the
  complete windows, and p = (k + 1) / (n + 1).
- **Disagreement** is fixed a priori: one p < 0.05 and the other ≥ 0.05, with a NaN p counting as
  not significant (`multiwalk_battery.nulls_disagree`). The flip stays the gate. A disagreement is
  printed beside a PASS and never changes the verdict.
- **Monte Carlo noise.** At 999 draws a p near 0.05 has a standard error of about 0.007, and the
  two nulls draw from different streams. A disagreement inside that band can be noise, not skew.

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
   index`. A draw's lift is scaled by the *observed* SD, a fixed unit per window. The
   cross-check `BlockBootstrapNull(grid, "NP", 21)` draws as many from its own stream and is
   scored the same way (`p_lift_boot`). Its p is reported, never gated.
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
the JSON, on the card's **Read with the PASS** line (`app/wfc_text.py`), and in the window
table's `region − grid $` column:

- **Both nulls:** the lift's p under the sign-flip (the gate) and under the centred block
  bootstrap (`p_lift_boot`). When one is below 0.05 and the other is not (`nulls_disagree`), the
  line says **the two nulls disagree**: the PASS rests on the flip's symmetry assumption.
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
5. **Symmetry cross-check: a centred block bootstrap beside the flip; the flip stays the gate.**
   - It closes the gap the flip left open: the flip is exact only for symmetric block
     deviations, and the passes that rest on few effective variants are where skew is likely.
   - Construction: the non-overlapping bootstrap on the flip's own 21-day blocks, centred per
     combination per day (see 'The null'). A circular block bootstrap was the alternative: exact
     window length, usable for path metrics, same variance bias. It was not chosen because the
     twin should differ from the flip in one thing only: resampling instead of flipping.
   - Disagreement is fixed before any result was seen: one p < 0.05, the other ≥ 0.05. It is
     printed with the PASS, never gated. Changing the verdict rule on a disagreement is the
     owner's call.
   - Evidence: size on symmetric noise 7.6% against the flip's 6.1% at 12 blocks per window, and
     3.5% against 3.2% at 36. On rare large winners the flip reads 2.2% and the bootstrap 3.5%; on
     rare large losers 4.4% and 5.1%. On the 24 real runs the two nulls fall on the same side of
     0.05 every time, including all five passes (see 'Evidence').

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
- **Size under both nulls** (false-pass rate of the lift at 5%, mean over the same 24
  configurations, 30 seeds and 299 draws each; worst single configuration in brackets). "Rare
  large winners" is `make_planted_grid(tail="jumps")`: 3% of trades are large winners, the rest
  small losers, mean 0, skew about 5.5. "Rare large losers" is the same grids negated.

  | noise | 21-day blocks per window | sign-flip | bootstrap | disagree |
  |---|---|---|---|---|
  | symmetric (Gaussian) | 12 | 6.1% (5/30) | 7.6% (5/30) | 2.4% |
  | symmetric, 750 OOS days | 36 | 3.2% (3/30) | 3.5% (3/30) | 0.8% |
  | rare large winners | 12 | 2.2% (2/30) | 3.5% (3/30) | 1.5% |
  | rare large losers | 12 | 4.4% (3/30) | 5.1% (4/30) | 1.5% |

  - The bootstrap's excess on short windows shrinks with more blocks, as its (B - 1) / B
    variance predicts.
  - Skew in either direction did not inflate the flip's size here.
  - `tests/test_region_wfc.py` holds the bootstrap at ≤ 10% on symmetric noise and both nulls at
    ≤ 10% on rare large winners (200 seeds each).

**Real.** The opt-in `tests/test_real_multiwalk.py` runs on the author's LE2601 project through
`SR_SAMPLE_MW_DIR` and reproduces the research prototype to the draw. With 499 draws and seed 0:
L +0.322, p 0.156; precision 0.292, p 0.302; ridge Jaccard 0.21, shift 2.9. A changed number
there means the statistics changed. The bootstrap reads p 0.126 there (0.130 on the 'two'
scheme): insignificant under both nulls.

On the 24 real runs of 'Decisions' (499 draws, seed 0), the bootstrap's p falls on the same side
of 0.05 as the flip's in every run:
- **The five passes:** flip 0.002, 0.002, 0.024, 0.028 and 0.032; bootstrap 0.002, 0.002, 0.018,
  0.026 and 0.032.
- **Closest call:** a plateau at flip 0.062, bootstrap 0.050.
- **The flip's own p-values** reproduce the earlier table to the draw.

## Known gaps

- **Unreadable schedules.** Walk-forward databases scheduled in trading days (period type ids
  1/1) are rejected by `walkforward_db.py`.
- **Thin windows.** Two windows are two windows: the method cannot add data.
- **Equal steps.** Chebyshev distance treats one step on every axis as equal. That is the grid
  designer's choice, not a measured distance.

## Where things are

| Concern | Code | Tests |
|---|---|---|
| sign-flip null | `robustness/null_signflip.py` | `tests/test_null_signflip.py` (literal sign patterns) |
| centred block bootstrap (cross-check) | `robustness/null_signflip.py` | `tests/test_null_signflip.py` (literal block picks), `tests/test_region_wfc.py` (size, symmetric and rare large trades) |
| disagreement rule | `robustness/multiwalk_battery.py` (`nulls_disagree`) | `tests/test_multiwalk_battery.py` |
| the Read with the PASS line | `app/wfc_text.py` | `tests/test_wfc_text.py`, `tests/test_app_smoke.py` |
| region lift, pooling, ridge, picks | `robustness/region_wfc.py` | `tests/test_region_wfc.py` (hand traces, literal null, size, power, decay) |
| Tinsley's correlation (continuity) | `robustness/wfc_grid.py` | `tests/test_wfc_grid.py` |
| verdict matrix, structure gate, guards | `robustness/multiwalk_battery.py` | `tests/test_multiwalk_battery.py`, `tests/test_multiwalk_multiwindow.py` |
| surface heatmaps, lift null chart | `robustness/charts.py` | `tests/test_charts_mw.py` |
| card and copy | `app/streamlit_app.py`, `app/copy.py` | `tests/test_app_smoke.py` |
| planted-truth grids (Gaussian or `tail="jumps"` noise) | `robustness/synthetic_multiwalk.py` | `tests/test_synthetic_multiwalk.py`; used by the oracles above |
| real-data oracle (opt-in) | none | `tests/test_real_multiwalk.py` |
