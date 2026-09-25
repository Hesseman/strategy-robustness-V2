"""Card copy - text only. Adapted from the ES RSI(2) deck's per-test slides."""
CARDS = {
    "baseline": {
        "title": "Baseline - edge is measured against the drift, not against zero",
        "tagline": "The T8a, T3 and T7 cards below score lift = the strategy's return minus what random timing earns",
        "catches": "A long strategy on a rising market shows positive returns by construction; a win rate proves nothing by itself.",
        "how": "For every trade, the average return a random entry with the same hold length and direction would have earned over the whole bar history. Lift is the strategy's mean minus that. Per-contract % of price, so early and late years weigh the same.",
    },
    "t8a": {
        "title": "T8a - Random entries: is it timing, or just being in the market?",
        "tagline": "Luck - build the distribution of what random timing achieves and see where the strategy lands",
        "catches": "N trades can beat the drift by chance. We need to know how much N random entries vary, and whether the strategy's mean is inside that range.",
        "how": "Draw one random entry bar per trade, hold it for that trade's own number of bars in its own direction, take the mean; repeat 1000 times. p = (random sets at least as good + 1) / (sets + 1) - the strategy's own result counts as one more draw, so p never reads exactly 0: 0 of 1000 shows as p ≤ 0.001. Gate: p < 0.05. It takes the strategy as given - it does not know how many strategies were tried.",
    },
    "t3": {
        "title": "T3 - Does it work in every era, and in a crisis?",
        "tagline": "Instability - an average over ten years can hide an edge that lived in one regime",
        "catches": "An edge concentrated in one bull run, or one that faded after a rule change, still averages positive over the full sample.",
        "how": "Split the bars into four equal windows and recompute lift in each (against that window's own drift); score = windows with positive lift. Crisis split: trades entered on top-decile volatility bars (20-bar rolling) versus the rest. Plus $ per calendar year for one contract.",
    },
    "t7": {
        "title": "T7 - Costs: how much friction would it take to kill it?",
        "tagline": "Friction - an assumption haircut and a stress test, not an execution simulation",
        "catches": "A real but tiny edge is a donation to the broker. One tick of slippage is easy on a quiet day and wrong on a panic day.",
        "how": "Every trade is haircut by the instrument's reference round-trip cost (commission + slippage, both sides, from the MultiWalk symbol lists). Gate: net lift over the drift after 1x cost > 0. Then multiply the cost until the lift disappears - the break-even multiplier is the margin of safety.",
    },
    "drawdown": {
        "title": "Drawdown & capital - what does it take to hold this strategy?",
        "tagline": "Sizing - one contract, dollar P&L, discrete drawdown episodes",
        "catches": "A single worst drawdown is one unstable number; an average drawdown hides the tail. Neither tells you what account size the strategy needs.",
        "how": "Cumulative $ P&L for one contract by trade close -> peak-to-recovery drawdown episodes -> CDaR-80 (mean of the worst 20% of episodes) -> capital = 5 x CDaR-80 -> annual % on that capital. Sharpe, Sortino, Calmar and profit per average drawdown are for ranking strategies, not for sizing. The margin line uses today's margin only and says nothing about the past.",
    },
    "wfc": {
        "title": "WFC - Walk Forward Correlation: did in-sample results predict out-of-sample results?",
        "tagline": "Over-fitting - does the best region of the parameter surface in-sample beat the grid average out-of-sample? (Tinsley 2026, region form)",
        "catches": "A walk-forward that validates only the single best parameter set per window can pass by chance. And a correlation over every combination is diluted by the flat majority of the grid: a ridge that persists but covers a fifth of an otherwise flat grid barely moves it, while a good out-of-sample year lifts every combination at once. If the combinations that looked best in-sample do no better than the grid average out-of-sample, the optimisation was fitting noise - or the settings do not matter.",
        "how": "Gate - the region lift, per walk-forward window, on net profit. Each combination's in-sample result is averaged with its one-step neighbours on the grid (the pooled surface); the top 20% of that pooled surface is the in-sample region. Lift L = (the region's mean out-of-sample net profit - the grid's) / the spread of out-of-sample net profit across the grid, in SDs, on the raw out-of-sample results. The null re-draws the out-of-sample surface by flipping the sign of each 21-trading-day block's deviations from the grid's daily average - the strategy's common path and how alike neighbouring combinations are both stay exactly as they were, only the alignment with in-sample is broken; p = (draws at least as large + 1) / (draws + 1), pooled over the complete windows. Verdict - Tinsley's matrix with the lift as the row test: lift significant (p < 0.05) and the region profitable out-of-sample - PASS, a structural edge localised in the region; significant but the region loses - FAIL, a consistent loser; not significant while the grid as a whole made money out-of-sample - PLATEAU: the parameter choice is immaterial and the strategy-level cards govern (gate not applied); not significant and the grid lost - FAIL, noise. A grid whose combinations are nearly one strategy (fewer than 3 effective independent variants, from the selection card) or whose out-of-sample results repeat (fewer distinct patterns than twice the region) is not scored: PLATEAU. No minimum trade count per combination - the null already prices each one's noise; below about 20 out-of-sample trades per combination nothing has power, and 2 windows (thirds) give longer blocks. Also shown: the overlap (share of the region inside the out-of-sample top 20% of the pooled surface; chance is 20%), the ridge (largest connected top-20% area) in vs out-of-sample with its overlap and how far its centre moved, where four picks landed out-of-sample, and Tinsley's own correlation - Spearman over every combination on the project's fitness, same null - for continuity, never as the gate. Tabs: Surface (heatmaps in and out-of-sample, raw and pooled, each ridge outlined; with 3+ parameters the slice through the best pooled in-sample combination); Scatter (one dot per combination, with the green best-fit line of Tinsley's chart); Ranked profile (combinations sorted by in-sample result, best on the left - the in-sample rank is the falling blue line, the out-of-sample rank of the same combination an orange dot, hollow when it lost money); Bands (grids of 100+ combinations: the mean out-of-sample result per tenth of the in-sample ranking); the two nulls. The top in-sample combinations are outlined in red, and each window's line says where they landed out-of-sample.",
    },
    "plateau": {
        "title": "Plateau - does the chosen parameter set sit on a plateau or on a spike?",
        "tagline": "Fragility - neighbours that also work mean the edge does not depend on one number",
        "catches": "An optimum surrounded by losses is a coincidence of that sample; the next window will land next to it, not on it.",
        "how": "Take the parameter set MultiWalk chose for each window and its neighbours (one step in any direction on the grid). Score = half the flatness of their out-of-sample values (1 - std / mean) plus half the share of neighbours that were profitable out-of-sample. 1 = flat and all positive, 0 = spike. The in-sample plateau is shown for comparison. Pooled over complete windows; a score, not a gate.",
    },
    "selection": {
        "title": "Selection haircut - how good does the best of N look when there is nothing to find?",
        "tagline": "Luck of the draw - the more combinations you try, the better the best one looks",
        "catches": "Picking the best of 240 variants guarantees an impressive in-sample number. The question is whether it is more impressive than the best of 240 variants with no edge at all.",
        "how": "Reality Check (White 2000): resample the in-sample days in blocks (stationary bootstrap, 20-day mean block), recompute every combination's mean daily P&L with the true means removed, and record the best; repeat 500 times. p = share of resamples whose best beats the observed best, and separately MultiWalk's pick. The effective number of independent variants comes from the correlation of the combinations' daily P&L. Reference only.",
    },
    "timing_entry": {
        "title": "Entry delay - does the edge live in the first bars after the signal?",
        "tagline": "Execution sensitivity - the same trades, entered 1, 2, ... bars late, exits as reported",
        "catches": "An edge that collapses within 1-2 bars of delay lives in the signal bar itself: it is exposed to latency and slippage on entry and, on slow bar sizes, is a warning sign for look-ahead in the signal. An edge that barely moves means the entry is a coarse regime filter, not a timing call.",
        "how": "Every trade's entry moves k bars later and fills at the open of that bar; the exit keeps its reported bar and price (the exit rule is treated as a time-fixed signal - we cannot re-run the strategy's stops and targets). A trade whose delayed entry reaches its exit bar is skipped at that k and counted. In 'fixed hold' mode the exit moves k bars too, at the open, so the hold length is kept. Gross $, one contract, no costs: costs move the curve's level, not its shape. Trades are independent - no position limit. Left of zero (shaded) the entry moves k bars earlier instead, filled at that bar's open and skipped if it falls before the first bar: hindsight, since no strategy can enter before its signal - read it as how much of the move the entry signal lags, not as a result you could trade.",
    },
    "timing_exit": {
        "title": "Exit delay - is the exit precisely timed?",
        "tagline": "Exit sensitivity - the same trades, closed 1, 2, ... bars late, entries as reported",
        "catches": "If a later exit improves the return, the exit rule fires early relative to the move. If it destroys the return, the exit is precisely timed (a stop or a target) and slippage on the exit bar matters more than on the entry bar.",
        "how": "Every trade keeps its reported entry; its exit moves k bars later and fills at the open of that bar. A trade whose delayed exit falls past the last bar is skipped at that k and counted. A delayed exit may overlap the next trade's entry - trades are treated as independent. Gross $, one contract, no costs. Identical in both modes. Left of zero (shaded) the exit moves k bars earlier instead, filled at that bar's open and skipped once it would reach the entry bar: hindsight - how much the exit signal lags the turn, not a tradable result.",
    },
}

CONCEPT = """**What the cards measure.** The baseline, T8a, T3 and T7 cards score *lift*: the strategy's return minus what random entries with the same hold lengths and directions would have earned on the same bars. A long strategy in a rising market is positive by construction; lift removes that drift.

**How verdicts work.** Gates decide (PASS / FAIL): T8a random entries (p < 0.05) and T7 costs (net lift after 1x cost > 0). Scores rank (T3: windows with positive lift). Reference cards inform (baseline, drawdown, and the two timing-sensitivity cards). There is no composite number on purpose: one failed gate is a failed strategy, however good the rest looks.

**How capital is set.** Cumulative $ P&L for one contract -> drawdown episodes (peak -> trough -> recovery) -> **CDaR-80** = the mean depth of the worst 20% of episodes -> **capital = 5 × CDaR-80** -> annual return = yearly $ / that capital. In the sidebar you can replace the capital with a fixed starting amount; CDaR-80 and 5 × CDaR-80 are still shown.

**Timing sensitivity** (under the five cards) shifts every entry, or every exit, by 1, 2, ... bars and shows how much of the gross $ survives - and, as hindsight, what acting earlier would have given. It is a picture, not a gate, and is not counted in "Gates passed".

**Optional: MultiWalk surface tests** (bottom of the page) read every parameter combination of a MultiWalk optimisation and ask whether the region of the grid that was best in-sample beat the grid average out-of-sample (WFC), whether the chosen parameters sit on a plateau, and how much of the best in-sample result is selection luck."""

HELP = {
    "report": "TradeStation: open the chart that runs the strategy, then View -> Strategy Performance Report. In the report window choose File -> Save As and the file type **CSV (comma delimited)**; keep every section (the app reads the Trades List and the Settings). Excel files are not read.",
    "bars": "Same chart, same symbol and interval, covering the whole traded range: View -> Data Window, then save the window as a comma-separated text file (.txt or .csv). Indicator (PLOT) columns are ignored. The report and the bars must come from the same chart so their timestamps agree.",
    "n_perm": "How many random-entry sets T8a draws. More sets = a finer p-value floor (1 / (sets + 1)) and a longer run. 1000 is enough for the 0.05 gate.",
    "seed": "Starting point of the random number generator. Same seed = same random draws = same p-values. Change it to check that a verdict does not hinge on one draw; it should not.",
    "margin": "The exchange's current margin per contract. Only the 'if deployed today' line uses it; it says nothing about the past.",
    "capital": "5 × CDaR-80 sizes the account from the strategy's own drawdown history. A fixed amount answers 'what if I trade this with $X': the annual % and the max-drawdown % of capital then use your number; CDaR-80 is still shown for reference.",
    "capital_usd": "The account size used for the annual % and the max-drawdown % of capital, and for the margin line. CDaR-80 and 5 × CDaR-80 are still computed and shown.",
    "multiwalk": """**Needs MultiWalk Pro.** MultiWalk normally stores the optimisation results in a sealed `.dat` file. Switch the project to the readable text format once:

1. Open `MultiWalkSetup.txt` in the project folder (next to `Optimization Files` and `Walkforward Files`) in Notepad and change `iUseLegacyOptimizationTextFileFormat: false` to `iUseLegacyOptimizationTextFileFormat: true`. Save.
2. In MultiWalk: **Settings -> Project Files/Folders -> Import Project Setup**, select that file, then **Save Project Setup** on the same screen. Editing the file alone is not enough - MultiWalk rewrites it from memory.
3. **Operations -> Run Project.** The `Optimization Files` folder now holds `..._MultiWalk.txt` instead of the `.dat` (about 40 MB for 240 combinations over ten years).

Upload two files: the `..._MultiWalk.txt` from `Optimization Files` and `WalkforwardData.db` from `Walkforward Files`. Nothing else is needed - these tests use no bars and no report.""",
    "mw_files": "The text file holds every parameter combination's daily P&L and trades; the database holds the walk-forward windows and which combination MultiWalk picked for each. Both come from the same project folder and the same run.",
    "n_boot": "Resamples for the selection haircut. 500 gives a p-value floor of 0.002; 200 is enough for a first look.",
    "mw_windows": "Which in-sample / out-of-sample windows the three cards use. MultiWalk's windows = this project's walk-forward schedule and the combination MultiWalk picked in each. With few trades per window (a daily strategy on 1-year windows has about 15), use 2 windows (the history in three equal parts: first → second, second → third) or 1 split (first half in-sample, second half out-of-sample); the pick is then the best in-sample combination. Choose before you look at the result - picking the split that passes is one more way to fit the data.",
    "timing": "Same two files as the cards above. How much of the gross $ survives when every entry, or every exit, is acted on 1, 2, ... bars late - and, left of zero, 1, 2, ... bars early (hindsight: no strategy can act before its signal). A picture of fragility, not a gate: no verdict and not counted in 'Gates passed'.",
    "timing_max_k": "How many bars each leg is shifted, both ways: 1..k bars later (a delay you could suffer live) and 1..k bars earlier (hindsight). A shifted leg fills at the open of the bar it moves to.",
    "timing_mode": "Fixed exit (default): a delayed entry keeps the reported exit bar and price, so the hold gets shorter and a trade whose delayed entry reaches its exit bar is skipped. Fixed hold: the exit moves the same number of bars, so the hold length is kept and the entry card shifts the whole trade. The exit-delay card and the 'Where timing matters' lines (one leg at a time) are the same in both modes.",
}
