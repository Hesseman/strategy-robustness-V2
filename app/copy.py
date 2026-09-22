"""Card copy - text only. Adapted from the ES RSI(2) deck's per-test slides."""
CARDS = {
    "baseline": {
        "title": "Baseline - edge is measured against the drift, not against zero",
        "tagline": "Every card below scores lift = the strategy's return minus what random timing earns",
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
        "tagline": "Over-fitting - the whole parameter surface, not one lucky parameter set (Tinsley 2026)",
        "catches": "A walk-forward that validates only the single best parameter set per window can pass by chance. If the ranking of all combinations in-sample says nothing about their ranking out-of-sample, the optimisation was fitting noise.",
        "how": "For every parameter combination MultiWalk optimised, take its in-sample metric (the project's fitness, NP / average drawdown) and its out-of-sample metric in each walk-forward window and correlate the two across the grid (Spearman; Pearson shown). The null shifts the out-of-sample surface across the grid, keeping its smoothness but breaking its alignment with in-sample; p = (shifts at least as correlated + 1) / (shifts + 1). Gate: pooled p < 0.05 AND at least half of the combinations that were positive in-sample are positive out-of-sample - correlation alone is not edge.",
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
}

CONCEPT = """**What the cards measure.** Every timing card scores *lift*: the strategy's return minus what random entries with the same hold lengths and directions would have earned on the same bars. A long strategy in a rising market is positive by construction; lift removes that drift.

**How verdicts work.** Gates decide (PASS / FAIL): T8a random entries (p < 0.05) and T7 costs (net lift after 1x cost > 0). Scores rank (T3: windows with positive lift). Reference cards inform (baseline, drawdown). There is no composite number on purpose: one failed gate is a failed strategy, however good the rest looks.

**How capital is set.** Cumulative $ P&L for one contract -> drawdown episodes (peak -> trough -> recovery) -> **CDaR-80** = the mean depth of the worst 20% of episodes -> **capital = 5 × CDaR-80** -> annual return = yearly $ / that capital. In the sidebar you can replace the capital with a fixed starting amount; CDaR-80 and 5 × CDaR-80 are still shown.

**Optional: MultiWalk surface tests** (bottom of the page) read every parameter combination of a MultiWalk optimisation and ask whether in-sample results predicted out-of-sample results across the whole grid (WFC), whether the chosen parameters sit on a plateau, and how much of the best in-sample result is selection luck."""

HELP = {
    "report": "TradeStation: open the chart that runs the strategy, then View -> Strategy Performance Report. In the report window choose File -> Save As and the file type **CSV (comma delimited)**; keep every section (the app reads the Trades List and the Settings). Excel files are not read.",
    "bars": "Same chart, same symbol and interval, covering the whole traded range: View -> Data Window, then save the window as a comma-separated text file (.txt or .csv). Indicator (PLOT) columns are ignored. The report and the bars must come from the same chart so their timestamps agree.",
    "n_perm": "How many random-entry sets T8a draws. More sets = a finer p-value floor (1 / (sets + 1)) and a longer run. 1000 is enough for the 0.05 gate.",
    "seed": "Starting point of the random number generator. Same seed = same random draws = same p-values. Change it to check that a verdict does not hinge on one draw; it should not.",
    "margin": "The exchange's current margin per contract. Only the 'if deployed today' line uses it; it says nothing about the past.",
    "capital": "5 × CDaR-80 sizes the account from the strategy's own drawdown history. A fixed amount answers 'what if I trade this with $X': the annual % and the max-drawdown % of capital then use your number; CDaR-80 is still shown for reference.",
    "multiwalk": """**Needs MultiWalk Pro.** MultiWalk normally stores the optimisation results in a sealed `.dat` file. Switch the project to the readable text format once:

1. Open `MultiWalkSetup.txt` in the project folder (next to `Optimization Files` and `Walkforward Files`) in Notepad and change `iUseLegacyOptimizationTextFileFormat: false` to `iUseLegacyOptimizationTextFileFormat: true`. Save.
2. In MultiWalk: **Settings -> Project Files/Folders -> Import Project Setup**, select that file, then **Save Project Setup** on the same screen. Editing the file alone is not enough - MultiWalk rewrites it from memory.
3. **Operations -> Run Project.** The `Optimization Files` folder now holds `..._MultiWalk.txt` instead of the `.dat` (about 40 MB for 240 combinations over ten years).

Upload two files: the `..._MultiWalk.txt` from `Optimization Files` and `WalkforwardData.db` from `Walkforward Files`. Nothing else is needed - these tests use no bars and no report.""",
    "mw_files": "The text file holds every parameter combination's daily P&L and trades; the database holds the walk-forward windows and which combination MultiWalk picked for each. Both come from the same project folder and the same run.",
    "n_boot": "Resamples for the selection haircut. 500 gives a p-value floor of 0.002; 200 is enough for a first look.",
}
