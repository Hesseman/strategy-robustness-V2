# Timing-sensitivity test: delayed entries, delayed exits

Status: plan, 2026-09-23. Build target: the `claude/strategy-timing-sensitivity-docker-fdsl60` branch.

## 1. What we are building

A second small Streamlit app in this repo (same Docker image, second compose service on port
8502) that takes the same two TradeStation exports the robustness app takes and answers one
question: **how sensitive is the strategy's return to the timing of its trades?**

Two experiments, each a curve with delay in bars on the x-axis (0..10) and return on the
y-axis:

1. **Delayed entry.** Keep every trade's exit as reported; move the entry k bars later.
2. **Delayed exit.** Keep every trade's entry as reported; move the exit k bars later.

k = 0 is the strategy exactly as reported (actual fill prices). This is a parameter-robustness
style test on the one parameter every strategy has: when it acts. It is an illustration of
fragility, not a gate. No verdict pill, no p-value.

Reads on the curve (goes into the card copy):

- Entry delay collapses the return within 1-2 bars: the edge lives in the first bars after the
  signal. It is execution-sensitive (latency, slippage on the signal bar) and, on slow bar
  sizes, a warning sign for look-ahead in the signal.
- Entry delay barely matters: the entry is a coarse regime filter, not a timing call.
- Exit delay improves the return: the exit rule fires early relative to the move.
- Exit delay destroys the return: the exit is precisely timed (stop or target) and slippage on
  the exit bar matters more than on the entry bar.

## 2. Conventions (decisions, stated so the build never has to guess)

Everything reuses the existing parse -> load -> join pipeline (`report_parser.parse_report`,
`bars_loader.load_bars`, `join.join_trades_to_bars`). The joined trade table gives
`entry_idx`, `exit_idx`, `direction`, `entry_price`, `exit_price` and the join's inferred
`point_value`. Nothing in the existing modules changes.

| Item | Decision |
|---|---|
| Fill price of a moved leg | The **open** of the bar it is moved to. TradeStation "next bar at market" semantics. |
| k = 0 | Actual fills from the report for both legs (`pct_returns`, `usd_per_contract`). |
| Delayed entry, exit leg | Unchanged: original exit bar, actual exit price. The exit rule is treated as a time-fixed signal because we cannot re-run the strategy's stop/target logic. Say this in the copy. |
| Delayed entry, trade no longer fits | If `entry_idx + k >= exit_idx` the trade is **skipped** at that k (you would not enter after the exit signal). Skipped trades are excluded from the metrics and counted; `n_alive` is shown per k. |
| Delayed exit, past the end of the bars | If `exit_idx + k > n_bars - 1` the trade is skipped at that k and counted. |
| Mode toggle (cheap, resolves the main ambiguity) | `mode = "fixed_exit"` (default, above) or `"fixed_hold"`: entry moves to `entry_idx + k`, exit moves to `exit_idx + k`, both at the open, so the hold length is preserved. In fixed_hold the exit-delay experiment is unchanged. |
| Overlap | Trades are independent. A delayed exit may overlap the next entry; we do not model position limits. Say this in the copy. |
| Costs | Gross, one contract, point value from the join. No cost haircut (costs are constant per trade; they do not change the shape of the curve, only its level). Say this. |
| Delays | Default 0..10 inclusive; sidebar slider for the max (1..20). Integers only. |
| Randomness | None. Deterministic. |

Metrics per k (both experiments): `n_alive`, `n_skipped`, `total_usd` (sum of one-contract
gross $ over alive trades), `mean_usd`, `mean_pct` (mean of direction x (exit/entry - 1)),
`win_rate`, `profit_factor` (inf when no losses; exported as null), `retention` =
`total_usd[k] / total_usd[0]` (NaN when `total_usd[0]` is 0 or negative; then the copy shows
the $ difference instead). Also `first_nonpositive_k`: the smallest k >= 1 with
`total_usd <= 0`, or None.

Reference for interpretation: the join's `prices_in_bar_range` check already reports the share
of entries filled at the bar open. Show it. If it is well below 100 % the strategy uses
stop/limit fills and the 0 -> 1 step contains a fill-price effect, not only a timing effect.

## 3. Files

New:

- `robustness/timing.py` — pure computation. Public API:
  - `@dataclass DelayPoint(k, n_alive, n_skipped, total_usd, mean_usd, mean_pct, win_rate, profit_factor, retention)`
  - `@dataclass DelayCurve(kind: "entry" | "exit", mode, points: list[DelayPoint], first_nonpositive_k: int | None, per_trade_pct: np.ndarray (n_trades x n_k, NaN where skipped))`
  - `@dataclass TimingResult(meta: dict, checks: list[Check], entry: DelayCurve, exit: DelayCurve, at_open_share: float, caveat: str)`
  - `delayed_returns(trades, open_, k, leg: "entry" | "exit", mode) -> (pct: np.ndarray, usd_points: np.ndarray, alive: np.ndarray)` — the one function that moves legs; everything else aggregates it.
  - `delay_curve(trades, open_, point_value, leg, mode, max_k) -> DelayCurve`
  - `run_timing(report, bars, *, max_k=10, mode="fixed_exit") -> TimingResult` — joins (raises `battery.ValidationFailed` on a failed error-severity check, same contract as `run_battery`), computes both curves, fills meta like the battery does.
  - `to_json(result) -> str` — reuse `battery._jsonable`.
  - Module docstring: convention table above in prose. Docstrings on public functions in the repo's Accepts/Returns/Guarantees style.
- `app/timing_app.py` — the Streamlit entry. Same sidebar pattern as `app/streamlit_app.py`: two uploaders, demo button, dev sample button via `SR_SAMPLE_REPORT` / `SR_SAMPLE_BARS`, `Clear loaded data`. Controls: max delay slider (default 10), mode radio. Header strip (symbol, interval, trades, point value, bars). Validation-check expander. Two cards (entry, exit): copy left, chart right, result lines like `delaying entry 1 bar keeps 63 % of the gross $ (…)`, `profit turns non-positive at k = 4` or `stays positive through k = 10`, `n alive at k = 10: 401 of 418`. Escape `$` in every `st.markdown` line exactly as the existing app does (`replace("$", chr(92) + "$")`). JSON download button. Copy in `app/copy.py` under new keys `timing_entry`, `timing_exit`.
- `tests/test_timing.py`, `tests/test_timing_app_smoke.py` (see section 5).
- `docs/plans/2026-09-23-timing-sensitivity.md` (this file).

Changed:

- `robustness/charts.py` — add `fig_delay_curve(curve, point_value_label)`: markers+lines of `total_usd` vs k, k = 0 point in ACCENT and the rest in MUTED, zero line dashed RED, secondary trace of `mean_pct` on a right y-axis, hover shows n alive. Keep the module's `_LAYOUT`. One figure per curve.
- `docker-compose.yml` — add service `timing`: same `build`/`image`, `container_name: strategy-timing`, `ports: "127.0.0.1:8502:8501"`, `command: ["streamlit", "run", "app/timing_app.py", "--server.address=0.0.0.0", "--server.port=8501"]`. `Dockerfile` needs no change (`COPY app ./app` already ships the new entry).
- `README.md` — a short section "Timing sensitivity app (port 8502)": what it shows, the conventions table in two sentences, `docker compose up --build` starts both, `docker compose up timing` starts only this one. Add the timing card row to nothing (it is not part of the battery table).

## 4. Implementation order

0. **Pre-existing, Linux-only failures (fix first, own commit).** `tests/fixtures/mini_report.csv`
   is committed with LF endings, but four tests in `tests/test_report_parser.py` (lines 69,
   83, 94, 103) do `mini_report_text.split("\r\n")`, so they pass only on a Windows checkout
   with `core.autocrlf`. Make them line-ending agnostic (`re.split(r"\r?\n", ...)` or
   `.splitlines()`); the joins back with `"\r\n"` can stay because `parse_report` normalises
   both. Do not touch the fixture or the parser. Suite must be 72 passed, 2 skipped (real
   sample) before any new code.
1. `robustness/timing.py` with `delayed_returns` first, then the aggregation, then `run_timing`. Vectorised NumPy over trades, a Python loop over k (11 iterations, trivial).
2. Tests in `tests/test_timing.py` against hand-built bars (section 5), run green.
3. `charts.fig_delay_curve`, `app/copy.py` entries, `app/timing_app.py`.
4. `tests/test_timing_app_smoke.py`, run green.
5. `docker-compose.yml`, README.
6. Full suite `python -m pytest -q` green, including the existing 100+ tests unchanged.
7. If Docker is available in the build environment: `docker compose build` and
   `docker run --rm strategy-robustness python -m pytest -q`. If it is not, say so in the final
   report; the user runs it locally.

Dev setup in the cloud build container (verified 2026-09-23): the pins need Python 3.12+
(`numpy==2.5.3`), the default `python3` there is 3.11, and `uv` plus `/usr/bin/python3.13`
exist. Use:

    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
    .venv/bin/python -m pytest -q

The Docker CLI and compose plugin are installed in that container but the daemon is not
running, so `docker compose config` is the only Docker check possible there; the image build
and in-container test run happen on the user's machine.

## 5. Tests and verifications

### 5.1 Unit tests, `tests/test_timing.py`

Build bars directly as a DataFrame (index `ts` 30-min, `open = 100 + i` for bar i, high/low
around it, close = open + 0.5) so every expected value is exact.

1. **k = 0 equals the report.** On synthetic trades (`conftest.make_bars` + `make_trades`),
   `delay_curve(..., leg="entry")` and `leg="exit"` at k = 0 reproduce `pct_returns(t)` and
   `usd_per_contract(t, point_value)` exactly; `n_alive == len(t)`, `n_skipped == 0`,
   `retention == 1.0`.
2. **Linear ramp, exact values.** One long trade entry_idx 10 (price 110), exit_idx 20 (price
   120). Entry delay k: `pct == 120 / (110 + k) - 1` for k = 1..9. Exit delay k:
   `pct == (120 + k) / 110 - 1`. Same trade short: both negated. `usd == direction x diff x point_value`.
3. **Entry delay skips trades past their exit.** Trade with hold 3 (entry 10, exit 13):
   alive for k = 0, 1, 2; skipped for k >= 3. `n_alive` per k is `[1, 1, 1, 0, 0, ...]`,
   `per_trade_pct` is NaN where skipped, `total_usd` is 0.0 there (sum over an empty set) and
   `mean_*` / `win_rate` are NaN.
4. **Exit delay skips trades past the series end.** Bars n = 30, trade exits at idx 28:
   alive k = 0, 1; skipped k >= 2.
5. **fixed_hold mode.** Entry delay k in fixed_hold equals `d x (open[exit+k] / open[entry+k] - 1)` and
   is skipped only when `exit + k > n - 1`. The exit curve is identical in both modes.
6. **first_nonpositive_k.** A ramp long trade with entry delay: total stays positive → `None`.
   A ramp where the exit price is 111 and entry 110: k = 1 gives 111/111 - 1 = 0 → `first_nonpositive_k == 1`.
7. **Retention when the baseline is not positive.** A losing strategy at k = 0 → `retention` NaN for all k (never a misleading ratio).
8. **Validation.** `max_k < 1`, non-int, `mode` not in the two names, `leg` not in the two names → `ValueError`. Empty alive set at k = 0 cannot happen after a passing join; assert `run_timing` raises `ValidationFailed` when trade timestamps are not in the bars (reuse the pattern in `tests/test_battery.py`).
9. **JSON.** `to_json(run_timing(...))` parses, `json.dumps(json.loads(s), allow_nan=False)` does not raise, top-level keys are the `TimingResult` fields, `profit_factor` inf → null.
10. **Determinism / no mutation.** Calling `run_timing` twice gives equal results; the input trade DataFrame is unchanged (compare before/after).
11. **Real sample oracle** (`tests/test_timing_real_sample.py`, `skipif` like `tests/test_real_sample.py`): 418 trades, `entry.points[0].n_alive == 418`, k = 0 `total_usd` equals `usd_per_contract(...).sum()`, every point finite, and print the two curves so the reviewer can eyeball them.

### 5.2 App smoke, `tests/test_timing_app_smoke.py`

Mirror `tests/test_app_smoke.py` with `streamlit.testing.v1.AppTest`:

1. No files → the "Upload both files" info.
2. Sample files via env + `session_state["sample"] = True` → no exception; markdown contains
   "Entry delay" and "Exit delay"; regex `(?<!\\)\$\d` finds no unescaped dollar amount; `"\\$"` present.
3. Demo (`session_state["demo"] = True`) → no exception, both cards rendered.
4. Charts: `fig_delay_curve` returns a figure with exactly `max_k + 1` x points on the first
   trace, and a mode change re-renders without exception.

### 5.3 Verifications before pushing

- `python -m pytest -q` all green (existing suite plus the new tests).
- `python -m streamlit run app/timing_app.py` starts without an import error (AppTest covers this).
- `docker compose config` validates the compose file even where Docker cannot build.
- Docker build + in-container pytest where Docker exists.
- Re-read the diff: no change to `robustness/battery.py` behaviour, no new dependency in
  `requirements.txt`, no sample data committed (`samples/` stays ignored).

## 6. Out of scope (say so in the README, do not build)

- Re-running the strategy's own stop/target logic on the shifted position.
- Cost haircuts, position limits, overlapping-trade netting.
- Earlier-than-signal entries (we cannot know the signal earlier than it fired).
- A gate or p-value. This is a picture, not a test with a verdict.
