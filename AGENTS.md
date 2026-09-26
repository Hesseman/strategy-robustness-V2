# Strategy Robustness App V2

Streamlit app (Docker): a TradeStation performance report plus its bars in → five robustness
cards and a timing section; optionally a MultiWalk optimisation in → WFC, plateau and selection
cards. The statistics live in `robustness/`, the UI in `app/`.

## Guardrails

- **Public repository.** Real reports, bars and MultiWalk files stay on the owner's disk. Tests
  and the dev buttons read them through `SR_SAMPLE_REPORT`, `SR_SAMPLE_BARS` and
  `SR_SAMPLE_MW_DIR`, and skip when those are absent.
- **Self-contained statistics.** `robustness/` uses numpy, pandas, plotly and the standard library
  only. Streamlit stays in `app/`.
- **Fixed verdict knobs.** Region share 0.2, pooling radius 1, sign-flip block 21 trading days,
  alpha 0.05. They are set a priori, never tuned per project. Changing one is a method change:
  re-run the oracles and record the decision in `docs/wfc-region-lift.md`.

## Before changing

- **WFC card** (region lift, sign-flip null, verdict matrix, printed guards, N_eff): read
  `docs/wfc-region-lift.md` for the method, the decisions and their reasons, the evidence, the
  known gaps and the code map.
- **Timing section:** read `docs/plans/2026-09-23-timing-sensitivity.md`.

## Tests

Run `.venv/Scripts/python -m pytest -q` (`.venv/bin/python` off Windows). Oracles are
hand-traced literals or planted-truth synthetics. `tests/test_real_multiwalk.py` reproduces the
research prototype to the draw, so a changed number there means the statistics changed.
