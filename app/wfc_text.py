"""The WFC card's 'Read with the PASS' line - the facts a PASS is printed with (docs/wfc-region-lift.md,
'Printed guards'). Plain text with bare $; the card escapes it for st.markdown. Only reason to
change: what a PASS is read with."""
from __future__ import annotations

import math

from robustness.multiwalk_battery import NEFF_MIN


def signed_usd(v: float) -> str:
    """Signed dollars for a margin: '+$9,180', '−$7,193'."""
    return f"{'+' if v >= 0 else '−'}${abs(v):,.0f}"


def _p(p: float) -> str:
    return "p n/a" if not math.isfinite(p) else (f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}")


def read_with_the_pass(guards: dict, p_flip: float) -> str:
    """The 'Read with the PASS' line.

    Accepts: meta['wfc_guards'] of a PASS (run_multiwalk_battery) and the pooled lift's sign-flip p
    (the gate's). Returns one markdown line: the lift's p under the sign-flip null and under the
    centred block bootstrap, with the disagreement flagged when guards['nulls_disagree']; the
    effective number of variants with the few-variants caution (left out when not finite); the
    region's and the grid's out-of-sample dollars; the region's margin in each complete window."""
    p_boot = guards["p_lift_boot"]
    if guards["nulls_disagree"]:
        nulls = (f"lift {_p(p_flip)} under the sign-flip null (the gate) but {_p(p_boot)} under the centred block bootstrap "
                 "(its cross-check) - **the two nulls disagree**: the PASS rests on the flip's assumption of symmetric block "
                 "deviations, which a few large divergent trades can break, so read it as borderline; ")
    else:
        nulls = (f"lift {_p(p_flip)} under the sign-flip null (the gate), {_p(p_boot)} under the centred block bootstrap "
                 "(its cross-check; the two agree); ")
    few = f" - fewer than {NEFF_MIN:g}, so the lift rests on the handful of trades where they differ" if guards["few_variants"] else ""
    neff = (f"≈ **{guards['n_eff_median']:.1f}** effective independent variants{few}; "
            if math.isfinite(guards["n_eff_median"]) else "")
    region, grid = guards["region_oos_mean"], guards["grid_oos_mean"]
    return (f"**Read with the PASS:** {nulls}{neff}out of sample the region made **${region:,.0f}** against the grid's "
            f"**${grid:,.0f}** ({signed_usd(region - grid)}); ahead of the grid in **{guards['windows_ahead']} of "
            f"{guards['windows_complete']}** complete window(s): " + ", ".join(signed_usd(m) for m in guards["region_minus_grid"]))
