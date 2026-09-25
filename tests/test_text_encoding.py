"""Source and README text is real UTF-8, never UTF-8 read as cp1252 and written back (mojibake
such as 'â†’' for '→' or 'Â·' for '·'), which the page would show verbatim."""
import re
from pathlib import Path

MOJIBAKE = re.compile("[ÂÃâ][\u0080-¿‘-›€Œ-ƒˆ˜]")
ROOT = Path(__file__).resolve().parents[1]


def test_no_mojibake_in_sources_or_readme():
    files = sorted(ROOT.glob("app/*.py")) + sorted(ROOT.glob("robustness/*.py")) + [ROOT / "README.md"]
    bad = {str(p.relative_to(ROOT)): MOJIBAKE.findall(p.read_text(encoding="utf-8")) for p in files}
    assert not {k: v for k, v in bad.items() if v}


def test_the_pattern_catches_a_double_encoded_arrow():
    assert MOJIBAKE.search("0 → 1".encode("utf-8").decode("cp1252")) and not MOJIBAKE.search("0 → 1 · 5 × CDaR-80")
