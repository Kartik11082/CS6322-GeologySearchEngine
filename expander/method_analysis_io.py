"""Write Section 6 method-analysis artifacts under expander/textfiles/."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

TEXTFILES_ROOT = PACKAGE_DIR / "textfiles"


def write_query_textfiles(
    method_folder: str,
    q_idx: int,
    lds_text: str,
    stems_text: str,
    correlation_text: str,
) -> None:
    """method_folder is 'Associative' | 'Scalar' | 'Metric'."""
    out_dir = TEXTFILES_ROOT / method_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = (
        ("lds", lds_text),
        ("stems", stems_text),
        ("correlation", correlation_text),
    )
    for suffix, body in mapping:
        p = out_dir / f"q{q_idx}_{suffix}.txt"
        p.write_text(body.rstrip() + "\n", encoding="utf-8")


def merge_textfiles_into_report(method_folder: str, header: str, report_filename: str) -> Path:
    """
    Concatenate q1–q3 lds/stems/correlation for a readable single report alongside textfiles/.
    """
    lines = [header.rstrip()]
    folder = TEXTFILES_ROOT / method_folder
    for q_idx in range(1, 4):
        for part in ("lds", "stems", "correlation"):
            p = folder / f"q{q_idx}_{part}.txt"
            if p.exists():
                lines.append(p.read_text(encoding="utf-8").rstrip())
                lines.append("")
    report_path = PACKAGE_DIR / report_filename
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path
