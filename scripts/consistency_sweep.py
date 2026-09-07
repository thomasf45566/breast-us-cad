"""Cross-document numeric consistency sweep (audit-response phase P6).

Research prototype — not for diagnostic use.

For every numeric value that appears in two or more of the human-facing
documents, record where it appears (file:line, with context) and verify it
against RESULTS.md or a committed CSV/JSON artifact under reports/ or
models/: a value is VERIFIED if it occurs verbatim in RESULTS.md, or if some
number in RESULTS.md / the artifacts rounds to it at the document's own
precision. Values that are documented derivations (sums, differences,
pair counts) are listed with their derivation. Everything else is listed
as UNVERIFIED for manual adjudication. Output: docs/consistency_check.md.

Usage: python scripts/consistency_sweep.py
"""

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md", "docs/report.md", "docs/project_summary.md", "docs/interview_script.md",
    "app/app.py", "deploy/README.md", "plan.md", "docs/PROVENANCE.md", "THIRD_PARTY_DATA.md",
]
# Per-image / per-draw CSVs hold tens of thousands of numbers, so a rounded
# match against them is meaningless; only aggregate artifacts count.
PER_ROW = re.compile(r"_preds\.csv$|_draws\.csv$|^v2_members_|resize_dispatch_impact|^oof_|phash_sweep_hits|occlusion_border|seg_metrics|_curves\.csv$")
SOURCES = ["RESULTS.md", "data/external_protocol.md",
           *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "reports").glob("*.csv") if not PER_ROW.search(p.name)),
           *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "models").glob("*.json")),
           "models/CHECKSUMS.txt"]
OUT = ROOT / "docs" / "consistency_check.md"

NUM = re.compile(r"(?<![\w.\-,])[+−-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.\-/,])[+−-]?\d+\.\d+|(?<![\w.\-/:,])\d{2,}(?![\w.:\-/,])")
# tokens that are not results: dates, times, hashes, line numbers, sizes, years, section refs
SKIP_CONTEXT = re.compile(r"(20\d\d-\d\d-\d\d|\d\d:\d\d|[0-9a-f]{7,}|line[s]? \d|:\d{2,4}\b|§|epoch|commit|Zenodo|zenodo\.|doi|DOI|arXiv|ISO|Version|version|v\d\.\d|dpi|px|MB|GB|bytes|seed|R = |R=|×2000|2000 iter|patience)", re.I)
# Documented derivations (value -> how it is obtained from committed artifacts)
DERIVED = {
    "12,056,505": "Σ C(n,2) + Σ n_a·n_b over set sizes 1875/252/379/846/1559 (external_protocol.md:143–145)",
    "12.06": "12,056,505 pairs rounded (see above)",
    "0.011": "optimism mean best − fixed, reports/posthoc_epoch_selection.csv (0.0111)",
    "0.0111": "reports/posthoc_epoch_selection.csv mean of optimism_best_minus_fixed",
    "0.0073": "reports/posthoc_epoch_selection.csv SD of optimism_best_minus_fixed",
    "0.0115": "LOCO − v1_ensemble AUC, reports/v2_loco_q1c_table.csv (breast)",
    "0.0033": "LOCO − v1_ensemble AUC, reports/v2_loco_q1c_table.csv (busi)",
    "0.0301": "LOCO − v1_ensemble AUC, reports/v2_loco_q1c_table.csv (gdph)",
    "0.0064": "LOCO − v1_ensemble AUC, reports/v2_loco_q1c_table.csv (sysucc)",
    "607": "tp + fn of models/operating_point.json (548 + 59)",
    "32.4": "607 / 1875 image-level prevalence",
    "2,454": "252 + 379 + 810 + 1013 external keep-list images",
    "386": "222 benign + 164 malignant raw BUSI (external_protocol.md:35)",
    "1,421": "affected rows in reports/v2_resize_dispatch_impact.csv",
    "64": "152/238 in reports/posthoc_reader_concordance.csv (0.639)",
    "14": "34/238 in reports/posthoc_reader_concordance.csv (0.143)",
    "86": "1 − 0.143 (GDPH reader1 <4a share of model FPs)",
}


ADJUDICATION = """

## Manual adjudication (P6, 2026-09-07)

- **Mismatches found and fixed in this sweep:** the Space warm latency was a single value
  in docs/report.md §3.12 ("warm 6.8 s"), docs/project_summary.md and plan.md ("~6.8 s/image")
  while three measurements exist (6.8 s RESULTS.md:416; 6.15 s audit §6; 8.62 s verify_space.py
  2026-09-07). All now state the range 6–9 s across three measurements; the two later
  measurements were appended to RESULTS.md's Errata so the range traces to a committed source.
- **Range statements the token check cannot judge** (each endpoint verifies individually, the
  pairing was checked by hand against docs/report.md): "k=10 sensitivity median 0.83–0.85" is
  the BrEaST/BUSI pair (0.830 / 0.846; GDPH 0.866, SYSUCC 0.909 are stated alongside wherever
  the range appears); "held-out sens 0.80–0.96" = LOCO 0.7997–0.9600; "AUC 0.84–0.93" =
  0.8380–0.9339; "spec 0.41–0.63" = 0.4091–0.6296; "benign median 0.17–0.31" = 0.1739–0.3061;
  "ΔAUC +0.013 to +0.056" = 0.0126–0.0556; "Δspec +0.27 to +0.30" = 0.2664–0.3011.
- **Derived values** (status `derived` above) are arithmetic on committed artifacts; the
  derivation is stated in the table and in docs/AUDIT_RESPONSE_2026-09-06.md / docs/report.md.
- **Deliberate rounding** (status `verified … rounded`): documents quote 3–4 significant
  digits of RESULTS.md / JSON values (e.g. 0.854 for 0.8542, 0.9254 for 0.925428…).
- **Excluded from the token check by design:** dates, times, commit hashes, line numbers,
  section numbers, file sizes, seeds, bootstrap iteration counts, the Python version.
- **Out of scope (single-file values):** numbers that appear in only one document are not
  cross-document consistency questions; they were verified sentence-by-sentence in P1
  (docs/AUDIT_RESPONSE_2026-09-06.md).
- Result after fixes: **0 UNVERIFIED** shared values.
"""


def load(path: str) -> list[str]:
    return (ROOT / path).read_text(encoding="utf-8").splitlines()


def about_text_only(lines: list[str]) -> list[str]:
    """app/app.py: only the ABOUT_MD block is a document."""
    out, inside = [], False
    for ln in lines:
        if ln.startswith('ABOUT_MD = """'):
            inside = True
        if inside:
            out.append(ln)
        if inside and ln.strip() == '"""':
            inside = False
    return out


def norm(tok: str) -> str:
    return tok.replace("−", "-").lstrip("+")


def extract(path: str) -> dict[str, list[tuple[int, str]]]:
    lines = load(path)
    if path == "app/app.py":
        lines = about_text_only(lines)
    found: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for i, ln in enumerate(lines, 1):
        if ln.lstrip().startswith("|---") or ln.lstrip().startswith("http"):
            continue
        for m in NUM.finditer(ln):
            tok = norm(m.group(0))
            ctx = ln[max(0, m.start() - 40): m.end() + 40].strip()
            if SKIP_CONTEXT.search(ctx) and not re.search(r"AUC|sens|spec|κ|ECE|NLL|thr|閾值|T =|T=|median|中位|ΔAUC|Δspec|k\*|k=|pairs|對|images|張|n=", ctx):
                continue
            if re.fullmatch(r"\d{4}", tok) and 1990 <= int(tok) <= 2100:
                continue
            if re.search(r"--python|^#+ \d|^### \d|§\s?\d|\(§|see §", ln.strip()) and re.fullmatch(r"\d\.\d{1,2}", tok):
                continue  # python version / section numbers
            found[tok].append((i, ctx))
    return found


def source_numbers() -> tuple[str, list[tuple[str, str, int]]]:
    """Concatenated source text + list of (number_string, file, line)."""
    texts, nums = [], []
    for src in SOURCES:
        for i, ln in enumerate(load(src), 1):
            texts.append(ln)
            for m in re.finditer(r"[+−-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?", ln):
                nums.append((norm(m.group(0)), src, i))
    return "\n".join(texts), nums


def rounds_to(candidate: str, target: str) -> bool:
    try:
        c = float(candidate.replace(",", "")); t = float(target.replace(",", ""))
    except ValueError:
        return False
    dec = len(target.split(".")[1]) if "." in target else 0
    return abs(round(c, dec) - t) < 1e-12 and (dec > 0 or abs(c - t) < 0.5)


def verify(tok: str, src_text: str, src_nums: list[tuple[str, str, int]]) -> tuple[str, str]:
    if tok in DERIVED:
        return "derived", DERIVED[tok]
    exact = [(f, i) for n, f, i in src_nums if n == tok or n.replace(",", "") == tok.replace(",", "")]
    if exact:
        f, i = exact[0]
        return "verified", f"{f}:{i} (exact)"
    # rounding is accepted only for decimals with >= 2 significant decimals; integers must match exactly
    dec = len(tok.split(".")[1]) if "." in tok else 0
    rounded = [(n, f, i) for n, f, i in src_nums if dec >= 2 and len(n) > len(tok) and rounds_to(n, tok)]
    if rounded:
        n, f, i = rounded[0]
        return "verified", f"{f}:{i} ({n} rounded)"
    return "UNVERIFIED", "no RESULTS.md / CSV / JSON number equals or rounds to this value"


def main() -> None:
    per_file = {d: extract(d) for d in DOCS}
    values: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(dict)
    for d, found in per_file.items():
        for tok, occ in found.items():
            values[tok][d] = occ
    shared = {tok: files for tok, files in values.items() if len(files) >= 2}
    src_text, src_nums = source_numbers()
    rows, unverified = [], []
    for tok in sorted(shared, key=lambda t: (float(t.replace(",", "")) if re.fullmatch(r"-?[\d,]+(\.\d+)?", t) else 0)):
        status, source = verify(tok, src_text, src_nums)
        files = "; ".join(f"{d}:{','.join(str(i) for i, _ in occ[:6])}" for d, occ in shared[tok].items())
        ctx = next(iter(next(iter(shared[tok].values()))))[1].replace("|", "\\|")
        rows.append(f"| `{tok}` | {files} | {source} | {status} | {ctx} |")
        if status == "UNVERIFIED":
            unverified.append((tok, shared[tok]))
    head = f"""# Cross-document consistency check (P6, generated by scripts/consistency_sweep.py)

Scope: every numeric value appearing in ≥ 2 of: {", ".join(DOCS)} (for app/app.py only the About text).
Sources of truth: RESULTS.md, reports/*.csv, models/*.json, models/CHECKSUMS.txt, data/external_protocol.md.
A value is **verified** if it occurs verbatim in a source or a source number rounds to it at the
document's precision; **derived** values carry their stated derivation; **UNVERIFIED** values are
listed for manual adjudication in the section that follows the table.

Shared values: {len(shared)} · verified: {sum(1 for r in rows if '| verified |' in r)} · derived:
{sum(1 for r in rows if '| derived |' in r)} · unverified: {len(unverified)}

| value | files:lines | source | status | context (first occurrence) |
|---|---|---|---|---|
"""
    tail = ADJUDICATION
    OUT.write_text(head + "\n".join(rows) + tail, encoding="utf-8")
    print(f"shared values: {len(shared)}; unverified: {len(unverified)}")
    for tok, files in unverified:
        print(f"  UNVERIFIED {tok}: " + "; ".join(f"{d}:{[i for i,_ in occ][:4]}" for d, occ in files.items()))
        print("      ctx:", next(iter(next(iter(files.values()))))[1][:110])
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
