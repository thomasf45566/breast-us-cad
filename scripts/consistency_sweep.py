"""Cross-document numeric consistency sweep (P6, rewritten P8' 2026-09-12).

Research prototype — not for diagnostic use.

For every numeric value that appears in two or more of the human-facing
documents, record where it appears (file:line, with context) and look for
it in RESULTS.md, data/external_protocol.md or a committed aggregate
CSV/JSON artifact under reports/ or models/.

CONTEXT-AWARE MATCHING (re-audit 2026-09-11, docs/AUDIT2_astra_2026-09-11.md
§2.6 and docs/AUDIT2_claude_2026-09-11.md "consistency_sweep.py re-run"):
the P6 version called a value "verified" whenever the same number occurred
anywhere in any source, so "55" (women under 55) was matched by a sha256
line and "4.0" (CC BY 4.0) by a parameter count. Now every document line
and every source line/cell is reduced to a set of semantic TAGS (cohort,
metric, quantity kind — see TAGS below), and a source number counts only
if it shares at least one tag with the document line. Statuses:

  verified   the number (exactly, or a source number rounding to it at the
             document's precision) occurs in a source line/cell that shares
             a context tag with the document line
  weak       the number occurs in a source, but the document line carries
             no recognizable tag — a coincidence cannot be excluded; listed
             for manual reading
  derived    a documented arithmetic derivation from committed artifacts
  UNVERIFIED no source number matches under a shared tag

CSV sources are matched per CELL (context = column name + the row's string
cells); JSON sources per key (+ file name); markdown sources per line.
models/CHECKSUMS.txt is no longer a source (hex digests are not numbers).

WHAT THIS STILL CANNOT DO: it checks that a number exists somewhere with
compatible context; it does not check that a sentence's claim about that
number (direction, predictor, inequality, cohort pairing) is correct. The
claim-level checks are the manual tables in docs/AUDIT_RESPONSE_2026-09-06.md
and docs/AUDIT2_RESPONSE.md. Output: docs/consistency_check.md.

Usage: python scripts/consistency_sweep.py
"""

import csv
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
PER_ROW = re.compile(r"_preds\.csv$|_draws\.csv$|^v2_members_|resize_dispatch_impact|^oof_|phash_sweep_hits|occlusion_border|seg_metrics|_curves\.csv$|posthoc_predictor_shift_members")
MD_SOURCES = ["RESULTS.md", "data/external_protocol.md"]
CSV_SOURCES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "reports").glob("*.csv") if not PER_ROW.search(p.name))
JSON_SOURCES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "models").glob("*.json"))
OUT = ROOT / "docs" / "consistency_check.md"

NUM = re.compile(r"(?<![\w.\-,])[+−-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.\-/,])[+−-]?\d+\.\d+|(?<![\w.\-/:,])\d{2,}(?![\w.:\-/,])")
# tokens that are not results: dates, times, hashes, line numbers, sizes, years, section refs, licences, ages, talk timings
SKIP_CONTEXT = re.compile(r"(20\d\d-\d\d-\d\d|\d\d:\d\d|[0-9a-f]{7,}|line[s]? \d|:\d{2,4}\b|§|epoch|commit|Zenodo|zenodo\.|doi|DOI|arXiv|ISO|Version|version|v\d\.\d|dpi|px|MB|GB|bytes|seed|R = |R=|×2000|2000 iter|patience|CC BY|Apache|歲|\d+ ?秒|< ?\d+ ?s\b|pytest|\d+ tests|\[\d:\d\d|TMI|vol\.|pp\.|10\.1\d{3}/)", re.I)
KEEP_CONTEXT = re.compile(r"AUC|sens|spec|κ|ECE|NLL|thr|閾值|T =|T=|median|中位|ΔAUC|Δspec|k\*|k=|pairs|\d ?對|images|張|n=|warm|latency|冷啟|Space|Dice|IoU", re.I)
HEX_LINE = re.compile(r"[0-9a-f]{32,}")

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
    "76": "external false negatives 8 + 7 + 11 + 50 (RESULTS.md:201–202 confusions)",
    "561": "external false positives 91 + 80 + 238 + 152 (RESULTS.md:201–202 confusions)",
    "1,360": "external malignant images 98 + 163 + 375 + 724 (RESULTS.md:201–202 tp + fn)",
    "0.09": "|2.2717 − 2.3644| = 0.0927 (reports/v2_cross_site.csv BrEaST src_temperature vs models/calibration.json)",
    "0.14": "|2.2253 − 2.3644| = 0.1391 (reports/v2_cross_site.csv SYSUCC src_temperature vs models/calibration.json)",
    "2432": "sum of thresholds_changed in reports/posthoc_threshold_rule_M1.csv (259 + 372 + 604 + 1197)",
    "11000": "sum of n_draws in reports/posthoc_threshold_rule_M1.csv (22 cells × 500)",
}

# tag name -> regex (applied to document lines and to source lines/cells)
TAGS = {
    "breast": r"BrEaST|breast_poland|波蘭|\bbreast\b",
    "busi": r"BUSI(?!_WHU)|埃及|\bbusi\b",
    "gdph": r"GDPH|廣東|\bgdph\b",
    "sysucc": r"SYSUCC|中山|\bsysucc\b",
    "internal": r"internal|OOF|BUS-BRA|內部|pooled|busbra|巴西",
    "auc": r"AUC(?!ROC)|判別|auc",
    "auroc": r"AUROC|error_auroc|誤判",
    "dauc": r"ΔAUC|delta_auc|delta-auc",
    "dspec": r"Δspec|delta_spec|spec gain|Δ ?specificity",
    "sens": r"sens|敏感|TPR|tpr",
    "spec": r"spec(?!t)|特異|specificity",
    "ppv": r"PPV|NPV|ppv|npv",
    "kappa": r"κ|kappa|agreement|一致",
    "ece": r"ECE|ece", "nll": r"NLL|nll",
    "temp": r"temperature|\bT\b ?[=≈]|\bT\b|T_local|refit T|重估 T",
    "thr": r"thr|threshold|閾值|operating point|工作點|Youden",
    "median": r"median|中位",
    "benign": r"benign|良性", "malignant": r"malignant|惡性",
    "recovery": r"recovery|恢復|oracle",
    "kstar": r"k\*|k_reliable|kstar|k_star|reliab|可靠",
    "kgrid": r"\bk ?= ?\d|k=\d|\bk\b",
    "seg": r"Dice|IoU|U-Net|分割|seg",
    "latency": r"latency|warm|cold|秒|s/image|/image|冷啟|startup|CPU inference|ms\b|快 \d+ 倍|faster|measurement|量測",
    "phash": r"pairs|\b對\b|pHash|phash|hash|雜湊|dup|重複|conflict|衝突|d ?[≤<=] ?8|near-dup|keep-list|keep_list|去重|removed",
    "count": r"images|\b張|n ?=|病人|patients|cases|folds?|影像|例|rows|n_images|\bn\b",
    "ci": r"\bCI\b|95%|ci95|_lo\b|_hi\b|percentile|band|帶",
    "prev": r"prevalence|盛行",
    "fitfail": r"fit.failed|擬合失敗|fallback|回退|non-positive",
    "degenerate": r"degenerate|單類別",
    "abst": r"abstention|abstain|\bq ?=|retained|轉介|保留集",
    "margin": r"margin", "ustd": r"U_std|U_range|disagreement|分歧",
    "enrich": r"enrich|富集",
    "loco": r"LOCO|loco|多來源|hold.?out|held.?out|multi-source|v1single|v1-single|單模型|single model",
    "biomedclip": r"BiomedCLIP|biomedclip|Q2|骨幹|shift_ratio|漂移比|ratio|USFM|domain-pretrained|領域預訓練",
    "resize": r"resize|dispatch|KleidiCV|gray|灰階|abs_delta|Δp|差 ?≤|翻轉|flip",
    "optimism": r"optimism|樂觀|fixed epoch|固定 epoch|best[- ]epoch|epoch",
    "reader": r"reader|判讀|BI-RADS|BIRADS|radiolog|≥ ?4a|放射",
    "xsite": r"cross[- ]site|跨場域|transfer|轉移|source|src_",
    "method": r"\bM1\b|\bM2a\b|\bM2b\b|\bM3\b|Platt|recalib|重校準|local",
    "nested": r"nested|嵌套|leave-fold|out-of-sample|in-sample|folds ≠|其餘四折",
    "params": r"params|Params|參數|\bM\)",
    "cv": r"\bCV\b|五折|5-fold|cross[- ]valid|cv_",
    "calib": r"calibrat|校準|p_cal|prob",
    "space": r"Space|HF|Hugging|live|線上|demo|展示",
    "ensemble": r"ensemble|5-ckpt|五模型|member|成員|ckpt|checkpoint",
    "tta": r"TTA|hflip|翻轉",
    "occl": r"occlusion|border|drop",
    "checks": r"self-check|selfcheck|digit-for-digit|逐位",
    "platt": r"slope|negative|負",
}
TAG_RE = {k: re.compile(v, re.I) for k, v in TAGS.items()}
COHORT_TAGS = {"breast", "busi", "gdph", "sysucc", "internal"}


SERIES = re.compile(r"\d(?:\.\d+)?\s*/\s*[+−-]?\d(?:\.\d+)?")


def tags_of(text: str) -> set[str]:
    t = {k for k, r in TAG_RE.items() if r.search(text)}
    if len(t & COHORT_TAGS) >= 2 or SERIES.search(text):
        t.add("multi")  # line spans several cohorts: cohort exclusivity does not apply
    return t


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


def extract(path: str) -> dict[str, list[tuple[int, str, frozenset]]]:
    """token -> [(line, context snippet, tags of the whole line)]"""
    lines = load(path)
    if path == "app/app.py":
        lines = about_text_only(lines)
    found: dict[str, list[tuple[int, str, frozenset]]] = defaultdict(list)
    for i, ln in enumerate(lines, 1):
        if ln.lstrip().startswith("|---") or ln.lstrip().startswith("http"):
            continue
        line_tags = frozenset(tags_of(ln))
        for m in NUM.finditer(ln):
            tok = norm(m.group(0))
            ctx = ln[max(0, m.start() - 40): m.end() + 40].strip()
            if SKIP_CONTEXT.search(ctx) and not KEEP_CONTEXT.search(ctx):
                continue
            if re.fullmatch(r"\d{4}", tok) and 1990 <= int(tok) <= 2100:
                continue
            if re.search(r"--python|^#+ \d|^### \d|§\s?\d|\(§|see §", ln.strip()) and re.fullmatch(r"\d\.\d{1,2}", tok):
                continue  # python version / section numbers
            found[tok].append((i, ctx, line_tags))
    return found


SourceNum = tuple[str, str, int, frozenset]  # (number, file, line, tags)


def source_numbers() -> list[SourceNum]:
    nums: list[SourceNum] = []
    num_re = re.compile(r"[+−-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?")
    for src in MD_SOURCES:
        for i, ln in enumerate(load(src), 1):
            if HEX_LINE.search(ln):
                continue
            t = frozenset(tags_of(ln))
            for m in num_re.finditer(ln):
                nums.append((norm(m.group(0)), src, i, t))
    for src in CSV_SOURCES:
        with (ROOT / src).open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header = rows[0]
        file_tags = tags_of(Path(src).stem.replace("_", " "))
        for i, row in enumerate(rows[1:], 2):
            strings = " ".join(c for c in row if not num_re.fullmatch(c.strip()))
            row_tags = tags_of(strings) | file_tags
            for col, cell in zip(header, row):
                cell = cell.strip()
                if not num_re.fullmatch(cell):
                    continue
                t = frozenset(row_tags | tags_of(col.replace("_", " ")))
                nums.append((norm(cell), src, i, t))
    for src in JSON_SOURCES:
        file_tags = tags_of(Path(src).stem.replace("_", " "))
        for i, ln in enumerate(load(src), 1):
            if HEX_LINE.search(ln) or '"' not in ln and not re.search(r"\d", ln):
                continue
            t = frozenset(tags_of(ln.replace("_", " ")) | file_tags)
            for m in num_re.finditer(ln):
                nums.append((norm(m.group(0)), src, i, t))
    return nums


def rounds_to(candidate: str, target: str) -> bool:
    try:
        c = float(candidate.replace(",", "")); t = float(target.replace(",", ""))
    except ValueError:
        return False
    dec = len(target.split(".")[1]) if "." in target else 0
    return abs(round(c, dec) - t) < 1e-12 and (dec > 0 or abs(c - t) < 0.5)


def compatible(doc_tags: frozenset, src_tags: frozenset) -> bool:
    """Shared context: at least one common tag; and if both sides name a
    cohort, the cohorts must intersect (a BrEaST number cannot verify a
    SYSUCC sentence)."""
    if not ((doc_tags - {"multi"}) & (src_tags - {"multi"})):
        return False
    if "multi" in doc_tags:
        return True
    dc, sc = doc_tags & COHORT_TAGS, src_tags & COHORT_TAGS
    return not (dc and sc) or bool(dc & sc)


def verify(tok: str, doc_tags: frozenset, src_nums: list[SourceNum]) -> tuple[str, str]:
    if tok in DERIVED:
        return "derived", DERIVED[tok]
    plain = tok.replace(",", "")
    dec = len(tok.split(".")[1]) if "." in tok else 0
    exact = [(f, i, t) for n, f, i, t in src_nums if n == tok or n.replace(",", "") == plain]
    rounded = [(n, f, i, t) for n, f, i, t in src_nums if dec >= 2 and len(n) > len(tok) and rounds_to(n, tok)]
    if doc_tags:
        for f, i, t in exact:
            if compatible(doc_tags, t):
                return "verified", f"{f}:{i} (exact; shared context {','.join(sorted(doc_tags & t))})"
        for n, f, i, t in rounded:
            if compatible(doc_tags, t):
                return "verified", f"{f}:{i} ({n} rounded; shared context {','.join(sorted(doc_tags & t))})"
        return "UNVERIFIED", "no source number with compatible context equals or rounds to this value"
    if exact:
        f, i, _ = exact[0]
        return "weak", f"{f}:{i} (exact) — document line has no recognizable context"
    if rounded:
        n, f, i, _ = rounded[0]
        return "weak", f"{f}:{i} ({n} rounded) — document line has no recognizable context"
    return "UNVERIFIED", "no source number equals or rounds to this value"


# Manual adjudication of tokens the tag matcher cannot place (kept in the
# script so regeneration is deterministic; every entry names its source).
MANUAL = {
}


ADJUDICATION_TAIL = """

## What this check establishes — and what it does not (P8', 2026-09-12)

- A **verified** value exists in RESULTS.md / a committed aggregate CSV/JSON / the protocol
  in a line or cell that shares at least one context tag (cohort, metric, quantity kind) with
  the document line, and never with a conflicting cohort. This rules out the P6-era false
  positives (a sha256 digit run, a parameter count for "4.0", a Youden threshold for κ) but
  it is still a number-with-compatible-context check, not a claim check: it cannot tell a
  correct sentence from a wrong one that quotes a real number with the wrong predictor,
  direction or inequality. Those checks are manual: docs/AUDIT_RESPONSE_2026-09-06.md and
  docs/AUDIT2_RESPONSE.md.
- **weak** values are found in a source but the document line carries no tag the matcher
  knows; they are listed so a reader can judge them, and are NOT counted as verified.
- **derived** values carry their arithmetic in the table.
- **UNVERIFIED** values are listed with their first occurrence; the manual adjudication of
  each is in docs/AUDIT2_RESPONSE.md (section "Consistency sweep").
- Excluded by design: dates, times, commit hashes, line numbers, section numbers, file
  sizes, seeds, bootstrap iteration counts, licence versions, ages, talk timings.
- Single-document values are out of scope (not a cross-document question).
"""


def main() -> None:
    per_file = {d: extract(d) for d in DOCS}
    values: dict[str, dict[str, list[tuple[int, str, frozenset]]]] = defaultdict(dict)
    for d, found in per_file.items():
        for tok, occ in found.items():
            values[tok][d] = occ
    shared = {tok: files for tok, files in values.items() if len(files) >= 2}
    src_nums = source_numbers()
    rows, counts = [], defaultdict(int)
    unverified = []
    for tok in sorted(shared, key=lambda t: (float(t.replace(",", "")) if re.fullmatch(r"-?[\d,]+(\.\d+)?", t) else 0)):
        # union of the tags over every document occurrence of the token
        doc_tags = frozenset().union(*[t for occ in shared[tok].values() for _, _, t in occ])
        if tok in MANUAL:
            status, source = "verified (manual)", MANUAL[tok]
        else:
            status, source = verify(tok, doc_tags, src_nums)
        counts[status.split(" ")[0]] += 1
        files = "; ".join(f"{d}:{','.join(str(i) for i, _, _ in occ[:6])}" for d, occ in shared[tok].items())
        first = next(iter(shared[tok].values()))[0]
        ctx = first[1].replace("|", "\\|")
        rows.append(f"| `{tok}` | {files} | {source} | {status} | {ctx} |")
        if status == "UNVERIFIED":
            unverified.append((tok, shared[tok]))
    head = f"""# Cross-document consistency check (generated by scripts/consistency_sweep.py, context-aware since P8' 2026-09-12)

Scope: every numeric value appearing in ≥ 2 of: {", ".join(DOCS)} (for app/app.py only the About text).
Sources: RESULTS.md, data/external_protocol.md, aggregate reports/*.csv (per cell), models/*.json (per key).
A value is **verified** only if a source occurrence shares a context tag (cohort / metric / quantity kind) with
the document line; **weak** = number found but the document line has no recognizable context; **derived** values
carry their derivation; **UNVERIFIED** values have no compatible-context source and are adjudicated by hand
in docs/AUDIT2_RESPONSE.md.

Shared values: {len(shared)} · verified: {counts['verified']} · weak: {counts['weak']} · derived: {counts['derived']} · UNVERIFIED: {counts['UNVERIFIED']}

| value | files:lines | source | status | context (first occurrence) |
|---|---|---|---|---|
"""
    OUT.write_text(head + "\n".join(rows) + ADJUDICATION_TAIL, encoding="utf-8")
    print(f"shared values: {len(shared)}; verified: {counts['verified']}; weak: {counts['weak']}; "
          f"derived: {counts['derived']}; UNVERIFIED: {counts['UNVERIFIED']}")
    for tok, files in unverified:
        print(f"  UNVERIFIED {tok}: " + "; ".join(f"{d}:{[i for i, _, _ in occ][:4]}" for d, occ in files.items()))
        print("      ctx:", next(iter(next(iter(files.values()))))[1][:110])
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
