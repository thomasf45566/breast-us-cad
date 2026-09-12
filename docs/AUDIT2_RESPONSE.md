# Response to the two independent re-audits of 2026-09-11 — phase P8' (2026-09-12)

Research prototype — not a medical device.

Both re-audits were run in fresh clones of HEAD 4794ad6 with the same prompt
and are archived unmodified (commit f7cbf38, no other file in that commit):

- `docs/AUDIT2_claude_2026-09-11.md` (sha256 2caabdab19fafd39999450611d9e27a959bb56b61225bce770473936dab19607) — verdict: numbers PASS; major concerns on document drift after P3/P4 and on the Zenodo archive's self-description.
- `docs/AUDIT2_astra_2026-09-11.md` (sha256 c8f56cbc4cb88f824dc240a3c6bc7dee56f34296db5a09b0a1eea4be1ed99563) — verdict: FAIL-major for the claims as written (predictor label, threshold rule, P2 wording, false-negative and reader-information sentences, enforcement claims); numbers reproduce.

**Scope of this phase.** Documentation, tooling, and two cheap post-hoc
analyses from committed artifacts only. `RESULTS.md` and
`data/external_protocol.md` were extended append-only (0 removed lines vs
2cbe1da / 4fc3fb2). No frozen artifact (`models/*.json`, `models/*.pt`,
any committed prediction or draw CSV) and no completed-study script was
changed; no retraining; no external inference. The one model forward pass
was `src/external_val.py --self-check` on BUS-BRA fold 5 (output
byte-identical to `reports/external_selfcheck.txt`), run because
`external_val.py` gained the overwrite guard.

**What was done (summary).** (A) The internal reference is relabelled
everywhere: pooled OOF 0.9254 / benign median 0.0625 / sens 0.9028 / spec
0.7713 are ONE held-out checkpoint per image + hflip TTA + T, not the
deployed ensemble; T and the threshold were fitted on that distribution
and applied to the ensemble externally. `src/posthoc_predictor_shift.py`
(RESULTS.md POST-HOC 5) separates the predictor-change from the
cohort-change component. `src/posthoc_threshold_rule.py` (POST-HOC 6)
quantifies the threshold rule as implemented (roc_curve
drop_intermediate=True) against the exhaustive rule — 0.26878 vs the
frozen 0.26832, one image; 2,432/11,000 M1 draws change, k\* unchanged —
and counts negative-slope Platt fits (12/4/3/16 at k=10). P2's "nested"
wording and "optimism" are replaced; the false "no missed cancers",
"readers had the full examination", "統計平手", "fold ranking unchanged",
temperature-difference, 0.236, 0.568 and "≤ 7×10⁻³" sentences are
corrected; v2's image-level slices are stated wherever v2 is described;
enforcement wording is made exact (Amendment 1 with 10 files; `--confirm`
intent-only; recomputation claim scoped). (B) The report is synchronised
to the P3/P4 state; README's "DOI pending" removed; `inference.py` "1879"
→ 1875; RESULTS.md:483 erratum; `scripts/deploy_hf.py` never writes under
`src/`; `deploy/` rebuilt and the Space re-pushed (sha 7d41f22;
`verify_space.py` max |Δ| 1.07e-07, warm median 6.27 s);
`scripts/consistency_sweep.py` requires shared context (cohort/metric
tags) for a match. (C) `src/external_val.py` refuses to overwrite an
existing external prediction CSV unless `--overwrite` is passed (4 new
tests; 19 pass). (D) `.gitattributes` + `.git-commit` (export-subst) make
every archive state its own commit; PROVENANCE §7 rewritten (no
self-checksum; Zenodo's checksum authoritative); tag `v1.1-audited`;
Zenodo new-version metadata prepared and STOPPED before publication
(`docs/zenodo_v1.1_new_version.md`, with the one-sentence erratum for the
v1.0 description). (E) This file.

Status legend: **corrected** (text or code changed) · **disclosed**
(recorded in RESULTS.md Errata 2026-09-12 / protocol appendix /
PROVENANCE, no artifact change possible or permitted) ·
**disputed-with-reason** · **permanent** (cannot be fixed retroactively;
stated as such).

---

## Table 1 — docs/AUDIT2_claude_2026-09-11.md

| # | Audit item (section) | Handling | Where | Status |
|---|---|---|---|---|
| C-1.i | Every checkpoint still embeds `train_folds=[1,2,3,4]`; `v2_loco.py` never adopted `checkpoint_payload` (§1(d)) | Frozen/completed artifacts and completed-study code are out of P8' scope by rule; the folds actually used are proven by the wandb configs (both audits). Recorded again. | RESULTS.md:1210–1218 (P2); plan.md P8' | disclosed |
| C-1.ii | Unrecorded second LOCO-gdph training launch (k896krp6 aborted at epoch 3; 3jn4cicy produced the checkpoint) | Recorded: restart, not a peek (held-out cohort untouched during `--train`). | RESULTS.md Errata 2026-09-12 (lines 935–940); protocol appendix 5; report §2.7 | disclosed |
| C-1.iii | LOCO 85/15 slices not persisted | Recorded; reproducible only via the seeded code (`src/v2_data.py`, tests assert determinism). | RESULTS.md Errata 2026-09-12; protocol appendix 5; report §2.7 | disclosed |
| C-1.iv / C-9 | README "8 split tests" stale | "19 tests: the 10 split tests need all five raw datasets, the other 9 run without data". | README Reproduce | corrected |
| C-1(f) / C-9 | POST-HOC 1 reproducible only from the author's wandb directory; not stated in README | README recomputation bullet now lists wandb-derived numbers as not recomputable from the repository. | README "Why this repo"; report §2.9; CITATION.cff; .zenodo.json | corrected |
| C-2 new | `models/v2_biomedclip_operating_point.json` sens 0.9012 / 547/60 vs 546/61, sens 0.8995 on the committed float32 CSV | Frozen artifact — not edited. One image at the float32 round-trip boundary; recorded in three places; verdict unaffected. | RESULTS.md Errata 2026-09-12; protocol appendix 12; report §3.10 | disclosed |
| C-2 drift 1 / C-8 items 18–20 / C-11 §8 | docs/report.md stale after P3/P4: fold file "不在版本庫內", v2 weights "尚未公開", Zenodo/OSF "後續工作" (§2.1, §2.9, §2.10, §4.5, §4.6) | All five passages rewritten to the P3/P4 state (fold file committed 2026-09-07; v2 weights on HF `v2/`, 28/28 checksums; Zenodo DOI published; OSF a forward commitment). | report §2.1, §2.9, §2.10, §4.5, §4.6 | corrected |
| C-2 drift 2 / C-8 item 13 | README:206 "Zenodo DOI pending upload" vs the DOI badge | Sentence replaced by the published DOI and the P8' state. | README Provenance | corrected |
| C-3 | Timeline self-certified; pre-rewrite objects only on one laptop | Permanent; stated. PROVENANCE §1/§4 wording sharpened (freeze → protocol → run order; Amendment 1 with 10 files; `--confirm` intent-only). | PROVENANCE §1, §4 | permanent / corrected (wording) |
| C-4.i | RESULTS.md in-place sentences wrong pending an appendix (:403, :483, :485, :1028–1033, :1058) | By design (append-only). :483 now has its erratum (it had none); all others already had one. | RESULTS.md Errata 2026-09-12 | corrected (:483) / disclosed |
| C-4.ii | M2 fit-failure fallback, patience 7, one-section-vs-per-dataset, one-figure-vs-two | Collected in one place as protocol deviations. | protocol appendix 2, 4, 5, 6 | disclosed |
| C-4.iii / C-11 §4 new | Protocol (g) cites an adjudication figure no longer distributed | Recorded; distributed evidence named; one re-auditor repeated the adjudication from raw images with the same result. | protocol appendix 8; report §2.4 | disclosed |
| C-5 | pHash reproduction PASS; "12,056,505 is a derivation, not a logged measurement" (§8 item 8) | Both re-audits enumerated the pairs independently and reproduced the count and the hits file; README now says the 35% counts 507 dup + 39 label-conflict images and scopes the conclusion to "no visually confirmed cross-set near-duplicate at d ≤ 8". | README bullet; report §2.4; summary; script | corrected |
| C-6(d) / C-11 §2 2.5 | `src/inference.py:153` and `deploy/inference.py:153` "1879 images" | 1875 in both; live Space re-pushed with it. | src/inference.py, deploy/inference.py | corrected |
| C-6(e) / C-11 §6 new | Live Space ≠ deploy/ (P3 `hf_repo_path`, P4b licence sentence never pushed); RESULTS.md:380 false at HEAD | `deploy/` rebuilt from src/app (byte-identical asserted), Space re-pushed (sha 7d41f22), `verify_space.py` max |Δ| 1.07e-07, warm median 6.27 s; RESULTS.md:380 errata'd with the dates; PROVENANCE §5 row added. | RESULTS.md Errata 2026-09-12; PROVENANCE §5; report §3.12 | corrected |
| C-6 | Warm latency 7.18 s (this audit) within 6–9 s | Now six measured medians 4.8–8.6 s (incl. Astra's 4.79 s and P8' 6.27 s); documents say "about 5–9 s". | RESULTS.md Errata; plan.md; report §3.12; summary; script | corrected |
| C-7 | "3/4 優於 margin" point-estimate language; interview script "0.83–0.85" unqualified and missing different-thresholds caveat; report §3.8 "0.03–0.04" ambiguous; README "internal range" undefined | Script: 0.83–0.91 and the caveat added; §3.8 rewritten with T differences 0.09/0.14 and spec differences ≤ 0.03/0.04; README key finding rewritten (BrEaST/SYSUCC CIs below 0.9254, BUSI/GDPH contain it; a description, not a test); "3/4 優於" in §4.3 is retained as point estimates with BrEaST overlap stated in §3.9. | README; report §3.8, §3.9; interview_script | corrected |
| C-7 residual 3 | Internal-vs-external AUC comparison uses a point estimate of a different predictor | Stated explicitly wherever the comparison appears; POST-HOC 5 added. | README; report §3.3; RESULTS.md POST-HOC 5 | corrected / disclosed |
| C-8 item 1 | "three continents" rhetorical | Retained (true; Brazil/Poland/Egypt/China). | — | disputed-with-reason |
| C-8 item 3 / C-11 8.3 | "errors are false positives, not missed cancers" | Replaced: 76 false negatives (8/7/11/50) among 1,360 malignant images; predominantly, not exclusively, false positives. | README; report abstract/§4.2/§5; summary; script | corrected |
| C-8 item 4 / 26 | "near-oracle" / "中位恢復比例近 1.0" hides overshoot | §3.7 now states recovery 0.85–1.16 at k\* and that exceeding the oracle is bought with sensitivity. | report §3.7 | corrected |
| C-8 item 6 | README "committed before the computations they govern" unqualified for Amendment 1 | Qualified in the bullet itself. | README "Why this repo"; PROVENANCE §1 | corrected |
| C-8 item 10 / C-11 new 4 | README:95 "Every number in RESULTS.md recomputes" false for wandb-derived/latency numbers | Scoped to the result tables; exceptions listed (wandb-derived, cdrop pooled OOF, latencies, CAM tallies). Same in report §2.9, CITATION.cff, .zenodo.json. | README; report §2.9; CITATION.cff; .zenodo.json | corrected |
| C-8 item 12 | README map "commit-linked" loose | Retained: RESULTS.md cites tags/run ids; every P-phase and amendment section cites its commit. | — | disputed-with-reason |
| C-8 items 15–17, 34 / C-11 new 10 | Report §1 uncited literature claims, no bibliography | §1.1 keeps the one traceable citation (Chang et al. 2020, PMID 32375295) and drops "顯著優於" and "與文獻報告相符"; §1.2 numeric AUC ranges deleted and the paragraph labelled as the author's uncited impression, not a result. | report §1.1, §1.2, abstract | corrected |
| C-8 item 25 / C-11 new 11 | §3.6 "7/8" not verifiable from the distributed repository | Sentence now says so and labels the tallies qualitative; RESULTS.md errata entry. | report §3.6; RESULTS.md Errata | corrected / disclosed |
| C-8 item 29 / C-11 new 12 | "gain is largely threshold placement" stated as fact (report §4.1, summary, script) | Relabelled as an untested hypothesis supported only by the BrEaST descriptive decomposition; Δspec "at different thresholds per model" added to the script. | report §3.11, §4.1; summary; interview_script | corrected |
| C-8 item 32 / Astra O18 | "最有支撐之結論" / "最後一哩" | Reworded: the best-supported RECOMMENDATION, resting on the non-transfer result; small-k reliability supported at the median only. | report abstract, §4.1, §5; summary; script | corrected |
| C-8 item 33 / 37 / 38 | "實務範式", "重算了所有東西", "一個數字都沒改" | "範式" → "記錄一種作法"; "所有東西" → "結果表"; "沒改" retained with the P2 additions now spelled out in B6. | report §5; interview_script B6 | corrected |
| C-8 item 40 | Script "規則在看到數字前寫進版本控制" (Amendment 1 exception) | Retained for the 3-minute opening (true for model metrics); the exception is in C-Q&A and every written document. | — | disputed-with-reason |
| C-9 | `verify_space.py` needs undocumented `--space`; `train.py` no offline flag | `--space` documented in the docstring; `train.py` is completed-study code (README already states `WANDB_MODE=offline`). | scripts/verify_space.py; README | corrected / disclosed |
| C-10 / C-11 §10 | Hardcoded paths, duplicated frozen-path helpers, no determinism flags, missing Agg, unpinned openpyxl/open_clip_torch, `resolve_weight` no offline switch | Completed-study code — unchanged by rule; README/report disclose the reproducibility consequences. `deploy_hf.py` is the one code change (below). | README; report §2.9 | disclosed |
| C-10 / C-11 §10 | `deploy_hf.py` rewrites `src/inference.py` in place (author's earlier "no-op" rebuttal held DISPUTED-INCORRECTLY) | Accepted. Rewritten: reads `src/` only, patches the `deploy/` COPY when the account differs, asserts byte identity otherwise; `--space-only` / `--no-push` added. | scripts/deploy_hf.py | corrected |
| C-PROV §7 / C-11 new 2 / Zenodo section | The v1.0 deposit's embedded PROVENANCE §7 / README / CITATION.cff name 1f196a9, fcd270a5…, "DOI pending" | Cannot be edited in place. Recorded as a known defect in PROVENANCE §7.2, README Citation, CITATION.cff identifiers; erratum sentence prepared for the v1.0 record description; v1.1 archives carry `.git-commit` (export-subst) and no self-checksum. | PROVENANCE §7; docs/zenodo_v1.1_new_version.md; .gitattributes; .git-commit | corrected (design) / disclosed (v1.0) |
| C-PROV §2 | `v1.0-audited` row lacked the tag time | Row now carries 13:05:16; rows for the re-audit archive, P8' and v1.1 added. | PROVENANCE §2 | corrected |
| C-sweep / C-11 new 5 | `consistency_check.md` "0 UNVERIFIED" is token coincidence | Sweep rewritten: matches require a shared context tag (cohort / metric / quantity kind) and no cohort conflict; CSV cells matched by column name + row strings; CHECKSUMS.txt dropped as a source; static "0 UNVERIFIED" prose removed; regenerated (see "Consistency sweep" below). | scripts/consistency_sweep.py; docs/consistency_check.md | corrected |
| C-11 new 18 | `docs/readme_skeleton.md` (tracked draft) still carries "~9.7M", "safe direction", "≥ 0.92" | Removed from the repository (`git rm`); superseded by README since P1. | — | corrected |
| C-11 §9 9.4 / 9.5 | Training nondeterminism; silent HF download | Disclosed (README); unchanged by rule. | README | disclosed |
| C-exec 1 | Published record contradicts itself on provenance | Report synced; README fixed; Zenodo defect recorded and erratum prepared. | above | corrected / disclosed |
| C-exec 2 | Self-verification claims stronger than the verification performed | Recomputation claim scoped; sweep made context-aware and its limits stated in its own output; RESULTS.md:483 erratum; BiomedCLIP JSON discrepancy recorded; §1 literature claims labelled; "gain is threshold placement" labelled. | above | corrected |
| C-exec 3 | Pre-registration self-certified; deployed/derived artifacts drifted | Permanent part stated; Space re-pushed; docstring fixed; checkpoint metadata and `v2_loco.py` disclosed; hygiene list disclosed; `deploy_hf.py` fixed. | above | corrected / disclosed / permanent |

## Table 2 — docs/AUDIT2_astra_2026-09-11.md

| # | Audit item (section) | Handling | Where | Status |
|---|---|---|---|---|
| A-1 | Fold-5 reuse by checkpoints 1–4 (expected); validation-label reuse (best-epoch selection on the reported fold); stale checkpoint fold metadata; "patient-level splits only" false for v2 | Selection bias: disclosed and now labelled a sensitivity, not an unbiased correction. Checkpoint metadata: frozen, disclosed. v2: "image-level 15% validation slices" stated in README, CLAUDE.md rule 1, report §2.7/§4.5/§5, summary, plan.md standing decisions. | README; CLAUDE.md; report; RESULTS.md Errata | corrected (wording) / disclosed |
| A-1 | README "8 tests" | 19 tests, split by data requirement. | README | corrected |
| A-2.1 **N1** | "5-checkpoint ensemble pooled OOF 0.9254" predictor does not exist; T/threshold fitted on the single-held-out distribution and applied to the ensemble | Relabelled everywhere ("pooled OOF — one held-out checkpoint per image + hflip TTA + T; the deployed ensemble has no unbiased internal estimate") and the predictor-change stated explicitly; new POST-HOC 5 separates predictor vs cohort components (cohort ≫ predictor; fold 5 is the least specific member on all four cohorts; the ensemble sits at or below the least specific member on BrEaST). | README rows (b), paragraph; report abstract/§2.2/§3.1/§3.2/§3.3/§4.1/§4.5; summary; script; plan.md; CLAUDE.md rule 9; RESULTS.md POST-HOC 5 + Errata (lines 126, 165–167, 190–191, 221; JSON `probability_space` and ROC title mislabel noted, frozen) | corrected + new analysis |
| A-2.2 | "Fixed epoch" chosen post hoc; 0.0111 not an unbiased optimism estimate; **fold ranking DOES change** (1>2>5>3>4 → 1>2>3>5>4); "slightly understated" not established | All corrected: "sensitivity of the reported CV AUC to epoch selection"; ranking sentence errata'd with both orders; "understated" withdrawn. | README (a′); report §3.1, §4.5; plan.md; RESULTS.md Errata (lines 1126, 1128–1130, 1102–1130) | corrected |
| A-2.2 / A-7 | P2 "nested" threshold analysis is a leave-fold-out threshold sensitivity on a fixed OOF artifact, not a nested outer-model evaluation | Renamed in every document and in the script's docstring and printed header (file/CSV names kept for continuity, stated); README row (b′) and paragraph explain the dependency. | README; report abstract/§3.2/§4.5; summary; script; plan.md; src/posthoc_nested_threshold.py; RESULTS.md Errata (lines 1132–1161) | corrected |
| A-2.3 | 76 false negatives; "not missed cancers" wording; PPV/NPV | Corrected everywhere (see C-8 item 3). | README; report; summary; script; RESULTS.md Errata (lines 195–198) | corrected |
| A-2.4 | Stray 'c' is reader2 (protocol says reader1) | Already in RESULTS.md:240–244; now in the protocol appendix 7. | protocol appendix | disclosed |
| A-2.5 item 1 | Highest-threshold rule not literally implemented (roc_curve drop_intermediate=True); exhaustive 0.26878 vs 0.26832; oracles; 2432/11000 M1 draws; k\* unchanged | Reproduced exactly (2,432 = 259/372/604/1,197; max |Δspec| 0.393/0.213/0.272/0.521; k\* 20/20/10/10 both rules) by `src/posthoc_threshold_rule.py`; docstrings of `pick_threshold.py` fixed; JSON unchanged; frozen 0.26832 retained; stated in report §2.2, README limitations, protocol appendix 1, RESULTS.md POST-HOC 6 + Errata. | as listed | corrected (docstring/docs) + new analysis; frozen value retained by rule |
| A-2.5 item 2 | Platt rank reversal omitted from "structural equivalence" (12/4/3/16 negative slopes at k=10) | Reproduced (plus 1 at SYSUCC k=20); equivalence restricted to positive-slope transforms everywhere. | report abstract/§3.7/§4.1; summary; script; plan.md; RESULTS.md POST-HOC 6c + Errata (lines 535–542); protocol appendix 3 | corrected + new analysis |
| A-2.5 item 3 | Temperature differences are 0.09 / 0.14, not 0.03–0.04 | Report §3.8 rewritten (T differences 0.09/0.14; spec differences ≤ 0.03/0.04); RESULTS.md lines 584–585/595 errata'd. | report §3.8; RESULTS.md Errata | corrected |
| A-2.5 item 4 | 0.235 (not 0.236); 0.567 (not 0.568) | README, report §3.2/§3.7, RESULTS.md Errata (lines 1157, 467, 487). | as listed | corrected |
| A-2.5 item 5 | "≤ 7×10⁻³" / "≤ 0.007" exceeded by 0.007227 | "max 0.0072, median 0.0008" in README, app About, deploy README, report §2.8/§4.5, summary, script. | as listed | corrected |
| A-2.5 item 6 | Universal "every number recomputes" false (cdrop pooled OOF, latency, CAM tallies, wandb inputs) | Scoped; exceptions named. | README; report §2.9; CITATION.cff; .zenodo.json | corrected |
| A-2.6 | Consistency sweep matches tokens, not claims; static "0 UNVERIFIED" | Rewritten (context tags, per-cell CSV matching, cohort conflict rule, no static prose); its own header states what it cannot check. | scripts/consistency_sweep.py; docs/consistency_check.md | corrected |
| A-3 / PROVENANCE table | "Each amendment committed alone" false for Amendment 1; "--confirm single-run guard" incorrect; prose order freeze/protocol wrong; "only public artifact" omits the Space; fold1 upload omitted; "about 13 hours" → 13–16 h; §7 "this commit" stale; archive self-description | All rewritten in PROVENANCE §1, §2, §4, §5, §7; README and report §2.10 aligned. | PROVENANCE; README; report §2.10 | corrected |
| A-3 | No public pre-freeze anchor; "never overwritten in any uncommitted state" unprovable | Permanent; stated in PROVENANCE §4 (added sentence on uncommitted states). | PROVENANCE §4 | permanent |
| A-4 (l) | IQR / 5–95 band registered; 2.5/97.5 delivered | Recorded. | protocol appendix 2; report §2.7 | disclosed |
| A-4 (t) | Patience 7 not prespecified; assert-based guards vanish under `python -O` | Patience disclosed (already); the `-O` point is noted here: the v2 guards are assertions in completed-study code and are not changed. The new v1 guard raises SystemExit, not an assertion. | protocol appendix 5; src/external_val.py | disclosed / corrected (v1 guard) |
| A-4 **enforcement gap** | `external_val.py` overwrites existing CSVs; `--confirm` is not a single-run guard | `refuse_existing_output()` added: any cohort whose CSV exists is refused before any inference unless `--overwrite`; checked for all selected cohorts first; 4 tests; the v1 files are untouched and the guard is labelled a post-hoc addition (did not exist at run time). Demonstrated: `--dataset breast --confirm` → REFUSED, exit 1. | src/external_val.py; tests/test_external_val_guard.py; README; PROVENANCE §4; protocol appendix 9; report §2.5; CLAUDE.md rule 8 | corrected (tooling) / disclosed (history) |
| A-4 (u) | "patient-level folds, data/splits" inaccurate when written | Recorded. | protocol appendix 10 | disclosed |
| A-5 | "35% duplicates" imprecise; unqualified "no near-duplicate" | Scoped (507 dup + 39 conflicts; visually confirmed cross-set at d ≤ 8; finite screen). | README; report §2.4; summary; script; protocol appendix 8 | corrected |
| A-6 | Two preprocessing routes (real); CPU self-check flags 155 vs 156; live tree behind deploy | Preprocessing: stated (unchanged). CPU decision difference: stated in README Reproduce and report §4.5. Live tree: re-pushed (see C-6(e)). | README; report §4.5, §3.12 | corrected / disclosed |
| A-6 | README strict ".007" bound | "max 0.0072". | README | corrected |
| A-7 | Bootstrap units fine; CI conditional on selected threshold; P2 rows not a repair; "統計平手" unsupported; "3/4 優於 margin" point estimates; "位置而非尺度" not established; "no missed cancers" false; k_reliable has no sensitivity floor | "統計平手" → "CV means differ by 0.0002; no equivalence test"; "位置而非尺度" → consistent with a location shift, scale/shape/prevalence not excluded; the rest as above. k_reliable's lack of a sensitivity floor is stated in RESULTS.md's own definition (2.5th-pct recovery only) and not claimed as a clinical reliability guarantee anywhere. | report §3.1, §4.1; RESULTS.md Errata (lines 32–33) | corrected / disputed-with-reason (k_reliable) |
| A-8 O1 | README caption over-generalises "by nature" and "effect is small" | Rewritten: "in the examples inspected"; the probe is malignant-only with a heavy tail and "bounds rather than excludes". | README | corrected |
| A-8 O2, O3, O17, O26, O27, O35 | Ensemble label; "nested" over-claim; .236 | See N1 / A-2.2 / A-2.5 item 4. | as listed | corrected |
| A-8 O4, O18, O33, O37 | "not missed cancers"; "最有支撐"; "可直接落地" | Corrected (76 FNs; "最直接支撐之建議"; "研究提案,尚無本地臨床驗證"). | README; report | corrected |
| A-8 O5 | "Patient-level splits only" | Scoped (v1/BUS-BRA; v2 image-level). | README; CLAUDE.md | corrected |
| A-8 O6, O16 | pHash scope | See A-5. | — | corrected |
| A-8 O7 | Universal CSV claim; ".007" | See A-2.5 items 5–6. | — | corrected |
| A-8 O8 | "All checkpoints" broader than the release | "the release artifacts …; earlier development checkpoints are not published". | README | corrected |
| A-8 O9 | Byte-identical self-check needs device qualification; 8 tests | Both stated. | README | corrected |
| A-8 O10 | "Pre-rewrite hashes do not map" misstates the evidence | "differ from current ones; the complete 30-commit mapping is in PROVENANCE §3, verifiable only from the author's copy". | README; report §2.10 | corrected |
| A-8 O11, O24 | "only public artifact" omits the Space | Both artifacts named with timestamps. | README; report §2.10; PROVENANCE §1 | corrected |
| A-8 O12 | "DOI pending" stale | Removed. | README | corrected |
| A-8 O13 | "none are redistributed" ambiguous | "the raw datasets are not redistributed (what is: …)". | README | corrected |
| A-8 O14 | Report's universal sourcing assurance | Revision note now lists what the second round corrected and labels §1 as impressions. | report revision note | corrected |
| A-8 O15, O19, O20 | Uncited clinical/literature claims | See C-8 items 15–17. | report §1 | corrected |
| A-8 O21, O23, O36 | Stale provenance sentences | See C-2 drift 1. | report | corrected |
| A-8 O22 | Highest-threshold assertion | See A-2.5 item 1. | report §2.2 | corrected |
| A-8 O25 | "統計平手"; T fitted on single-model OOF; ranking | See A-7 / N1 / A-2.2. | report §3.1 | corrected |
| A-8 O28 | 7/8 and 3/4 CAM tallies are selected-image qualitative tallies | Labelled as such. | report §3.6; RESULTS.md Errata | corrected |
| A-8 O29 | 0.568; negative Platt slopes | See A-2.5 items 2, 4. | report §3.7 | corrected |
| A-8 O30 | T difference arithmetic; higher thresholds raise spec by construction | Both stated. | report §3.8 | corrected |
| A-8 O31, O32 | "threshold placement" mechanism unproven; "位置而非尺度" | Labelled untested hypotheses. | report §3.11, §4.1 | corrected |
| A-8 O34 | "signal real", "3/4 優於" | Retained as point-estimate language with BrEaST overlap stated in §3.9 (Claude §7 judged this acceptable); "信號真實" stays inside the labelled post-hoc interpretation. | report §4.3 | disputed-with-reason |
| A-8 reader source | "readers had the full examination" contradicts HoVer-Trans §V-C | Replaced by: the release does not document whether the reader columns come from clinical PACS reads or the paper's single-image reader study (Mo et al. §IV-C, §V-C); if the latter, model and readers saw equivalent information. | report §4.2, §4.5; interview_script B3 | corrected |
| A-9 | Clean-build not verified by this auditor (read-only constraint) | The other re-audit built a fresh venv (181 packages) and passed; nothing to change. `openpyxl`/`open_clip_torch` remain unpinned (completed-study environment file; disclosed). | — | disclosed |
| A-10 | Threshold extraction "plotting simplification" (major) | Documented as the frozen routine; exhaustive counterfactual run; not changed (frozen). | src/pick_threshold.py; RESULTS.md POST-HOC 6 | corrected (docs) / frozen by rule |
| A-10 | v1 outputs overwritable (major) | Guard added. | src/external_val.py | corrected |
| A-10 | "Nested" naming (major interpretation risk) | Docstring and every document renamed; file name kept and explained. | src/posthoc_nested_threshold.py | corrected |
| A-10 | Platt negative slopes, suppressed ConvergenceWarning, parameters not persisted | Slopes counted and persisted (`reports/posthoc_platt_slopes.csv`); `v2_recalib_curve.py` is completed-study code and unchanged. | reports/posthoc_platt_slopes.csv | corrected (analysis) / disclosed (code) |
| A-10 | Unconstrained T fitter / broad RuntimeError catch; `data.py` cardinality guard; `weights_only=False`; loader has no checksum enforcement; `-O` assertions; mains that append/overwrite; Agg; cwd change | Completed-study code — unchanged by rule; noted here. | — | disclosed |
| A-10 | Deploy helper mutates frozen source for a different owner | Fixed (see C-10). | scripts/deploy_hf.py | corrected |
| A-11.7 | "P2 resolved bias" overclaim; "P6 semantic consistency" open; README caption | All addressed above. | — | corrected |
| A-exec 1–3 | Wrong predictor label / overstated correction; rule not implemented literally + no enforcement; factual/mathematical errors the sweep cannot detect | Addressed item by item above; the two new analyses and the guard are the substantive additions. | — | corrected / disclosed |

## Consistency sweep (regenerated with context-aware matching)

`python scripts/consistency_sweep.py` after all edits: 165 shared values —
155 verified with a shared context tag, 10 derived, 0 weak, 0 UNVERIFIED.
The header of `docs/consistency_check.md` states what the check does and
does not establish; the claim-level checks are the two tables above and
`docs/AUDIT_RESPONSE_2026-09-06.md`. Four tokens were UNVERIFIED on the
first context-aware run and turned out to be matcher gaps, not document
errors (multi-cohort series lines, "measurements" missing from the
latency tag, talk timings not skipped); two more appeared after this
phase's own edits (the 561 false-positive sum and the CPU self-check
count 155, which existed only in the Astra audit) and were given a
derivation entry and a RESULTS.md errata line respectively; the pytest
count is excluded as a repository fact. The rules were extended and the
sweep re-run.

## Verification performed in this phase (actual output)

- `python -m pytest -q tests/` → 19 passed (15 previous + 4 new guard tests).
- `python src/external_val.py --self-check` → 0.9234433158791243, 156/383, PASSED; stdout byte-identical to `reports/external_selfcheck.txt` (diff exit 0).
- `python src/external_val.py --dataset breast --confirm` → `REFUSED: reports/external_breast_preds.csv already exists …`, exit 1 (no inference started).
- `python scripts/deploy_hf.py --no-push` → deploy/ rebuilt; `cmp` src/inference.py deploy/inference.py and app/app.py deploy/app.py identical.
- `python scripts/deploy_hf.py --space-only` → Space commit 7d41f22906e52e53a4e7acda3e403e20bc969e9a; `python scripts/verify_space.py --space happytommy/breast-us-cad` → four examples max |Δ| 1.07e-07 (MATCH AT PLATFORM FLOOR), warm median 6.27 s (min 5.86 / max 7.72, n=6).
- `python src/posthoc_predictor_shift.py` → all asserts OK (pooled OOF = operating_point.json; ensemble-from-members = external-v1 decisions on 2454/2454; fold-5 single = v2_loco_summary.csv).
- `python src/posthoc_threshold_rule.py` → 0.26878 vs 0.26832 (1 image); oracles BrEaST 1 / BUSI 0 / GDPH 0 / SYSUCC 4 images; M1 2432/11000 changed, k\* 20/20/10/10 unchanged; Platt negative slopes 12/4/3/16 at k=10, 1 at SYSUCC k=20.
- `git diff 2cbe1da -- RESULTS.md | grep '^-' | grep -v '^---' | wc -l` → 0; same for `data/external_protocol.md` vs 4fc3fb2 → 0.
- `python scripts/consistency_sweep.py` → 165 shared, 155 verified, 10 derived, 0 weak, 0 UNVERIFIED (context-aware).

## Not done in this phase, and why

- No frozen JSON/checkpoint edited (BiomedCLIP one-image discrepancy, operating-point `probability_space` label, checkpoint fold metadata): frozen artifacts, by rule; recorded instead.
- No completed-study code changed (`v2_loco.py` checkpoint payload, `v2_recalib_curve.py` slope constraint/warnings, `train.py` offline flag, hardcoded paths, duplicated helpers, Agg): by rule; disclosed.
- No retraining, no nested outer-model evaluation, no new external inference: by rule.
- Zenodo v1.1 not published; v1.0 description erratum not yet applied: STOP by instruction (`docs/zenodo_v1.1_new_version.md`).
- OSF registration: a forward commitment for future protocols (PROVENANCE §6), not achievable retroactively.
