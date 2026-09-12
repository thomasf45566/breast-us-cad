# Independent adversarial RE-AUDIT — breast-us-cad (fresh clone, HEAD 4794ad6)

Date: 2026-09-11. Auditor: independent, no prior context beyond the two archived
documents the task names (docs/AUDIT_2026-09-06.md, docs/AUDIT_RESPONSE_2026-09-06.md),
which were read first and then treated as claims to re-verify, not as facts.
Read-only: nothing in this clone, in the author's working copy, on GitHub, HF or
Zenodo was modified; nothing was committed, trained, or scored on new data.

**Environment actually used.** `$DATA_ROOT=/Users/thomaswang/projects/breast-us-cad/data`
and `$MODELS_ROOT=/Users/thomaswang/projects/breast-us-cad/models` (the author's
working copy, at the same HEAD 4794ad6). This clone carries no `data/raw`, no
`models/*.pt` and no `wandb/`. I built an overlay in the session scratchpad
(rsync of the clone without `.git`; symlinks `data/raw/* -> $DATA_ROOT/raw/*`,
`models/*.pt -> $MODELS_ROOT/*.pt`, `models/pretrained`, and — for one post-hoc
script only — the author's git-ignored `wandb/`), and a fresh `uv venv --python 3.12`
+ `uv pip install -r requirements.txt` (CPython 3.12.13; 181 packages; torch 2.13.0,
timm 1.0.28, opencv-python-headless 5.0.0.93, albumentations 2.0.8, scikit-learn
1.9.0, imagehash 4.3.2, numpy 2.5.2, pandas 3.0.5; MPS available). Every command
below ran in that overlay unless a path says otherwise; read-only `git` commands
also ran in the clone and in the author's copy. Sections 1, 2, 4–8 and 10 were
executed by parallel worker processes of this auditor in the same overlay; their
scripts and full logs (`sec1_10_*.log`, `sec2_recompute.py/.log`,
`sec5_phash_repro.py/.log`, `sec6_ast.py`, `sec6_selfcheck.out`,
`sec6_verify_space.log`, `prerewrite_map.txt`, `zenodo.json`, `gh.json`,
`v1.0-audited.zip`) are in the scratchpad.

Model forward passes performed: `src/external_val.py --self-check` (BUS-BRA fold 5,
permitted) and the four bundled example images through the live Space. No other
inference, no training.

Severity scale (as in the first audit): **critical** = a headline claim is wrong
or leakage exists; **major** = a claim as written is unsupported/misleading or
reproducibility/provenance is broken; **minor** = imprecision, hygiene.

Section order: 1–10 as specified, then the three extra checks the task added
(PROVENANCE.md claim by claim; Zenodo record vs tag and archive; the consistency
sweep re-run), then §11 (resolution of every 2026-09-06 item), then the executive
summary.

---

## 1. Split integrity (re-audit, independent)

Environment: overlay `$S` (rsync of the clone HEAD 4794ad6 without .git; `data/raw/*` and `models/*.pt` are read-only symlinks into the author's copy; `wandb/` symlinked read-only for the post-hoc script), fresh `uv venv --python 3.12` from `requirements.txt`. Logs: `sec1_10_splits.log`, `sec1_10_pytest.log`, `sec1_10_ckpt_meta.log`, `sec1_10_wandb.log`, `sec1_10_posthoc_epoch.log` in the scratchpad.

### 1(a) Official partition, own pandas code (not the repo's test)

```
sha256 data/splits/busbra_official_5fold.csv : 8bed2d2efdf6ea370ab82a3cbda4c213f498274a68b434828ceeb636ce43a68d
sha256 data/raw/busbra/5-fold-cv.csv          : 8bed2d2efdf6ea370ab82a3cbda4c213f498274a68b434828ceeb636ce43a68d
byte-identical: True
bus_data.csv 1875 rows merged 1:1 with fold file -> 1875 rows, 1064 Case ids
patients spanning >1 fold: 0
fold sizes (images): {1: 376, 2: 385, 3: 366, 4: 365, 5: 383}
patients per fold: {1: 212, 2: 213, 3: 213, 4: 214, 5: 212}
patients with mixed pathology: 0
patient-level pathology: benign 722 / malignant 342 ; image-level: 1268 / 607
Pathology column in fold file == bus_data.csv on 1875/1875
reports/oof_vit_preds.csv            : 1875 rows, 1875 unique ids, fold == official kFold on 1875/1875
reports/v2_biomedclip_oof_preds.csv  : 1875 rows, 1875 unique ids, fold == official kFold on 1875/1875
```
The committed fold file is the official partition (hash and bytes), it is what `data.resolve_fold_file` prefers (`src/data.py:33-52`, raises `ValueError` on hash mismatch), and both saved OOF artifact sets (v1 and v2 BiomedCLIP) were produced with it.

### 1(b) Test suite, fresh venv, all five raw datasets present

```
$ python -m pytest -q tests/
...............                                                          [100%]
15 passed in 36.80s
```
15 = `test_splits.py` 5 (3 original + 2 new fold-file tests) + `test_checkpoint_config.py` 3 + `test_hf_paths.py` 2 + `test_v2_splits.py` 5. Note: README.md:155 still says "`pytest` runs the 8 split tests" — stale (minor).

### 1(c) Code paths that could leak fold 5 / external images into v1 training

- `src/train.py:129-157` (`train_one_fold`) builds loaders only from explicit `train_folds`/`val_folds`; `src/data.py:179-191` (`make_dataloaders`) raises on any fold overlap; `pos_weight` uses train folds only (`train.py:159-161`).
- `src/cross_validate.py:58-66`: `train_folds = [f for f in ALL_FOLDS if f != k]`, `val_folds=[k]`; `eval_fold` (`:24-41`) scores only `val_folds`.
- Every v1 config (`configs/{baseline,convnext,vit,vit_cdrop,seg,v2_biomedclip}.yaml`) has `data.root: data/raw/busbra`; `build_master_df` reads only `bus_data.csv` + the fold file under that root.
- `grep -rn 'breast_poland|busi|gdph_sysucc|busi_whu|external' src scripts app deploy tests` (excluding `external_val, dedup_busi, phash_*, birads_comparison, v2_*, explain_breast_fp, plot_prob_shift, posthoc_*`): remaining hits are docstrings/About text, `src/explain.py` (reads saved `reports/external_*_preds.csv` for the FP gallery — post-hoc, no training), `scripts/*` (deploy text). No v1 training or CV script imports or opens an external directory.
- `src/train_seg.py` uses `make_seg_dataloaders` with the same overlap guard (`data.py:212-223`).

### 1(d) Checkpoint provenance

Embedded config (`torch.load(..., weights_only=False, map_location="cpu")`):

| checkpoint | stored train_folds / val_folds | epoch | val_auc |
|---|---|---|---|
| cv_vit_fold1..5.pt | **[1,2,3,4] / [5] in all five** | 29 / 10 / 18 / 16 / 19 | 0.95256 / 0.94114 / 0.91929 / 0.91320 / 0.92720 |
| v2_biomedclip_fold1..5.pt | **[1,2,3,4] / [5] in all five** (init_state_dict models/pretrained/biomedclip_vitb16_timm.pt) | 24 / 13 / 23 / 26 / 22 | 0.94385 / 0.92286 / 0.91189 / 0.88348 / 0.92303 |
| v2_loco_{breast,busi,gdph,sysucc}.pt | none (no fold keys); `hold_out` key present | 4 / 17 / 4 / 2 | 0.93790 / 0.95055 / 0.92765 / 0.92865 |
| cv_convnext_fold1.pt, cv_vit_cdrop_fold1.pt | [1,2,3,4] / [5] | 21 / 15 | 0.95414 / 0.95582 |

So the misleading metadata the 2026-09-06 audit reported is **unchanged in every existing checkpoint** (expected: the response says existing checkpoints were not rewritten). The P2 fix `train.checkpoint_payload` (`src/train.py:94-114`) **is** wired into the real save path (`train.py:216-226`); `tests/test_checkpoint_config.py` exercises only the helper, not a training run (no test asserts that a checkpoint written by `train_one_fold` carries the resolved folds). **`src/v2_loco.py:138-140` was not updated and still saves the raw `cfg`** — harmless here only because `configs/v2_loco.yaml` has no `train_folds` key and the `hold_out` field is stored.

wandb (46 local run dirs read; `sec1_10_wandb.log`). The five `--config configs/vit.yaml --prefix cv_vit` runs (pre-rewrite commit c9adedd, 2026-08-28 00:44–08:42Z) record `train_folds/val_folds` = [2,3,4,5]/[1], [1,3,4,5]/[2], [1,2,4,5]/[3], [1,2,3,5]/[4], [1,2,3,4]/[5] with `best_val_auc` 0.9525622821737446 / 0.9411384615384615 / 0.9192920054200542 / 0.9131994261119081 / 0.9271970222698883 = `reports/cv_vit_summary.csv` `auc` column to all digits, and = the `val_auc` stored in each checkpoint. Same pattern for cv_convnext (1a02bd6), cv_vit_cdrop (82bd016) and v2 BiomedCLIP (b15a981/48f2475) runs. Fold k was therefore excluded from checkpoint k's training for every CV family.

Two undocumented facts from the wandb dirs (not leakage): (i) LOCO `--hold-out gdph --train` was launched twice at commit eeca3fc: `run-20260903_195053-k896krp6` (no summary; `output.log` ends at epoch 3, val_auc 0.8948) and `run-20260903_200947-3jn4cicy` 19 min later, which produced the checkpoint. RESULTS.md/plan.md do not mention the aborted first attempt; the held-out cohort is untouched during `--train`, so this is a restart, not a peek — but it is an unrecorded second training attempt on the same pool. (ii) Two aborted baseline CV runs (n2dhyt0w, ctrl1a3h) — irrelevant to frozen artifacts. LOCO wandb `epoch` (11/24/11/9) = ckpt best epoch (4/17/4/2) + patience 7, consistent with RESULTS.md:892-960.

### 1(e) v2 LOCO split integrity (`src/v2_data.py`)

- BUS-BRA: fold 5 → val, folds 1–4 → train (`:83-92`, `BUSBRA_VAL_FOLD = 5`), patient-level via the official file.
- External cohorts: keep-list rows sorted by `image_path`, 15 % val permuted with `np.random.default_rng([42, cohort_index])` (`:51-71`), independent of `hold_out` → identical slices across runs (asserted by `tests/test_v2_splits.py::test_external_val_slices_identical_across_runs`).
- Held-out cohort: excluded from **both** train and val (`:93` builds parts only from `c != hold_out`; `:96-101` asserts no held-out `image_path` anywhere in the pool). `v2_loco.py --evaluate` refuses if the preds CSV exists (`:211-213`) and requires `--confirm` (`:319-321`).
- The 85/15 slices are **not stored** in any committed artifact (no `split` column in `reports/v2_members_*.csv` or `v2_loco_*_preds.csv`); they are reproducible only by re-running the seeded code against the keep-lists. Image-level (not patient-level) for BUSI/GDPH/SYSUCC — declared in Amendment 4 (t) and `v2_data.py:8-10`.

### 1(f) Epoch selection

`train.py:214-226` saves at `val_auc > best_auc` on the validation fold; `cross_validate.py:72` scores that checkpoint on the same fold. Re-ran `python src/posthoc_epoch_selection.py` in the overlay (reads `wandb/run-*/run-*.wandb` datastores, writes `reports/posthoc_epoch_selection.csv`):
```
assert: best-epoch AUC and best_epoch reproduce reports/cv_vit_summary.csv (max |Δ| 1.1e-16) — OK
fixed epoch = median of the five best epochs [10, 16, 18, 19, 29] = 18
mean ± SD: best 0.9307 ± 0.0161 | epoch 30 0.9235 ± 0.0181 | fixed 0.9195 ± 0.0164 | optimism 0.0111 ± 0.0073
optimism best − final: 0.0071 ± 0.0059
diff vs committed reports/posthoc_epoch_selection.csv: IDENTICAL
```
The script reads only the local wandb datastores, so this quantification is reproducible only from the author's disk (wandb/ is git-ignored) — stated in the script docstring, not in README.

**Verdict §1: PASS on patient-level split integrity and no v1 leakage** (0 patients across folds; official file committed, hash-verified and byte-identical to the raw copy; both OOF artifact sets on the official folds; wandb configs prove fold k excluded from checkpoint k; 15/15 tests pass in a fresh env). **CONCERN — minor:** (i) every existing checkpoint still embeds `train_folds=[1,2,3,4]/val_folds=[5]` and `v2_loco.py` never adopted `checkpoint_payload`; (ii) the LOCO-gdph aborted first training attempt is unrecorded; (iii) LOCO 85/15 slices are not persisted; (iv) README:155 "8 split tests" is stale (15). Evidence I would have expected for a FAIL — a patient with >1 fold, an OOF fold column diverging from `kFold`, a wandb config showing fold k in train for checkpoint k, or a v1 script importing an external directory — was searched for and not found.

---

## 2. Provenance of every headline number (re-audit, 2026-09-11)

Environment: overlay of the clone at HEAD 4794ad6 (scratchpad), fresh `uv venv --python 3.12` +
`uv pip install -r requirements.txt` (torch 2.13.0, sklearn 1.9.0, openpyxl 3.1.5, wandb 0.28.2),
`data/raw` and `models/*.pt` read-only symlinks; `wandb/` symlinked read-only from the author's
working copy for POST-HOC 1 only. No inference, no training; everything below is recomputed
from committed CSV/JSON artifacts with the repo's own functions (`external_val.patient_bootstrap`,
`pick_threshold.*`, `calibrate.*`, `v2_disagreement.*`, `v2_recalib_curve.*`, `v2_loco_report.*`,
`birads_comparison.*`, `posthoc_*`), bootstrap 2000 × seed 42 as protocol (d). Script and full
log: `sec2_recompute.py`, `sec2_recompute.log` (scratchpad). "=" = identical to the stated
precision; "= (exact)" = identical to all printed digits / bitwise.

| Number (where stated) | Producing script → artifact | Recomputed | Match |
|---|---|---|---|
| CV AUC 0.9307 ± 0.0161, folds 0.9526/0.9411/0.9193/0.9132/0.9272, best epochs 29/10/18/16/19 (README:31, report abstract:22 & §2.2, RESULTS.md:9,23) | `cross_validate.py` → `reports/cv_vit_summary.csv` | mean 0.9307, sd(ddof=1) 0.0161; per-fold values identical; best_epoch identical. Also convnext 0.9304 ± 0.0173, effb0 0.8910 ± 0.0271, cdrop 0.9329 ± 0.0199, BiomedCLIP 0.9170 ± 0.0220 | = |
| (a′) fixed-epoch 0.9195 ± 0.0164; optimism 0.011 ± 0.007 (0.0111 ± 0.0073); best−final 0.0071 ± 0.0059; fixed epoch 18 (README:32,44; report §3.1; RESULTS.md POST-HOC 1) | `posthoc_epoch_selection.py` → `reports/posthoc_epoch_selection.csv`; wandb datastores run-20260828_{084412,100807,113529,151003,164240} | CSV stats identical; **wandb histories re-read independently**: fixed epoch 18, max\|Δ\| vs CSV 1.1e-16, best epochs [29,10,18,16,19], 30 epochs each, max val_auc == `cv_vit_summary.csv` auc to 1e-12 | = (exact) |
| Pooled OOF AUC plain 0.9231 / TTA 0.9254; per-fold TTA AUCs; TTA mean 0.9323 ± 0.0162 (RESULTS.md:95-101, report §3.2) | `tta_eval.py` → `reports/oof_vit_preds.csv`, `tta_vit_summary.csv` | 0.9230942630405521 / 0.9254283620640372; per-fold plain/TTA reproduce to ≤ 1 ulp (e.g. fold 5 TTA 0.9234433158791243 vs CSV …244) | = |
| T = 2.3644; ECE 0.0716 → 0.0401; NLL 0.4411 → 0.3237; AUC unchanged (report §3.2, RESULTS.md:113-125) | `calibrate.py` (LBFGS) → `models/calibration.json` | T refit 2.364384651184082 (Δ = 0.0 vs JSON); ECE 0.07156697875330724 → 0.04008772073058075; NLL 0.44110173028383276 → 0.32365252038822134; AUC 0.9254283620640372 — all bitwise equal to JSON | = (exact) |
| Threshold 0.2683; sens 0.9028 (0.874–0.928); spec 0.7713 (0.745–0.798); PPV 0.654; NPV 0.943; 548/290/59/978 (README:33,46; report abstract, §3.2; RESULTS.md:134-148) | `pick_threshold.py` → `models/operating_point.json` | thr 0.26832120350764344 (== JSON); confusion (548,290,59,978); sens 0.9028006589785832, spec 0.7712933753943217, PPV 0.6539379474940334, NPV 0.9431051108968177; CIs [0.8741595649654773, 0.9278470605623914] / [0.7454055459205599, 0.7983060882247078] — bitwise equal to JSON | = (exact) |
| (b′) nested OP: thresholds 0.236–0.311 (0.2753 ± 0.0354), sens 0.900 ± 0.057 (0.9003 ± 0.0569), spec 0.780 ± 0.132 (0.7802 ± 0.1315), pooled 0.9012/0.7784, sens < 0.90 on 3/5 folds, sens range 0.849–0.976, spec 0.550–0.878 (README:34,48-49; report §3.2; RESULTS.md POST-HOC 2) | `posthoc_nested_threshold.py` → `reports/posthoc_nested_threshold.csv` | **Re-derived from `oof_vit_preds.csv` alone** (thr picked on 4 folds with `pick_operating_point`, evaluated on the 5th): thresholds 0.244982/0.235452/0.310775/0.310775/0.274332, confusions identical to CSV; mean ± sd 0.2753 ± 0.0354 / 0.9003 ± 0.0569 / 0.7802 ± 0.1315; pooled 0.9012/0.7784; 3/5 folds < 0.90; sens 0.8487–0.976, spec 0.55–0.878 | = |
| BrEaST AUC 0.8542 (0.8019–0.9024), sens 0.9184 (0.8605–0.9678), spec 0.4091 (0.3333–0.4897), 90/91/8/63 (README:35; report Table 2; RESULTS.md:185,201) | `external_val.py` → `reports/external_breast_preds.csv` | identical; n=252, 252 patient ids, prevalence 0.389, 2000 valid resamples | = |
| BUSI 0.9339 (0.9062–0.9580), 0.9571 (0.9250–0.9868), 0.6296 (0.5603–0.6927), 156/80/7/136 | idem `external_busi_preds.csv` | identical | = |
| GDPH 0.9154 (0.8949–0.9343), 0.9707 (0.9529–0.9866), 0.4529 (0.4060–0.5000), 364/238/11/197 | idem `external_gdph_preds.csv` | identical | = |
| SYSUCC 0.8380 (0.8098–0.8655), 0.9309 (0.9119–0.9489), 0.4740 (0.4169–0.5318), 674/152/50/137; PPV 0.816 / NPV 0.733 | idem `external_sysucc_preds.csv` | identical | = |
| `y_pred == (p_cal ≥ thr)`; `calibrate(y_prob_raw, T) == p_cal` (all 4 external CSVs, 4 BiomedCLIP external CSVs, 4 LOCO CSVs) | — | True on all 12 files; max \|Δ\| 3e-7…3e-5 (float32 CSV round-trip of raw prob through the logit) | = |
| Benign medians 0.0625 (internal) / 0.3061 / 0.1739 / 0.2894 / 0.2869; malignant 0.8533 / 0.7561 / 0.8309 / 0.7698 / 0.6804; frac benign ≥ thr 0.2287 / 0.5909 / 0.3704 / 0.5471 / 0.5260 ("0.06 → 0.17–0.31", "37–59 %") (README/report abstract; RESULTS.md:218-224) | `plot_prob_shift.py` | identical | = |
| BI-RADS readers: GDPH model 0.9707/0.4529, r1 0.9760/0.8943, r2 0.9787/0.5126, agreement 0.7593, κ 0.5149; SYSUCC model 0.9309/0.4740, r1 0.9130/0.6505, r2 0.9931/0.1315, agreement 0.7887, κ 0.2152 (report §1.1, §3.5; RESULTS.md:247-270) | `birads_comparison.py` + `data/raw/gdph_sysucc/BIRADS&FOLD.xlsx` | identical (1:1 join, no unmatched IDs, no invalid reader values on the keep-lists — stray 'c' exclusion vacuous) | = |
| POST-HOC 4 concordance: GDPH r1 34/238 (0.143) vs TN 12/197 (0.061); r2 152/238 (0.639) vs 60/197; SYSUCC r1 81/152 (0.533) vs 20/137; r2 143/152 (0.941) vs 108/137 (report §3.5; project_summary "64 %", "14 %") | `posthoc_reader_concordance.py` + xlsx | identical | = |
| POST-HOC 3 overconfidence: fold-5 malignant 121, TTA in [0.6,0.95] 32 (26.4 %), > 0.95 61, < 0.6 28; plain 27; all folds 607: 116/358/133 and 97 (report §3.1) | `posthoc_overconfidence.py` ← `oof_vit_preds.csv` | identical | = |
| Member persistence "bitwise on all 252/379/810/1013" (RESULTS.md:612-617; report §3.9) | `v2_dump_members.py` → `reports/v2_members_*.csv` | With the frozen reduction order (per-ckpt (orig+flip)/2, then mean over 5, float32) the committed member columns reproduce `y_prob_raw` **bitwise on 252/252, 379/379, 810/810, 1013/1013**. (A naive 10-way float32 mean matches only ~65 % of rows, 1-ulp differences — reduction order matters; the claim as written is correct.) | = (exact) |
| Disagreement AUROCs U_std 0.774 (0.710–0.831) / 0.856 (0.812–0.895) / 0.802 (0.772–0.833) / 0.626 (0.589–0.665); margin 0.664 (0.595–0.729) / 0.754 (0.694–0.809) / 0.717 (0.682–0.752) / 0.747 (0.710–0.784); errors 99/87/249/202 (report §3.9; RESULTS.md:626-631) | `v2_disagreement.py` → `reports/v2_abstention_metrics.csv` | recomputed from members + external preds: identical to 4 dp (U_range too) | = |
| Abstention q=10 %: BrEaST 0.9184→0.9091 / 0.4091→0.4532 ×1.53; BUSI 0.9571→0.9527 / 0.6296→0.7047 ×2.64; GDPH 0.9707→0.9675 / 0.4529→0.5038 ×1.77; SYSUCC 0.9309→0.9226 / 0.4740→0.5150 ×1.14; margin rows (report §3.9) | `v2_abstention_curves.csv` | identical; verdict 0/4 (c2 sens clause fails on all four; SYSUCC also c1, c3) follows mechanically | = |
| k* GDPH 10 / SYSUCC 10 / BrEaST 20 / BUSI 20; k_reliable only GDPH 200 (M1/M2b/M3), not reached elsewhere (BrEaST/BUSI grid to 100, SYSUCC to 200; recovery_p2.5 at max k −0.62/+0.03/+0.60/−0.27); k=10 sens medians 0.830/0.846/0.866/0.909; degenerate 0.8/0.6/0/2.4 %; M2 fit-failed 12/20/20/7 % at k=10, 0 at k ≥ 100; M2b k* 10/20/20/20 (README:60-64; report abstract, §3.7) | `v2_recalib_curve.py` → `v2_recalib_M1_{draws,summary}.csv`, `v2_recalib_methods_{draws,summary}.csv` | k*/k_reliable recomputed with the repo's `k_star`/`k_reliable` — identical; **both summaries re-derived from the per-draw CSVs (11 000 M1 draws, 500 per cell; 44 000 method draws): max \|Δ\| 4.4e-16**; k=10 medians identical | = |
| Oracle thresholds 0.3755/0.3851/0.4484/0.3314 → spec 0.623/0.819/0.775/0.568; refit T 2.2717/1.5804/1.3778/2.2253; internal row reproduces external-v1; GDPH threshold transferred: sens 0.857/0.865/0.800 (report §3.7, §3.8, §4.6) | `v2_cross_site.py` → `reports/v2_cross_site.csv` | thresholds re-picked with `pick_operating_point` on each cohort's full data and T refit with `fit_temperature`: identical to 4 dp; oracle spec 0.6234/0.8194/0.7747/0.5675 | = |
| LOCO: AUC 0.8657 (0.817–0.910) / 0.9371 (0.912–0.959) / 0.9454 (0.930–0.960) / 0.8444 (0.817–0.870); v1-single 0.8467 / 0.9069 / 0.8899 (0.867–0.913) / 0.8319; ΔAUC +0.0190 (−0.0155,+0.0550) / +0.0302 (+0.0074,+0.0550) / +0.0556 (+0.0371,+0.0743) / +0.0126 (−0.0099,+0.0346); Δspec +0.2792 (+0.1961,+0.3618) / +0.2731 (+0.2087,+0.3378) / +0.3011 (+0.2484,+0.3573) / +0.2664 (+0.2122,+0.3206); sens 0.857/0.847/0.960/0.800; spec 0.753/0.880/0.713/0.716; benign medians; thr/T 0.4738/1.2216, 0.4116/2.4800, 0.5410/0.9545, 0.4968/0.8761 (README:65-72; report Table 3) | `v2_loco_report.py` ← `v2_loco_*_preds.csv`, `v2_members_*.csv`, `external_*_preds.csv` | all identical (paired bootstrap re-run); branch A 4/4, branch B 2/4 | = |
| LOCO vs deployed v1 **ensemble** ΔAUC +0.0115 / +0.0033 / +0.0301 / +0.0064, "only 2/4 ≥ +0.01" (README:69-70; report §3.11; RESULTS Errata) | derived from `v2_loco_q1c_table.csv` | +0.0115 / +0.0033 / +0.0301 / +0.0064 (BrEaST, GDPH ≥ 0.01) | = |
| BiomedCLIP: pooled OOF 0.9109; T 2.8645; thr 0.2007; external AUC 0.8441 (0.789–0.894) / 0.9102 (0.875–0.942) / 0.8821 (0.858–0.904) / 0.8308 (0.801–0.858); ΔAUC −0.0101/−0.0237/−0.0333/−0.0072; sens 0.929/0.926/0.968/0.946; spec 0.325/0.602/0.345/0.443; own-OOF benign medians 0.0625/0.0808; shift ratios 0.892252/0.689073/0.961516/0.700163; branch A 0/4, B 1/4; unpaired CIs overlap on all four (report §3.10; RESULTS.md:820-859) | `v2_biomedclip_freeze.py`, `v2_biomedclip_external.py` → `v2_biomedclip_*` | all identical; CI overlap confirmed on 4/4 | = |
| **BiomedCLIP internal in-sample "sens 0.901 / spec 0.718" (report §3.10; RESULTS.md ≈ line 808; `models/v2_biomedclip_operating_point.json` confusion 547/358/60/910)** | `v2_biomedclip_freeze.py` | At the **stored** threshold 0.20070531964302063 applied to the committed `v2_biomedclip_oof_preds.csv`: confusion **546/358/61/910, sens 0.8995** (< 0.90). Re-picking the threshold from the CSV gives 0.20070531598248914 (3.7e-9 lower) and 547/…/60 = 0.9012. Cause: the freeze script picked the threshold on in-memory probabilities and the CSV stores float32-rounded probabilities; one malignant image sits exactly on the boundary. The v1 JSON does **not** have this issue (thr and confusion reproduce exactly). | **≠ (1 image; 0.8995 vs 0.9012)** |
| Resize dispatch: 1421/2454 affected (99/0/309/1013), max Δp_cal 0.0072, median 0.0008, 2 flips case038 (0.2689→0.2678), case200 (0.2695→0.2657) (README:99; report §2.8, §4.5; RESULTS.md:1079-1091) | `v2_resize_impact.py` → `v2_resize_dispatch_impact.csv` | 1421 rows (sysucc 1013, gdph 309, breast 99); max 0.0072268, median 0.00077; flips exactly those two | = |
| Pair count 12,056,505 (within 3,432,028 + cross 8,624,477); SYSUCC 546/1559 = 35.0 %; GDPH 36 removed; keep-lists 379/810/1013; 623 hit rows, cross-set hits exactly 2 (busbra–sysucc, gdph–sysucc) (README:86-88; report abstract, §2.4) | arithmetic; `data/splits/*_clean.csv`; `reports/phash_sweep_hits.csv` | identical | = |
| Report-body extras: occlusion median drop < 0.01 (ViT 0.0047, ConvNeXt 0.0079), ~16 % > 0.2 (0.165 / 0.157); U-Net Dice 0.9016 / IoU 0.8326, median 0.9325 / 0.8736, 280/383 > 0.9; CoarseDropout pooled OOF 0.9139, gate-1 margin 0.0008; CPU latency 40 vs 376 ms; cold start 9.7 s (report §3.1, §3.12) | `occlusion_border15.csv`, `seg_metrics_seg_unet_effb0.csv`, RESULTS.md:30,70-72,414-416 | identical | = |

**Numbers not reproduced to the stated precision (1):**

1. `models/v2_biomedclip_operating_point.json` — sens 0.9012 / confusion tp 547, fn 60 — does not
   reproduce from the committed OOF CSV at the stored threshold (546 / 61, sens 0.8995). Report
   §3.10 and RESULTS.md's Q2b section quote "sens 0.901 / spec 0.718 (in-sample)". Minor: not a
   headline number, the Q2 verdict (NOT CLAIMED) is unaffected, and the external v2 decisions are
   consistent with the stored threshold. But strictly, the v2 frozen threshold does not satisfy its
   own "sens ≥ 0.90" rule on the artifact that documents it (the previous audit noted the 4e-9
   threshold difference but not the resulting count change).

**Numbers/statements that differ between documents (all non-numeric drift except as noted):**

1. **docs/report.md is stale relative to README/PROVENANCE on three provenance facts.** §2.1:44 and
   §2.9:83 say the BUS-BRA fold file "不在版本庫內" / lives only in git-ignored `data/raw/`; §2.9:83 and
   §4.5:172 say the nine v2 checkpoints and the BiomedCLIP export "尚未公開" / "未公開"; §2.10:90 and
   §4.6:175 list the Zenodo/OSF timestamp and the publication of v2 checkpoints + fold file as
   future work. README:106-131, 198-206 and docs/PROVENANCE.md §7 state that all three were done
   on 2026-09-07 (P3/P4: `data/splits/busbra_official_5fold.csv` committed, v2 weights on HF under
   `v2/`, Zenodo DOI 10.5281/zenodo.22630912). The report's own revision note (line 8-12) claims
   every factual sentence traces to committed sources — these four sentences trace to a state
   that no longer holds.
2. **README.md is internally inconsistent about the DOI:** line 206 "CITATION.cff and .zenodo.json
   prepared; Zenodo DOI pending upload" vs line 242-244 (DOI badge and record).
3. `app/app.py` == `deploy/app.py` byte-identical (cmp); About text numbers (0.84–0.93, ≥ 0.918,
   0.77 → 0.41–0.63, ≤ 0.007, 10–20 labels, 0.83–0.91, GDPH k=200) all verify.
4. `docs/project_summary.md`, `docs/interview_script.md`, `plan.md`: every number spot-checked
   (external table, 0.9307/0.9195/0.011, 0.900 ± 0.057 / 0.780 ± 0.132, 3/5 folds, κ 0.22/0.51,
   64 %/14 %, 12,056,505, 35 %, 39 conflicts, 0.331/0.376/0.385/0.448, 6–9 s) matches RESULTS.md
   or the derived value; no numeric disagreement found.
5. RESULTS.md Errata line references are accurate: line 403 does read "1,879 … + the 4 examples",
   line 485 "≥ 0.92 everywhere", line 1058 "held sens ≥ 0.92 on all four", line 416 "6.8 s".

**Verdict: PASS for numeric reproducibility — every headline number in README.md, the report
abstract and RESULTS.md (including the P2 post-hoc section and the Errata) recomputes from
committed artifacts to the stated precision, several of them bitwise, and the POST-HOC epoch
table reproduces from the raw wandb datastores. CONCERN — minor — for (i) the BiomedCLIP
operating-point JSON whose stored sensitivity/confusion is off by one image from what its own
threshold yields on the committed CSV, and (ii) docs/report.md carrying four provenance sentences
(fold file not in repo, v2 checkpoints unpublished, Zenodo/OSF as future work) that README and
PROVENANCE.md contradict, plus README:206 "DOI pending" beside the DOI.**

Commands run (from the overlay root, venv active):
```
python ../sec2_recompute.py                       # blocks A–M, log ../sec2_recompute.log
python - <<EOF ... posthoc_epoch_selection.find_cv_vit_runs/read_history/tabulate ... EOF   # wandb re-read
python - <<EOF ... member reduction-order variants (torch float32) ... EOF
python - <<EOF ... v2 biomedclip threshold vs oof csv confusion ... EOF
sed -n '403p;416p;485p;1058p' RESULTS.md; grep -n "尚未公開\|不在版本庫內\|pending\|未公開\|後續工作\|Zenodo" docs/report.md README.md ...
cmp app/app.py deploy/app.py
```

---

## 3. Freeze-before-test timeline

**Git facts (this clone, HEAD 4794ad6, 73 commits).** One identity on all 73
commits (`Thomas Wang <thomasf45566@gmail.com>` as author and committer);
author date == committer date on 72/73 (6c180ee: 04:26:23 vs 04:26:29, the
6-second amend PROVENANCE §2 describes). Tags: `frozen-v1` annotated (tag object
af43104, tagger 2026-08-29T18:38:35+08) → f167665 (18:38:34); `external-v1`
lightweight → 2cbe1da (23:14:26); `v2-biomedclip` annotated → 568fdac;
`v2-loco` → 14c3bf2; `v1.0-audited` annotated (2e7da4d, tagger 2026-09-07
13:05:16) → fa193a8. The tag object SHAs are identical in this clone, the
author's working copy and on GitHub (`git/refs/tags/frozen-v1` → af43104…,
`v1.0-audited` → 2e7da4d…).

**Ordering (all +08:00), from `git log --format='%h|%ad|%cd|%s'`:**

| step | commit | date | files |
|---|---|---|---|
| calibration T=2.3644 | 25ef75a | 08-29 18:33:03 | |
| FREEZE `frozen-v1` (operating_point.json) | f167665 | 08-29 18:38:34 | |
| protocol (a)–(d) + BUSI keep-list + self-check | a16a64f | 08-29 18:49:09 | 10 files, protocol −0 lines |
| Amendment 1 (GDPH/SYSUCC keep-lists, pHash) | 436ab56 | 08-29 21:44:18 | 10 files, protocol −0 lines |
| `external-v1` (four preds CSVs) | 2cbe1da | 08-29 23:14:26 | |
| BI-RADS secondary (protocol h) | 5afac64 | 08-29 23:24:18 | |
| Amendment 2 | 610e87f | 08-31 16:46:51 | **1 file** |
| first M1 results | fbc4bba | 08-31 17:56:44 | |
| identity rewrite recorded | 2c24500 | 08-31 17:58:46 | |
| Amendment 3 | 78f3991 | 08-31 20:27:20 | **1 file** |
| members persisted (n) | 9fc3be1 | 08-31 21:18:46 | |
| Amendment 4 | 3faa70b | 08-31 22:53:53 | **1 file** |
| LOCO data layer (no training) | 2f92e65 | 08-31 23:12:56 | |
| BiomedCLIP substitution note (protocol edited again, +0/−0) | 4fc3fb2 | 09-01 18:38:49 | 5 files |
| BiomedCLIP CV trained | 6c180ee | 09-02 04:26 | |
| `v2-biomedclip` | 568fdac | 09-02 14:34:40 | |
| LOCO runs 1–4 | 99aeb3e…7764aba | 09-03 14:08 – 22:01 | |
| `v2-loco` | 14c3bf2 | 09-04 07:10:16 | |

**Prediction CSVs never overwritten.** `git log --diff-filter=M` on
`reports/external_{breast,busi,gdph,sysucc}_preds.csv`, `oof_vit_preds.csv`,
`tta_vit_summary.csv`, `external_selfcheck.txt`, `calibration.json`,
`operating_point.json`, the four keep-list/fold CSVs, `cv_vit_summary.csv`,
all nine `v2_*_preds.csv` and four `v2_members_*.csv`: **zero modification
commits; each file has exactly one adding commit** (2cbe1da / 3c3407e /
a16a64f / 25ef75a / f167665 / 436ab56 / 67915b0 / cb53138). The one artifact
with modification commits is `reports/v2_loco_summary.csv` (3 M commits,
99aeb3e→7764aba): `git log -p` shows each is a pure one-row append for the
next held-out cohort, no existing row touched.

`git diff 2cbe1da HEAD -- RESULTS.md | grep '^-' | grep -v '^---'` → **0
lines** (the external-v1 section was never edited; P2 errata appended). Every
commit touching `data/external_protocol.md` removes 0 lines (append-only
verified for a16a64f, 436ab56, 610e87f, 78f3991, 3faa70b, 4fc3fb2).

**History rewrite.** 2c24500's message records the filter-branch identity
rewrite; the current history is consistent with "dates preserved": all 30
pre-rewrite commits are in the author's working copy as **unreachable
objects** (`LC_ALL=C git fsck --unreachable --no-reflogs` → 31 unreachable
commits, matching PROVENANCE §3's count; note that under the author's zh_TW
locale the same command prints "無法取得 commit", which is why a naive grep
for "unreachable" returns 0), and for the 11 pairs I checked by hand
(d0cd52c→2d38e90, 51e237c→5e78373, 1a02bd6→0ae8efc, c9adedd→0c23853,
82bd016→a14990f, 402a193→b213aa1, dd1b9fd→f167665, 7cc9414→2cbe1da,
3befcec→436ab56, 10bf7f4→a16a64f) the tree hash and author date are
identical and only the e-mail differs (`…@Thomass-MacBook-Air.local` →
gmail). 8e058bd→6c180ee has a different tree (the amend), as PROVENANCE says.
Those objects exist only on the author's disk; this clone has none of them
(`git rev-parse d0cd52c` fails here), so the mapping remains unverifiable from
any published artifact, exactly as PROVENANCE §3 states. They are also
subject to `git gc` pruning at any time — the evidence has no durable home.

**External anchors (network, 2026-09-11).**
- GitHub `thomasf45566/breast-us-cad`: **public**, `created_at`
  2026-08-26T02:45:16Z (i.e. the remote existed, private, 4.5 h before the first
  commit's author date; "Nothing had been pushed" before 08-31 cannot be
  checked from outside), `pushed_at` 2026-09-07T08:15:39Z; the 7 tags on GitHub
  point at the same commits as the local tags; HEAD on GitHub = 4794ad6 = this
  clone = the author's copy.
- HF weights repo created 2026-08-30T04:25:49Z; cv_vit_fold1–4 uploaded
  04:27–04:31Z; fold5 + seg + the two JSONs 07:12:50–07:13:06Z; v2 commit
  5ca5ba0 2026-09-07T04:11:35Z. Space created 2026-08-30T09:24:53Z, last
  modified 2026-09-07T03:27:26Z (sha b7878b3). All identical to PROVENANCE §5.
- Zenodo 10.5281/zenodo.22630912 created 2026-09-07T07:30:33Z.

The first third-party timestamp of any kind (HF, 2026-08-30 04:25Z) still
postdates the external-v1 commit (2026-08-29 15:14Z) by ~13 h; Zenodo and
GitHub postdate it by 9 days. Nothing has changed in this respect since the
first audit, and nothing can.

**Verdict: PASS on internal consistency (freeze → protocol → amendment →
single-commit, never-modified prediction CSVs; Amendments 2/3/4 committed
alone before their first governed computation; external-v1 never amended;
rewrite preserved trees and dates, now documented with the pre→post map).
CONCERN — major, unchanged and permanent — the pre-registration ordering is
self-certified: private rewritten history, no contemporaneous external
timestamp, and the pre-rewrite objects that corroborate the map live only as
unreachable objects on one laptop.**

---

## 4. Protocol compliance (data/external_protocol.md read end to end, 524 lines; HEAD 4794ad6)

Method: every rule/clause of the base protocol, Amendments 1–4 and the substitution note was
matched against the enforcing code (file:line) or the committed artifact; the prior audit's
§4 table was NOT trusted — each row re-derived. Read-only; no inference run in this section.

| Rule | Enforcement / evidence (re-verified) | Status |
|---|---|---|
| Task: normals out of scope | `src/external_val.py:69` `meta[meta["Classification"] != "normal"]`; preds n=252 | OK |
| (a) BrEaST labels/mapping/main image only/patient=case | `external_val.py:66-78` (`Image_filename`, `CaseID` as patient_id, map benign→0/malignant→1, `assert notna`) | OK |
| (b) BUSI keep-list, normals excluded, image-level bootstrap | `external_val.py:81-92` `patient_id = keep["filename"]` ⇒ image-level groups in `patient_bootstrap` (:107); `DATASETS["busi"]=(…, False)` (:99) prints "IMAGE-level (no patient IDs published)" (:147) | OK |
| (c) Preprocessing = frozen val path | `external_val.py:56-63` `BusDataset` + `get_transforms("val")` + `ensemble_tta_probs_from_loader`; docstring :5-9 | OK |
| (d) per-cohort, never pooled; AUC on calibrated probs; 2000× seed 42 percentile CI; sens/spec at 0.2683; outputs | `run_external` per cohort (:133-176); `N_BOOT=2000, BOOT_SEED=42` (:46-47); `roc_auc_score(labels, probs)` on calibrated (:142); CSV/ROC/CM written (:150-162). **Deviation:** (d) says "one RESULTS.md section per dataset" — RESULTS.md:172-206 is ONE section with a four-row table. Presentational | OK (minor presentational deviation, undeclared) |
| (e) GDPH/SYSUCC separate cohorts, keep-lists, image-level | `DATASETS` entries (:100-101) via `build_keeplist_df`; keep-lists `data/splits/{gdph,sysucc}_clean.csv` tracked | OK (counts re-derived by §5 worker) |
| (f) BUSI_WHU excluded | absent from `DATASETS` (:97-102, comment :96); `grep -ril busi_whu reports data/splits` → nothing but the protocol text | OK |
| (g) cross-set sweep, candidates adjudicated | `reports/phash_sweep_hits.csv`, `phash_cross_pairs_table.csv`; **the adjudication figure `reports/phash_cross_pairs.png` is no longer distributed** (untracked P4c, `.gitignore`); protocol text (g) still cites it as the evidence | OK for the run; **the distributed repo can no longer show the visual adjudication** — only a table + BUS-BRA-side thumbnail (minor) |
| (h) BI-RADS ≥ 4a, after primary, saved preds only, stray 'c' excluded from reader comparison only | `src/birads_comparison.py:23` `POSITIVE={"4a","4b","4c","5"}`; `:32` reads preds CSV only; `:52` `df[df[r].isin(VALID)]` drops non-BI-RADS values per reader column only (generalises the "reader1 'c'" clause; RESULTS.md:239-244 records the 'c' sat in reader2 and was dedup-dropped anyway) | OK |
| Am.2 (i) inputs = saved CSVs; logit recovery | `v2_recalib_curve.py` loads `reports/external_*_preds.csv`, `probs_to_logits` (RESULTS.md:490-493) | OK |
| Am.2 (j) k-grid drop k≥n/2; R=500 seed 42; natural prevalence; degenerate fallback | `K_GRID` :57 with per-cohort drop; `rng = default_rng([seed, ci, k, draw])` :175; degenerate branch :122-125 returns frozen thr, flagged; `frac_degenerate` in summary CSV | OK |
| Am.2 (k) M1/M2a/M2b/M3 | `draw_metrics` :112-152 matches (k) | OK |
| Am.2 (l) held-out only; recovery vs in-sample oracle | `assert len(set(cal_idx)&set(eval_idx))==0` :274; oracle asserted sens ≥ 0.90 :261 | OK |
| Am.2 (m) k* rule; all k reported | `KSTAR_RECOVERY=0.80, KSTAR_SENS=0.85` :63-64, `kstar()` :221-225; full grid in `v2_recalib_M1_summary.csv` (22 rows, checked) | OK |
| **Am.2 — M2 "fit_failed" fallback** | NOT in the amendment; `:135-141` catches `RuntimeError` (non-positive T) → frozen pipeline, flagged; 7.2–20.4 % of k=10 draws (`v2_recalib_methods_summary.csv` frac_fit_failed: 0.122/0.204/0.204/0.072). Declared "run-time necessity" RESULTS.md:499-504; report §2.7:73 says "執行時新增、非 amendment 所載" | deviation, declared everywhere it matters (minor) |
| **Am.2 — k_reliable** | POST-HOC by the script's own docstring (`:20`, `:62`) and RESULTS.md:509-512; labelled POST-HOC in README:62, report abstract:22/§2.7:73/§3.7:126, project_summary:35, interview_script:26/45, app About (app.py:81) | OK, consistently labelled |
| Am.2 — cross-site matrix | "NOT pre-registered" RESULTS.md:565; report §3.8 heading "(post-hoc、描述性)"; project_summary:35 "(post-hoc、in-sample、描述性)" | OK |
| Am.3 (n) constrained re-run, bitwise or abort, no metrics | `v2_dump_members.py:62-77` bitwise|≤1e-9 check, `sys.exit(1)` on mismatch; no metric computed in the script | OK |
| Am.3 (o) signals U_std/U_range/−M only | `v2_disagreement.py` (three signals; no combination) | OK |
| Am.3 (p) AUROC CIs, q grid, stable tie-break, enrichment | `:52` 2000/42; `:119-121` `sort_values([signal,"image_path"], kind="mergesort")`; **single figure `v2_abstention.png` split into `v2_disagreement_auroc.png` + `v2_abstention_curves.png`** — declared RESULTS.md:608-610, report §2.7:75 | OK; presentational deviation, declared (minor) |
| Am.3 (q) three-clause criterion, mechanical | `:152-156` c1/c2/c3 exactly as registered; 0/4 recorded RESULTS.md:663-680; no relaxation in any document (checked README:91-94, report §3.9/§4.3, summary:36, script:45) | OK |
| Am.3 (r) no internal reference stated | RESULTS.md:689-693 | OK |
| Am.4 (s) external-v1 untouched | `git diff 2cbe1da HEAD -- RESULTS.md \| grep '^-' \| grep -v '^---' \| wc -l` → **0** (run 2026-09-11); Errata appended only | OK |
| Am.4 (t) pool 85/15 image-level seed 42 once, held-out absent, val-only selection, single model + TTA, single-shot | `v2_data.py:11-12, 98-104` asserts; `v2_loco.py:211-212` refuses if preds exist; `--confirm` :297-321; T + thr fit on pooled val :165-189 | OK |
| Am.4 (t) "all other hyperparameters identical to v1" | `diff configs/vit.yaml configs/v2_loco.yaml` → only data root/folds removed, `early_stop_patience: 7` added, run naming. **Patience 7 not in the amendment** ("early stopping (on validation AUC)" only) | under-specified, declared (report §2.7:77, §4.5:172; RESULTS.md:875) (minor) |
| Am.4 (t) comparator = fold-5 single + TTA from saved members | `v2_loco.py:249-258`; `v2_loco_report.py:71-83,139-140` asserts | OK |
| Am.4 (t) criterion mechanical | `v2_loco_report.py:256-268` `a = da >= 0.01`, `b = dsp >= 0.10 and sens >= 0.85`, `met = na>=3 or nb>=3`; 4/4 A, 2/4 B | OK |
| Am.4 (u) USFM coverage check → fallback declared pre-training | substitution note in protocol; RESULTS.md:707-740; commit order verified by §3 worker | OK |
| Am.4 (u) criterion mechanical, no rounding | `v2_biomedclip_external.py:56-57, 184-187`; SYSUCC 0.700163 counted fail | OK, strict |
| Am.4 (v) backbone rule | LOCO used ImageNet ViT (config diff above) | OK |
| Am.4 (u) "patient-level folds, data/splits/*.csv" | at amendment time the folds were NOT in data/splits (audit §1); they are now (`busbra_official_5fold.csv`) | text was inaccurate when written; now true (minor) |

**Softening / reinterpretation after results (re-checked, nothing new relaxed):**
- Q1: the registered *question* (protocol:390-392, specificity collapse) vs the fired AUC branch. RESULTS.md:1028-1033 STILL reads "multi-source training reduces the domain-shift specificity collapse, carried by the AUC branch" **in place**; the correction lives only in the appended Errata (RESULTS.md:1245-1258). Every other document (README:65-72, report abstract:22/§3.11:150/§5:180, project_summary:38, interview_script:26/48) now carries the four caveats in the same paragraph. A reader of RESULTS.md alone meets the over-claim first and the erratum 220 lines later.
- Same in-place/erratum pattern for RESULTS.md:485 and :1058 ("≥ 0.92") and :403 ("1,879") — corrected only by appendix (by design, to keep `git diff 2cbe1da` at 0 removed lines). Acceptable but it means RESULTS.md is no longer self-consistent line by line.
- Abstention: 0/4 reported first; the reframing is labelled "作者事後詮釋(非判準之一部分)" (report §3.9:132, §4.3:166). No relaxation.
- BiomedCLIP: NOT CLAIMED everywhere (README:93, report abstract:22/§3.10:135/§4.1:160/§4.3:166/§5:180, summary:37, script:48). The ratio numbers 0.69–0.96 still appear in report §3.10 but are framed as "判準所定義之良性漂移比" with the verdict in the same sentence. OK.
- CoarseDropout: rule quoted verbatim from plan.md:375-378 ("adopt only if pooled OOF AUC >= (current ViT OOF AUC - 0.01) AND the saliency check shows visibly reduced caliper-adjacent heat") in RESULTS.md:61-64; gate 1 PASS by 0.0008, gate 2 FAIL, discarded (RESULTS.md:66-77); report §2.2:47 quotes the AND rule; README:91-92 and report §4.3 add the "visual gate on three images, no robustness metric" caveat. Faithful.
- TTA adoption not pre-registered: disclosed README:195-197, report §2.2:49, plan.md:362-365, PROVENANCE §4.
- P2 post-hoc analyses: header "POST-HOC — not pre-registered" RESULTS.md:1096; scripts' docstrings begin "POST-HOC:" (`posthoc_epoch_selection.py:1`, `posthoc_nested_threshold.py:1`); README rows a′/b′ tagged POST-HOC. The Errata section does not alter any number silently — each entry quotes old→new with the reason; the only numeric content it introduces (6.15/8.62 s latencies) is attributed.

**Verdict: PASS — every registered rule is enforced in code or evidenced by artifacts, and no negative verdict was relaxed. CONCERN — minor — for (i) the in-place RESULTS.md sentences that remain wrong pending an appendix the reader must find (:485, :1028-1033, :1058, :403), (ii) the two undeclared-in-advance implementation choices (M2 fit-failure fallback, patience 7) and the one-section-vs-per-dataset and one-figure-vs-two presentational deviations, and (iii) protocol (g)'s cited adjudication figure no longer being distributable.**

---

## 5. Data audit reproduction

**Environment.** Overlay of the clone (no `.git`) with `data/raw/*` and `models/*.pt` symlinked read-only from `$DATA_ROOT` / `$MODELS_ROOT`; fresh `uv venv --python 3.12` + `requirements.txt`. Script: scratchpad `sec5_phash_repro.py`, log `sec5_phash_repro.log`. It imports the repository's own functions (`phash_sweep.collect_sets/hash_matrix/pair_hits/dedup_keep_list`, `dedup_busi.phash_file/find/union`, `NEAR_DUP_MAX = 8`) so the rules are the committed ones (64-bit pHash, d ≤ 8, union-find groups, lexicographic keep, label-conflict groups dropped whole). Runtime ≈ 40 s.

**Set sizes and pair counts (my run):**
```
busbra 1875 | breast 252 | busi (keep-list) 379 | gdph 846 | sysucc 1559 | busi_raw 386
busbra vs busbra: pairs=1756875 min_d=8  hits=1        busbra vs breast: 472500  min_d=10 hits=0
busbra vs busi:   710625  min_d=10 hits=0             busbra vs gdph:   1586250 min_d=10 hits=0
busbra vs sysucc: 2923125 min_d=8  hits=1             breast vs breast: 31626   min_d=12 hits=0
breast vs busi:   95508   min_d=10 hits=0             breast vs gdph:   213192  min_d=12 hits=0
breast vs sysucc: 392868  min_d=10 hits=0             busi vs busi:     71631   min_d=10 hits=0
busi vs gdph:     320634  min_d=10 hits=0             busi vs sysucc:   590861  min_d=10 hits=0
gdph vs gdph:     357435  min_d=0  hits=37            gdph vs sysucc:   1318914 min_d=8  hits=1
sysucc vs sysucc: 1214461 min_d=0  hits=583
TOTAL pairs (5 sets, within+cross): 12056505   cross only: 8624477   formula ΣC(n,2)+Σn_a·n_b: 12056505
```
The 12,056,505 figure in README:86, report §2.4 and external_protocol.md reproduces exactly (the old "9.7M" is gone from every document).

**Hits file:** 623 hit rows, mine vs `reports/phash_sweep_hits.csv` (624 lines incl. header): `hits identical to committed: True` (row-for-row equality after sorting on set_a/set_b/image_a/image_b).

**Cross-set candidates at d ≤ 8 — exactly two, as registered:**
```
busbra sysucc  bus_0999-l.png  malignant(81).png  8
gdph   sysucc  benign(784).png malignant(23).png  8
```
Every other set pair has minimum distance ≥ 10 (BrEaST–BrEaST and BrEaST–GDPH ≥ 12). BUSI keep-list vs each of the other four sets: min d = 10, 0 hits.

**Within-set dedup:**
```
gdph   within hits 37 (34 at d=0) -> 846 -> 810  (dropped 34 dup, 2 label-conflict)   keep-list == data/splits/gdph_clean.csv:   True
sysucc within hits 583 (120 at d=0) -> 1559 -> 1013 (dropped 507 dup, 39 label-conflict) keep-list == data/splits/sysucc_clean.csv: True
busi_raw within hits 7 (min d 2) -> 386 -> 379 (dropped 7 dup, 0 conflict)           filenames == data/splits/busi_clean.csv: True; phash column matches: True
```
Twelve SYSUCC label-conflict groups and one GDPH group were printed by the repo's own function (e.g. `['benign(457).png','malignant(1184).png']`, `['benign(283).png','benign(288).png','malignant(697).png','malignant(699).png']`). SYSUCC 546/1559 = 35.0 % (README:87 "35 %"). Per-class: GDPH benign 443→435, malignant 403→375; SYSUCC benign 443→289, malignant 1116→724.

**BUSI_WHU absence.** `grep -rni busi_whu` over `src scripts tests configs app deploy data/splits reports/*.csv models/*.json RESULTS.md README.md plan.md data/README.md data/external_protocol.md docs/report.md THIRD_PARTY_DATA.md`: hits only in comments/prose — `src/external_val.py:17,96` ("busi_whu EXCLUDED"), `plan.md:51`, `data/external_protocol.md:84,116,121`, `docs/report.md:52,169`. Zero hits in any CSV under `reports/`, in `data/splits`, in `models/`, in `src/v2_data.py`, and — notably — zero hits in `RESULTS.md` (it does not even mention the exclusion). `DATASETS` in `src/external_val.py:97-102` has four keys (breast, busi, gdph, sysucc); `src/v2_data.py:38-40` imports that registry and hard-codes `EXTERNAL_COHORTS = ("breast","busi","gdph","sysucc")`. No code path reads `data/raw/busi_whu` (`grep -rn "raw/busi" src scripts app deploy tests` → only `dedup_busi.py:32` = `data/raw/busi/images`). The raw directory exists on the author's disk and is unused.

**What I would have expected to find had there been a problem:** a keep-list differing from a fresh re-hash (it would show as `False` above), a third cross-set hit, or a busi_whu path in the registry/CSVs. None present. Residual limitation unchanged from the prior audit: pHash d ≤ 8 cannot exclude re-scans/crops at d > 8; the documents now say exactly that (README:89-90, report §2.4).

**Verdict: PASS.** Counts (846→810, 1559→1013 = 507 + 39, 386→379), all three keep-lists, the 623-row hits file, the two d = 8 cross-set candidates, min d ≥ 10 elsewhere, and the 12,056,505 pair count reproduce exactly with the repository's own functions in a fresh environment; BUSI_WHU is absent from every registry and result.

---

## 6. Pipeline parity

### (a) Byte level
```
sha256 src/inference.py    158d30ece3b5f9ddc79934029ca7a4cceff10f78c4924ad996721332affbf8cd
sha256 deploy/inference.py 158d30ece3b5f9ddc79934029ca7a4cceff10f78c4924ad996721332affbf8cd   (equal)
sha256 app/app.py          8abeea5fd6636bc09bc263fd263c3831e411a9fd4a807ad8a899ca6c45357812
sha256 deploy/app.py       8abeea5fd6636bc09bc263fd263c3831e411a9fd4a807ad8a899ca6c45357812   (equal)
```
The four example PNGs are byte-identical between `app/examples`, `deploy/examples` and the live Space (`cmp`; `benign_bus_0186-r.png` md5 a8ebac44… in both trees; `git log -- app/examples deploy/examples` shows no change since f3170bc 2026-08-30). The previous audit's §6 statement that `app/examples/benign_bus_0186-r.png` had a different md5 (7578b648…) is not reproducible from the repository at HEAD.

### (b) AST level (scratchpad `sec6_ast.py`: docstrings stripped, `ast.dump` md5 per top-level function/class, 12 v1 files + all `src/v2_*.py`)
- All 19 functions of `src/inference.py` are AST-identical to `deploy/inference.py`; all 8 top-level functions of `app/app.py` (`build_ui, lesion_contour_panel, load_pipeline, predict, preprocess, result_card, startup_latency, …`) are AST-identical to `deploy/app.py`.
- `predict_hflip_pair`: SAME across `src/inference.py`, `deploy/inference.py`, `src/evaluate.py`.
- `make_cam`: SAME across inference and `src/explain.py`; `vit_reshape_transform`: SAME across inference and `src/gradcam_check.py`.
- DIFF `probs_to_logits` (`src/calibrate.py:33` uses module constant `EPS = 1e-6`; `inference.py` uses default arg `eps=1e-6`) — same semantics.
- DIFF `load_rgb01`: `src/gradcam_check.py:41` uses `cv2.resize(img, (224,224))` (cv2 bilinear) whereas `inference.load_rgb01` uses `resize_bilinear_frozen`; DIFF `cam_overlay`: `src/explain.py:71` feeds the CAM input through `get_transforms("val")` (albumentations) whereas `inference.cam_overlay` uses `val_transform_rgb`. These affect only Grad-CAM figures, not any metric.
- Other DIFFs (`main`, `load_cohort`, `sens_spec`, `plot_curves`, `self_check`) are per-script entry points / helpers, not shared frozen code.

**Which preprocessing each path actually uses (grep of call sites):**
| path | preprocessing | inference core |
|---|---|---|
| `src/train.py` / `data.make_dataloaders` (`data.py:197-204`) | albumentations `A.Resize` (cv2 INTER_LINEAR, `data.py:134`) | — |
| `src/evaluate.py:126,133` | `get_transforms("val")` (albumentations) | `predict_hflip_pair` |
| `src/tta_eval.py:49,54` | `get_transforms("val")` | `predict_hflip_pair` |
| `src/external_val.py:57,63` | `get_transforms("val")` | `inference.ensemble_tta_probs_from_loader` |
| `src/v2_*` (dump_members, loco, biomedclip_*, cross_site inputs) | `get_transforms("val")` | `predict_hflip_pair` / `ensemble_tta_probs_from_loader` |
| `app/app.py:104,154` and `deploy/app.py` (identical) | `inference.preprocess_gray` → `resize_bilinear_frozen` (integer KleidiCV port) | `inference.ensemble_prob` (per-image float64 `.item()` mean) |

So: the **model loading, TTA and calibration code is shared** (AST-identical or the same imported function); the **resize stage is not shared** — training/eval/external validation use albumentations/cv2, the demo uses the integer port. This is exactly what the code and documents now say (`inference.py` module docstring lines 13-24; `README.md:95-99`; `docs/report.md` §2.8; `app/app.py` About lines 73-75; `deploy/README.md` "Preprocessing in this Space differs …"). The reduction also differs (batched float32 mean in the loader path vs per-image `.item()` in `ensemble_prob`); the prior audit measured this at ≤ 3e-8 and nothing here changes that.

### (c) Self-check in the fresh venv (MPS)
```
$ python src/external_val.py --self-check
self-check: fold-5 val, fold-5 ckpt only, n = 383
  expected (reports/tta_vit_summary.csv): 0.9234433158791244
  reproduced by external_val code path:   0.9234433158791243
  calibration (T=2.3644) + threshold (0.2683) applied: 156/383 flagged malignant
  SELF-CHECK PASSED — AUC reproduced digit-for-digit.
$ diff <stdout> reports/external_selfcheck.txt   → BYTE-IDENTICAL   (exit 0)
```
(The 1-ulp difference between "expected" and "reproduced" is inside the script's own 1e-12 tolerance and is the same value the committed file records.) I did not run `v2_biomedclip_external.py --self-check`: its code (`self_check`, lines 87-108) touches only BUS-BRA fold 5 with the v2 fold-5 checkpoint and would have been permitted, but it adds nothing to the v1 parity question; the prior audit reported 0.9230647908649297 PASSED.

### (d) Resize-dispatch discrepancy — docstrings vs artifact
`reports/v2_resize_dispatch_impact.csv` (1421 rows; columns `cohort,image_path,y_true,p_cal_canonical,p_cal_space,abs_delta,abs_delta_pure_resize,pred_canonical,pred_space,decision_flip`):
```
affected per cohort: breast 99, gdph 309, sysucc 1013 (busi 0 → no rows)      total 1421 / 2454
abs_delta: max 0.0072268, median 0.0007700     decision_flip: 2
  case038.png  y=0  0.268900 → 0.267838  |Δ| 0.001062  pred 1→0
  case200.png  y=0  0.269467 → 0.265696  |Δ| 0.003771  pred 1→0
```
`inference.py` module docstring (lines 13-24): "1421/2454 external keep-list images, e.g. all of SYSUCC … max |delta calibrated prob| 0.0072, median 0.0008, decisions … flip on 2/1421 affected images (both borderline BrEaST cases, p within 0.002 of 0.2683)" — **accurate** (both flips have p_canonical within 0.0007–0.0012 of 0.2683). `resize_bilinear_frozen` docstring (lines 147-162) correctly carries the CAVEAT that bitwise equality holds on the BUS-BRA envelope only. `v2_data.py:20-25` docstring gives the per-cohort counts 99/0/309/1013 — accurate. README:95-99 ("preprocessing differs … on non-BUS-BRA image sizes … impact ≤ 7×10⁻³"), report §2.8 (99/252, 0/379, 309/810, 1013/1013; max 7×10⁻³, median 8×10⁻⁴, 2 flips case038/case200), app About and deploy/README ("≤ 0.007") all agree with the CSV.

**One stale number survives in code:** `src/inference.py:153` and `deploy/inference.py:153` (`resize_bilinear_frozen` docstring) still say "every BUS-BRA image (1879 images, 713 distinct sizes)". BUS-BRA has 1875 images; the prior audit flagged the same "1,879" in RESULTS.md:403 and it was corrected in plan.md/report and via the RESULTS.md errata (response A-2.5), but the docstring — the one place a user of the inference module reads — was not. Minor.

### (e) Live Space
`scripts/verify_space.py` reads `reports/app_example_probs.json` and `app/examples/*.png`, queries `/predict` on the Space and writes nothing to disk; run with `--space happytommy/breast-us-cad` (the default calls `whoami()`, which needs an HF token). Log `sec6_verify_space.log`:
```
first prediction (already warm?): 82.5 s     (4 ReadTimeouts while the Space woke)
  benign_bus_0186-r.png:    raw=0.021605699649080636 cal=0.1662192214946394 delta=3.32e-09 [7.23 s]
  benign_bus_0223-l.png:    raw=0.05352667736187868  cal=0.22882984950348725 delta=9.61e-09 [7.48 s]
  malignant_bus_0328-r.png: raw=0.8696612179279327   cal=0.6905585278113721 delta=1.07e-07 [9.31 s]
  malignant_bus_0663-l.png: raw=0.8794305920600891   cal=0.6985512526504918 delta=2.98e-08 [7.13 s]
warm latency: median 7.18 s (min 5.92 / max 9.31, n=6)
RESULT: MATCH AT PLATFORM FLOOR (max delta 1.07e-07 < 1e-05; identical at display precision)
```
Max |Δ| 1.07e-7 equals the value recorded in RESULTS.md/plan.md; all four classify correctly; warm median 7.18 s sits inside the "6–9 s" range the P6 sweep introduced (RESULTS.md Errata 1276-1280: 6.8 / 6.15 / 8.62 s).

**Live Space files vs `deploy/` at HEAD** (fetched via `huggingface_hub`; Space sha `b7878b33…`, last modified 2026-09-07 03:27:26Z, matching PROVENANCE.md §5):
```
app.py            IDENTICAL
requirements.txt  IDENTICAL
examples/*.png    IDENTICAL (4/4)
inference.py      DIFFERS: live lacks HF_V2_PREFIX / hf_repo_path(); live resolve_weight() downloads `filename` verbatim
README.md         DIFFERS: live ends "Used under the dataset's research license." — deploy/ adds the CC BY 4.0 / BUSBRA_LICENSE.txt / Apache-2.0 sentence
```
`git show` pins the live files: live `inference.py` == `deploy/inference.py` at 5652f0a (P1) and f3a616f; live `README.md` == `deploy/README.md` at 5652f0a. They differ from 67915b0 (P3, 12:12 +08, added `hf_repo_path`) and be627ad (P4b, 12:34 +08, licence sentence) — both committed **after** the last Space push (11:27 +08) and never pushed. Consequences: (i) `RESULTS.md:379-380` "deploy/ bundles app.py + inference.py + the 4 examples and is the Space repo verbatim" is false at HEAD by two files; (ii) functionally harmless for the demo — the Space only loads v1 root-level files, which is why the four probabilities still match — but the live Space cannot resolve any `v2_*` weight, and its About/README lacks the licence attribution that P4b added for publication. The audit response nowhere claims the Space was re-pushed after P1, so this is an omission, not a false statement.

### (f) Checksums
```
$ cd models && shasum -a 256 -c CHECKSUMS.txt     → 28/28 OK (19 .pt incl. pretrained/biomedclip_vitb16_timm.pt, 10 JSON; .pt via read-only symlinks to $MODELS_ROOT)
HF happytommy/breast-us-cad-weights, model_info(files_metadata=True): 16/16 LFS objects (.pt) sha256 == CHECKSUMS.txt
HF JSONs are not LFS → downloaded and hashed: 12/12 == CHECKSUMS.txt; HF CHECKSUMS.txt byte-identical to models/CHECKSUMS.txt
HF repo: created 2026-08-30 04:25:49Z; cv_vit_fold1-4 04:27-04:31Z; fold5/seg/JSONs 07:12-07:13Z; v2/ + CHECKSUMS.txt commit 5ca5ba0 2026-09-07 04:11:35Z
```
28/28 verified against both the local files and the published ones; the HF commit timeline matches PROVENANCE.md §5 to the second.

**Verdict: PASS on the mechanics** (frozen module byte- and AST-identical between src/ and deploy/; self-check byte-identical in a fresh environment; live Space at the platform floor; 28/28 checksums; resize-dispatch numbers in every docstring and document match the artifact). **CONCERN — minor** on two points: the live Space runs the P1 version of `inference.py`/`README.md`, so `RESULTS.md:380` "is the Space repo verbatim" no longer holds and the v2 path/licence text are not live; and `inference.py:153` still says "1879 images". The preprocessing split (cv2 for validation, integer port for the demo) is real but is now stated accurately everywhere I looked.

---

## 7. Statistical claims

**Bootstrap units vs protocol (d)** (re-verified in code, not from the prior audit):
- Internal operating point: `pick_threshold.py:67-98` groups by `patient_id` (1064 patients) — patient-level. ✓
- External BrEaST: `external_val.py:73` patient_id = CaseID; 252 cases = 252 images — case-level ≡ image-level. ✓
- BUSI/GDPH/SYSUCC: patient_id = filename (`external_val.py:89`) — image-level, and the limitation is printed (:147) and stated in RESULTS.md:178-180, README:236, report §2.3:52/§4.5:172. ✓
- Amendment 2 draws: `v2_recalib_curve.py:89-90` asserts BrEaST 1 image/patient; image-level elsewhere, stated RESULTS.md:452-455. ✓
- Amendment 3 AUROC CIs: `v2_disagreement.py:84-87` same grouping. ✓
- LOCO CIs and paired ΔAUC/Δspec: `v2_loco_report.py:95-99` same groups, paired resamples. ✓
- BiomedCLIP external: `v2_biomedclip_external.py:72` via `patient_bootstrap`. ✓
- Nested operating point (POST-HOC 2): per-fold point estimates only, no CI — none claimed. ✓
All CIs in README/report/RESULTS trace to one of these. No CI uses a wrong unit.

**Strong-language check** (every hit of significant/exceeds/beats/non-overlapping/不重疊/顯著/優於/higher/met/達成 in README.md, docs/report.md, docs/project_summary.md, docs/interview_script.md; values from the committed CSVs):

| Claim (file:line) | Evidence | Status |
|---|---|---|
| RESULTS.md:634 / report §3.9:132 "U_std 與 margin 之 CI 於 BUSI、GDPH 不重疊,於 BrEaST 重疊" | `v2_abstention_metrics.csv`: BUSI 0.8119–0.8950 vs 0.6943–0.8089 (disjoint); GDPH 0.7718–0.8329 vs 0.6824–0.7518 (disjoint); BrEaST 0.7100–0.8314 vs 0.5945–0.7286 (overlap) | supported; prior audit item 3 now correct |
| report §4.3:166 "3/4 世代優於 margin" (abstention) | point estimates 0.774>0.664, 0.856>0.754, 0.802>0.717; BrEaST CIs overlap | point-estimate language; criterion (q)3 is itself a point comparison — acceptable, but "優於" on BrEaST is not CI-backed |
| report §3.11:150 / RESULTS.md:1008 "GDPH … AUC CI 不重疊" (LOCO vs v1-single) | `v2_loco_q1c_table.csv`: 0.9302–0.9601 vs 0.8668–0.9128 | supported |
| README:65-72, report abstract:22, §3.11:150, §5:180, summary:38, script:26/48 "criterion MET via the AUC branch (4/4)" | paired ΔAUC CIs: BrEaST −0.0155…+0.0550, SYSUCC −0.0099…+0.0346 **cross zero**; BUSI +0.0074…+0.0550, GDPH +0.0371…+0.0743 exclude zero | met as registered (point rule). **Caveat "2/4 CIs include zero" is present in the SAME sentence/paragraph in all six places** (README:67-68; report:22, :150, :180; summary:38; script:26, :48). app About does not mention LOCO at all. ✓ |
| README:71, report §3.11:148/:150, §4.1:160, summary:38 "Δspec +0.27 to +0.30 (paired CIs exclude zero…)" | Δspec CIs +0.196…+0.362 / +0.209…+0.338 / +0.248…+0.357 / +0.212…+0.321 | supported; "at different thresholds per model" stated alongside in README:71-72, report:150; summary:38 says "於各自閾值下" ✓; **interview_script:26/48 omit the different-thresholds caveat** (minor) |
| report §3.10:135, abstract:22 "外部 AUC 點估計於四世代皆較低" (BiomedCLIP) | `v2_biomedclip_external_comparison.csv`: unpaired CIs overlap on all four (e.g. GDPH 0.895–0.934 vs 0.858–0.904 barely overlap at 0.895<0.904); no paired test | correctly labelled "點估計"; §3.10 adds "未配對 CI 皆重疊,未做配對檢定" ✓ |
| report §1.1:31 "緻密乳房中超音波敏感度顯著優於攝影" | literature claim, no citation, not a result | uncited (see §8) |
| report §3.5:120 "Reader1 之 specificity 遠高於模型(0.89 vs 0.45,點估計;未做任何檢定)" | 0.8943 vs 0.4529 | correctly de-escalated from the prior "顯著" |
| report §3.3:114 / README:53-55 "BrEaST … SYSUCC 之 CI 上界皆低於內部 0.9254" | 0.9024 < 0.9254, 0.8655 < 0.9254 | arithmetically true, but compares an external CI to an internal POINT estimate of a different predictor on a different population — not a test; README's "within or near the internal range" leaves "internal range" undefined (GDPH 0.9154 is below the pooled 0.9254 and inside the per-fold 0.913–0.953) — ambiguous |
| report §3.7:126 "中位恢復比例近 1.0" | M1 recovery medians at k*: GDPH(10) 1.16, SYSUCC(10) 0.85, BrEaST(20) 1.10, BUSI(20) 1.04; at k=10: 1.64/1.43/1.16/0.85 | loosely true at k*; note values > 1 mean the local threshold overshoots the oracle's specificity by giving up sensitivity — "近 1.0" hides that |
| report §3.8:129 "M2 temperature 轉移效果甚微(BrEaST T 2.27 / SYSUCC T 2.23,與內部差 0.03–0.04)" | `v2_cross_site.csv`: M2 row spec deltas vs internal ≤ 0.026 (BrEaST) / ≤ 0.027 (SYSUCC); the T values differ from 2.3644 by 0.09 and 0.14 | the "0.03–0.04" is the specificity difference (RESULTS.md:586,:595 "within 0.03/0.04 of the internal row"), but the sentence reads as a T difference — ambiguous wording |
| README:31 / report:22 internal CV 0.9307 ± 0.0161 (SD over 5 folds) | SD is descriptive, no CI claimed | ✓ labelled best-epoch, optimism quantified |
| README:34 / report:22 nested 0.900 ± 0.057 / 0.780 ± 0.132 | 5-fold SD, no CI | ✓ |
| project_summary:16, script:36 "held-out sensitivity 於 3/5 折低於 0.90" | `posthoc_nested_threshold` 0.9426/0.9760/0.8500/0.8487/0.8843 | supported |
| summary:35, script:26/45 "k=10 sensitivity median 0.83–0.85" | BrEaST 0.830, BUSI 0.846; GDPH 0.866, SYSUCC 0.909 | summary:35 and script:45 qualify "(BrEaST/BUSI)"; **script:26 states 0.83–0.85 with no qualifier** while README/report say 0.83–0.91 — the interview opening under-reports the range (minor) |
| report abstract:22 "sensitivity 皆 ≥ 0.918" | BrEaST 90/98 = 0.9184 | supported |

**Claims resting on a CI that crosses zero:** only the LOCO ΔAUC on BrEaST and SYSUCC. In every document the "MET" statement is accompanied in the same paragraph by "2/4 paired ΔAUC CIs include zero". No document now asserts non-overlap or significance where the CIs do not support it (the prior audit's two unsupported sentences — "前三世代 CI 不重疊", "Reader1 顯著高於" — are gone).

**Residual statistical weaknesses (not mis-statements):**
1. The Q1 verdict is a point-estimate rule by registration; with 2/4 paired CIs including zero, "4/4" overstates the evidential strength even though it is mechanically correct. The documents say so; a reviewer will still object that a pre-registered criterion should have been CI-based.
2. Δspec CIs "exclude zero" at different thresholds per model — the comparison is confounded by threshold placement (disclosed), so the CI is not a test of the training intervention.
3. Internal-vs-external AUC comparisons use an internal point estimate with no CI (0.9254) — the internal ensemble has no unbiased estimate at all (RESULTS.md:168-171).
4. Image-level bootstraps on 3/4 external cohorts (disclosed) likely narrow every external CI; no sensitivity analysis of the effect exists.

**Verdict: PASS on units and on language-vs-evidence (every "significant/non-overlapping/met" sentence is now backed or caveated in the same paragraph). CONCERN — minor — for the interview script's unqualified "0.83–0.85" and missing different-thresholds caveat, the ambiguous "0.03–0.04" in report §3.8, and the undefined "internal range" in README.**

---

## 8. Overstatement scan (verbatim quotes; reason; source line checked)

Rule applied: a sentence is listed if a hostile reviewer could call it overstated, ambiguous or unsupported, even when the repository elsewhere qualifies it. Items marked ★ are new (not in the 2026-09-06 audit).

**README.md**
1. :5-7 "with a frozen, internally pre-registered, single-shot external validation across four cohorts on three continents" — factually true (Poland/Egypt/China + Brazil training); "three continents" is a rhetorical framing of what is, for two cohorts, one country and one image source (HoVer-Trans release). Ambiguous but not false.
2. :53-55 "discrimination transfers reasonably (BUSI and GDPH within or near the internal range; BrEaST and SYSUCC lower, both CIs below the internal 0.9254)" — "internal range" undefined; GDPH 0.915 is below the pooled 0.9254; comparing external CIs to an internal point estimate is not a test (RESULTS.md:190-193 uses the same words). Ambiguous.
3. :57-59 "the errors are false positives, not missed cancers — whether that direction is 'safe' was not assessed" — "not missed cancers" is too absolute: 8/98, 7/163, 11/375, 50/724 malignant images WERE missed (RESULTS.md:201-202); the sentence means "the shift adds FPs, not FNs". Ambiguous.
4. :60-62 "re-selecting only the threshold on 10–20 local labels recovers near-oracle specificity in the median (pre-registered k*)" — supported (RESULTS.md:449-476) but "near-oracle" hides that median recovery > 1 at k=10 comes with median sens 0.83–0.87, i.e. the threshold overshoots the oracle by sacrificing sensitivity; the sensitivity cost follows in the same sentence. OK with caveat.
5. :65-66 "Multi-source LOCO training met its pre-registered criterion via the AUC branch (ΔAUC +0.013 to +0.056 vs the v1 fold-5 single model on 4/4 held-out cohorts)" — mechanically true; the four caveats follow. A reviewer will still note the criterion was point-estimate based. Acceptable as written.
6. :79-82 "Internally pre-registered protocols (external protocol + 4 amendments) committed before the computations they govern" — Amendment 1 contained the data-audit results it registers (stated :193-194); "before the computations" is true for model metrics only. Qualified elsewhere, not here.
7. :83-85 "Freeze-then-test: single-shot external validation, results recorded regardless of outcome (`git tag external-v1`); each prediction CSV has exactly one adding commit" — rests entirely on local rewritten history (README:187-192 admits it). Fine as long as §3 of this audit re-confirms.
8. :86-90 "12.06M-pair perceptual-hash sweep (all within- and cross-set pairs among 1875/252/379/846/1559 images …)" — arithmetic re-derived: Σ C(n,2) = 3,432,028 + Σ n_a·n_b = 8,624,477 → 12,056,505 ✓. But the basis is inconsistent: BUSI enters as its 379-image keep-list while GDPH/SYSUCC enter raw (846/1559), and no log records the sweep actually comparing all within-BUS-BRA pairs (§5 worker to confirm). ★ The count is a derivation from set sizes, not a logged measurement — README presents it as a measurement.
9. :89-90 "Result: no near-duplicate at pHash d ≤ 8 — not 'zero overlap'" — supported (protocol (g)); correctly weakened.
10. ★ :95-96 "Every number in RESULTS.md recomputes from committed CSV/JSON artifacts" — FALSE as stated: the five per-fold CV AUCs at fixed epoch (RESULTS.md:1112-1122) come from local wandb datastores that are git-ignored; the 1-epoch smoke val AUC 0.8434 (RESULTS.md:751), USFM coverage 149/150 (RESULTS.md:716), epoch times, CPU latencies (RESULTS.md:29, :360, :427), the "~16% of images with drop > 0.2" occlusion tail (RESULTS.md:47-49 — `occlusion_border15.csv` exists, so this one is fine), MPS-vs-CPU deltas (RESULTS.md:747) and the training-side val metrics of each LOCO run (RESULTS.md:880-882 etc.) are not recomputable from anything committed. The claim should read "every number in the result tables".
11. :96-99 "the deployed demo shares the same model, calibration and inference module as the validation pipeline, while its preprocessing differs from the validation path on non-BUS-BRA image sizes (documented; impact on calibrated probability ≤ 7×10⁻³)" — accurate (RESULTS.md:1079-1091).
12. :113 repository map "RESULTS.md every number, chronological, commit-linked" — "commit-linked": RESULTS.md cites wandb run ids and tags, not commit hashes, for most sections; only Amendment sections cite commits. Loose.
13. ★ :206 "CITATION.cff and .zenodo.json prepared; Zenodo DOI pending upload." vs :242-245 DOI badge and "deposition of tag v1.0-audited" — the README contradicts itself (stale P4 bullet left in place after the P4 deposition). Minor but visible.
14. :233-238 Limitations — adequate.

**docs/report.md**
15. ★ :18 (abstract 背景) "深度學習模型之跨機構可攜性——尤其 operating point 層級——鮮少被嚴謹評估" — literature claim, no citation.
16. ★ :31 "台灣 55 歲以下女性逾八成乳房攝影屬不均質或極度緻密(Chang et al.)" — author-year citation with NO reference list anywhere in the report (the report has no bibliography); "緻密乳房中超音波敏感度顯著優於攝影" uncited; "與文獻報告相符" (κ) uncited.
17. ★ :34 (§1.2) "既有文獻多以單一資料集之影像層級隨機切分報告 AUC 0.93–0.98 … 外部驗證研究普遍顯示 AUC 降至 0.85–0.88" — two numeric literature ranges with zero citations; "BUSI 之重複影像已見諸文獻" uncited. These numbers violate the report's own P1 rule (every factual sentence traces to RESULTS.md/protocol/CSV) — they trace to nothing in the repo.
18. :44 (§2.1) "官方 patient-level 五折切分(資料集發布之 `5-fold-cv.csv`,存於 git-ignored 之 data/raw/,不在版本庫內…)" — ★ STALE and now FALSE: `data/splits/busbra_official_5fold.csv` is tracked (README:119-123, P3). Same stale statement at :83 (§2.9 "不在版本庫內者:… BUS-BRA fold 檔") and :172 (§4.5 "v2 檢查點與 fold 檔未公開").
19. ★ :83 (§2.9) "v2 之九個檢查點 … 與 BiomedCLIP 匯出權重尚未公開,Q1/Q2 目前僅能自已存預測 CSV 層級重現" — STALE/FALSE: README:127-131 and models/CHECKSUMS.txt say the v2 weights are on HF under v2/ (P3). The report and README now contradict each other on what is public.
20. ★ :90 (§2.10) "公開時間戳(Zenodo/OSF)列於後續工作" and :175 (§4.6) "(6) Provenance:v2 檢查點與 fold 檔公開、Zenodo/OSF 外部時間戳" — listed as future work, while README:242 and project_summary:5 cite the Zenodo DOI as done. Internal contradiction; the report title says "v2.1 依稽核修訂" but was not re-synced after P3/P4.
21. :44 "病理確診 722 良性 / 342 惡性病例" with "以下資料集描述取自該發表論文" — dataset facts from the source publication; permitted, but the report gives no bibliographic entry for Gómez-Flores 2024 either.
22. :67 (§2.5) "四世代以單一指令一次評分 … 結果直接入帳" — supported by RESULTS.md:172-176; the "未記錄之 session 內觀察" clause is honest.
23. :100 (§3.2) "註:ensemble 無無偏內部估計 … 外部驗證為其首次考試" — good; but this undercuts the abstract's presentation of pooled-OOF 0.9254 as the ensemble's AUC (it is not — OOF uses one member per image).
24. :117 (§3.4) "*未檢定之詮釋:漂移幅度可能與資料集風格差異有關*" — labelled; fine.
25. ★ :123 (§3.6) "外部 FP 7/8 熱區落於病灶本體(SYSUCC 四張皆低回音分葉狀良性)" — the supporting figure `reports/gradcam_external_fp.png` is now untracked (`.gitignore`, confirmed MISSING in the clone); the sentence admits the BrEaST replacement "其閱讀未納入上述 7/8 之陳述". The claim is therefore unverifiable from the distributed repository — the report should say so in the sentence, or drop the 7/8.
26. :126 (§3.7) "中位恢復比例近 1.0(GDPH spec 0.453 → k=10 中位 0.827,oracle 0.775)" — the example itself shows recovery 1.16 (> oracle), i.e. overshoot; "近 1.0" is generous; SYSUCC k=10 is 0.85.
27. :129 (§3.8) "M2 temperature 轉移效果甚微(… 與內部差 0.03–0.04)" — reads as a T difference; it is a specificity difference (see §7).
28. :150 (§3.11) "GDPH 判別力提升最明確(AUC CI 不重疊、唯一 LOCO sens 高於 v1-single 之世代)" — supported.
29. ★ :160 (§4.1) "惟增益之大宗仍為「多場域 validation 產出更佳之起始閾值」" — asserted, not tested; the only evidence is the BrEaST mechanism paragraph (RESULTS.md:1035-1040: benign median barely moves while the threshold moves +0.21) — for BUSI/GDPH RESULTS.md itself says the gain is distribution/ROC driven. Generalising "大宗" to all four is the author's reading, not a result; project_summary:38 and interview_script:26/48 repeat it as "the gain is largely threshold placement" without the "未檢定" label they apply elsewhere.
30. :160 "*未檢定之詮釋:BUS-BRA 之良性影像相對「乾淨」致閾值系統性偏低*" — labelled; fine.
31. :163 (§4.2) "模型的假陽性 … 對該判讀者不成立,對 reader2 僅部分成立;未做任何檢定" — fine.
32. :24 / :180 "**部署之最後一哩為以本地資料設定 operating point**,此為本研究最有支撐之結論" — the best-supported *external* result is the threshold-transfer failure (spec 0.41–0.63 at held sens); "set the operating point locally" is the recommended remedy, and the study's own evidence that local setting works is median-only with k_reliable unreached on 3/4 cohorts and the POST-HOC nested analysis showing sens < 0.90 on 3/5 internal folds even with 1500 fitting images. "最有支撐" is arguable: what is best supported is that the frozen threshold does not transfer, not that 10–20 local labels fix it.
33. :180 (§5) "本研究同時提供公開資料集稽核之實務範式" — self-assessment; harmless but unsupported by any external uptake.
34. :34 "(BUSI 之重複影像已見諸文獻)" — uncited.

**docs/project_summary.md / docs/interview_script.md (new documents, 2026-09-07)**
35. ★ summary:5 and script:54 "程式碼、權重、fold 檔與 DOI 全部公開" / "DOI 10.5281/zenodo.22630912" — consistent with README, inconsistent with the report (items 18-20). Whether the DOI resolves is checked by the parent (§PROVENANCE).
36. ★ script:26 "k=10 的 sensitivity 中位數 0.83–0.85" (unqualified) — BrEaST/BUSI only; README/report state 0.83–0.91.
37. ★ script:51 "稽核在乾淨 clone 裡重算了所有東西 … 全部逐位重現" — the audit reproduced the tables; it did not reproduce training, and it stated the per-fold CV AUCs only from CSVs. "所有東西" is rhetorical.
38. ★ script:51 "逐句修正文件、一個數字都沒改" — P2 added new numbers (post-hoc tables) and the Errata re-stated four; no existing result number changed — acceptable if the listener understands "changed" ≠ "added".
39. summary:36 / script:45 abstention "信號本身在三個世代 AUROC 0.77–0.86" — point estimates; BrEaST CI overlaps margin's. Fine as descriptive.
40. script:16 "規則在看到數字前寫進版本控制" — true for model metrics; Amendment 1 contained data-audit results (disclosed elsewhere, not here — a 3-minute talk can't carry every caveat, but this is exactly the sentence a hostile examiner will probe).

**Three pre-registered negative results — original criteria vs reporting (re-checked in all four documents):**
- **CoarseDropout.** Original rule (plan.md:375-378, entered a14990f before the first cdrop run per PROVENANCE §4): "adopt only if pooled OOF AUC >= (current ViT OOF AUC - 0.01) AND the saliency check shows visibly reduced caliper-adjacent heat on malignant TPs. Otherwise discard, no iteration". Reported: RESULTS.md:61-77 (both gates, margin 0.0008, 1/3 improved, discarded); report §2.2:47 quotes the AND rule and §3.1:97 gives the numbers, §4.3:166 adds "三張影像之主觀判斷,未測任何 robustness 指標"; README:91-92; summary:47 ("gate 2 未過即棄"); script:60. **Faithful.** Residual: calling it a "robustness negative result" (README:91 heading "Three pre-registered negative results") is generous — gate 1 passed; what failed was a subjective visual gate; no robustness metric was measured (README says so).
- **Abstention.** Original criterion (protocol (q), three clauses) quoted verbatim in RESULTS.md:655-660 and report §2.7:75; outcome 0/4 with the per-clause table RESULTS.md:663-680; report §3.9:132, §4.3:166; README:92; summary:36; script:45. **Faithful**; reframing labelled post-hoc.
- **BiomedCLIP.** Original criterion (protocol (u)) quoted RESULTS.md:846-859 and report §2.7:77; NOT CLAIMED with 0/4, 1/4 and the 0.700163 fail everywhere (README:93, report abstract:22/§3.10:135/§4.1/§4.3/§5, summary:37, script:48). **Faithful** in all four documents now (the prior audit's abstract contradiction is gone).
- Count: README "Three", report §4.3 heading "三個", summary "三個", script C "三個負面結果" — consistent.

**P1 traceability sample (30+ factual report sentences traced):** §2.1 fold sizes 376/385/366/365/383 → RESULTS.md:84-90 ✓; prevalence 607/1875 → operating_point.json ✓; §2.2 lr/epochs/warmup → RESULTS.md:8,:157 ✓; best_epoch 29/10/18/16/19 → cv_vit_summary.csv ✓; §2.3 table prevalences 0.389/0.430/0.463/0.715 → RESULTS.md:184-188 ✓; §2.4 counts → protocol (e),(g) ✓; §2.5 self-check value → external_selfcheck.txt ✓; §2.7 k grid, R, seed, k*, criteria → protocol (j)-(m),(q),(t),(u) ✓; patience 7 → configs/v2_loco.yaml ✓; §2.8 99/0/309/1013, 0.0072/0.0008, case038/case200 → RESULTS.md:1083-1089 ✓; 1.07e-7 → RESULTS.md:420 ✓; §3.1 0.9304/0.9307/Δ0.0002, 40 vs 376 ms → RESULTS.md:23,:29-30 ✓; ~16% → RESULTS.md:47-49 ✓; cdrop numbers → RESULTS.md:66-71 ✓; 32/121 → posthoc_overconfidence.csv ✓; §3.2 all → RESULTS.md:113-146, posthoc_nested_threshold ✓; §3.3 confusions → RESULTS.md:201-202 ✓; §3.4 medians → RESULTS.md:218-224 ✓; §3.5 reader values, κ, concordance → RESULTS.md:249-270, posthoc_reader_concordance ✓; §3.7 oracle thresholds/specs, k=10 sens, degenerate %, fit-fail 7–20% → RESULTS.md:449-476, methods_summary.csv ✓; §3.8 thresholds/T → v2_cross_site.csv ✓; §3.9 all → abstention CSVs ✓; §3.10 all → RESULTS.md:783-859 ✓; §3.11 table → v2_loco_q1c_table.csv ✓ (verified numerically above); §3.12 Dice/IoU/280/383, 9.7 s → RESULTS.md:329-337,:427 ✓. **Untraceable:** §1.1 Chang et al. and "顯著優於攝影"; §1.2 both AUC ranges and "BUSI 重複影像已見諸文獻"; §1.1 "與文獻報告相符"; §3.6 "7/8" (figure not distributed); §4.1 "增益之大宗仍為閾值定位" (interpretation, unlabelled). Stale-false: §2.1/§2.9/§4.5 fold-file and v2-weights statements; §2.10/§4.6 Zenodo as future work.

**Verdict: CONCERN — major.** The results-bearing sentences of README and report are now tightly sourced and every P1 correction from the prior audit holds (re-verified: 8/8 §2 items, 10/10 §7 rows, 29/29 §8 items no longer present in their original form). What remains is of a different kind: (a) the report was NOT re-synchronised after P3/P4 — it states in three places that the fold file and v2 weights are unpublished and in two places that the Zenodo timestamp is future work, contradicting README, project_summary and PROVENANCE; (b) README contradicts itself on the DOI (:206 vs :242); (c) README:95 "every number recomputes from committed artifacts" is false for the wandb-derived and latency numbers; (d) the report's §1 carries four uncited literature claims, violating its own P1 rule; (e) one qualitative result (7/8 external FP CAMs) is no longer verifiable from the distributed repo; (f) the "gain is largely threshold placement" reading is presented as fact in three documents without the "未檢定" label used elsewhere. Items (a) and (b) are the kind of contradiction a reviewer finds in five minutes.

---

## 9. Reproducibility from scratch (README "Reproduce", followed literally on this clone)

| README step | Outcome (fresh overlay, 2026-09-11) | Deviation needed |
|---|---|---|
| `uv venv --python 3.12 && source .venv/bin/activate` | uv used CPython 3.12.13 (system python3 is 3.14.6) | venv created in the scratchpad overlay, not the read-only clone |
| `uv pip install -r requirements.txt` | "Resolved 181 packages in 1.67s", 181 installed, exit 0; torch 2.13.0, timm 1.0.28, opencv-python-headless 5.0.0.93, albumentations 2.0.8, scikit-learn 1.9.0, imagehash 4.3.2, numpy 2.5.2, pandas 3.0.5; MPS available | none; `openpyxl` and `open_clip_torch` are still unpinned (requirements.txt:175-176) |
| "Dataset layout (hardcoded paths)" block | matches what the code opens (`data/raw/{busbra,breast_poland,busi,gdph_sysucc}`, the two xlsx names); `data/raw` is git-ignored, so I symlinked the five raw directories from `$DATA_ROOT` | none beyond obtaining the data; GDPH/SYSUCC still have no licence file (README says so) |
| `python src/external_val.py --self-check` | `reproduced 0.9234433158791243`, "SELF-CHECK PASSED", stdout **byte-identical** to `reports/external_selfcheck.txt` (`diff` exit 0); ran on MPS with `models/cv_vit_fold5.pt` present (symlink) | none; with the file absent it would download from HF (README says so) |
| "`pytest` runs the 8 split tests (all five raw datasets required)" | `python -m pytest -q tests/` → **15 passed in 36.80s** | README:155 is stale: 15 tests (5 splits + 5 v2 splits + 3 checkpoint-payload + 2 HF-path); the two helper-only test files do not need data |
| "Regenerating frozen-v1 from scratch (approximate)" chain | not run (training forbidden); the four commands exist with the stated defaults (`cross_validate.py --prefix`, `tta_eval.py`, `calibrate.py`, `pick_threshold.py`); README states non-determinism and `WANDB_MODE=offline` | none needed in text; `train.py` still has no offline flag of its own |
| v2 reproduction | prediction CSVs recompute (§2); the nine v2 checkpoints + BiomedCLIP export are now on HF under `v2/` with sha256 in `models/CHECKSUMS.txt` (28/28 verified locally and against HF LFS, §6) — but **docs/report.md §2.9:83 and §4.5:172 still say they are unpublished** | README is right, the report is stale |
| `scripts/verify_space.py` (interview checklist) | with no argument it calls `HfApi().whoami()` and needs an HF token; `--space happytommy/breast-us-cad` runs anonymously; result: max Δ 1.07e-7, warm median 7.18 s | undocumented `--space` flag needed for a third party |
| `scripts/consistency_sweep.py` | reproduces `docs/consistency_check.md` byte-for-byte | none (see the sweep section for what it does not prove) |
| `src/posthoc_epoch_selection.py` (P2 table) | reproduces `reports/posthoc_epoch_selection.csv` byte-for-byte **only because the author's git-ignored `wandb/` was symlinked in**; from the published repository this table cannot be regenerated | not stated in README (only in the script docstring) |
| CLAUDE.md eval command | `python src/evaluate.py --ckpt models/cv_vit_fold5.pt --split val` — valid now | fixed since the prior audit |

**Verdict: PASS for what README promises about inference (environment builds from the pins, tests and the self-check pass digit-for-digit in a fresh environment, v1 and v2 weights are published with checksums). CONCERN — minor — for the stale "8 tests", the undocumented `--space` flag, `train.py` still lacking an offline switch, and the post-hoc epoch table being reproducible only from the author's wandb directory.** (Frozen-v1 checkpoints themselves remain non-regenerable by design of MPS training; README now says so.)

---

## 10. Code-quality risks (re-audit)

- **Hardcoded paths** (CLAUDE.md rule 5 "no hardcoded paths"): 17 `data/raw/...` literals remain in code (`src/phash_sweep.py:42,51,52`, `src/birads_comparison.py:20`, `src/phash_tables.py:27`, `src/dedup_busi.py:32-33`, `src/external_val.py:42`, `src/v2_load_usfm.py:33`, `src/gradcam_check.py:94`, `src/data.py:33,54` defaults, tests). `models/...` literals in 29 files (e.g. `src/external_val.py` ×6, `src/v2_loco.py` ×7, `src/v2_biomedclip_*` ×7). Only `src/inference.py:45-46` is env-overridable (`BREAST_US_CAD_MODELS_DIR`, `BREAST_US_CAD_WEIGHTS_REPO`). No `/Users/...` or author-name paths in tracked code (grep clean); wandb metadata does show `--ckpt-out /Users/thomaswang/.claude/jobs/...` for smoke runs only. Unchanged since the previous audit — minor.
- **Exception handling** (`grep -rn '^\s*except' src scripts app deploy`): 5 blocks — `src/v2_recalib_curve.py:135 except RuntimeError` (declared M2 fit-failure fallback), `scripts/verify_space.py:65,92,110 except Exception` (print + retry/skip, network client), `scripts/consistency_sweep.py:144 except ValueError` (float parse). No bare `except`, none silent. OK.
- **Nondeterminism:** `train.py:25-28` seeds python/numpy/torch only; no `torch.use_deterministic_algorithms`, no DataLoader `generator`/`worker_init_fn`; `num_workers: 4` in every config; `PYTORCH_ENABLE_MPS_FALLBACK` set nowhere in code (CLAUDE.md says shell env). Training therefore not bitwise reproducible (README now states this). All analysis-side RNGs are seeded (`default_rng(42)` / seed lists in `pick_threshold.py:88`, `external_val.py:110`, `v2_data.py:60`, `v2_recalib_curve.py:175,269`, `v2_disagreement.py:90`, `v2_loco_report.py:103`). `src/compare_backbones.py:47 torch.manual_seed(0)` for latency timing only. Minor (training), OK (analysis).
- **Stale / confusing code:** `src/evaluate.py:7-9` docstring still promises external loaders "when that time comes" and `--split` still has one choice (`:103`); `src/cross_validate.py:49` default `--prefix cv_effb0`; `src/compare_backbones.py:96-97` appends to `RESULTS.md` on every run (its own docstring says so); `app/app.py:25` / `deploy/app.py:25` `os.chdir(HERE.parent)` at startup; `scripts/deploy_hf.py:90-95,114-118` **still rewrites `src/inference.py` in place** (`patch_default_repo` on the default weights-repo string; a no-op when the id already matches, but a deploy script mutating the frozen inference source remains a hazard if run under a different HF account). Duplicate definitions across files: `predict_hflip_pair` (evaluate.py, inference.py, deploy/inference.py), `probs_to_logits` (calibrate.py, inference.py ×2), `vit_reshape_transform`/`load_rgb01` (gradcam_check.py, inference.py ×2), `patient_bootstrap` (pick_threshold.py, external_val.py), `sens_spec` ×5 — drift risk for a "frozen" path; deploy/ copies are by design. Minor.
- **Network access:** `inference.resolve_weight` (`src/inference.py:76-86`) downloads from HF whenever `models/<file>` is missing — no offline switch (README now says so); `train.py:168`, `train_seg.py:113`, `v2_loco.py:100` `wandb.init` (online unless `WANDB_MODE=offline`); `v2_export_biomedclip.py:43-46` `hf_hub_download`; `v2_load_usfm.py:182` / `v2_export_biomedclip.py:97` `timm pretrained=True`; `scripts/verify_space.py`, `upload_v2_weights.py`, `deploy_hf.py` use `HfApi`/`gradio_client` (credentials via the HF cache/`whoami`, no tokens in code — grep for token/api_key/password clean). Nothing reads `data/real` (path does not exist; only the archived audit mentions it). Minor.
- **Scripts writing into the repo:** `scripts/upload_v2_weights.py:67` writes `models/CHECKSUMS.txt` and `:86` `reports/_weights_card.md`; `deploy_hf.py:100-101` copies `app/app.py`/`src/inference.py` into `deploy/` and `:127` writes the card; `compare_backbones.py` appends to RESULTS.md; `posthoc_epoch_selection.py:120` writes `reports/posthoc_epoch_selection.csv`. All intentional, documented in docstrings.
- **pyplot backend:** 10 of 19 pyplot importers call `matplotlib.use("Agg")`; `src/gradcam_check.py:16`, `backbone_diagnostic.py:18`, `plot_prob_shift.py:18`, `birads_comparison.py:16`, `adoption_eval.py:15`, `explain.py:30` do not (`posthoc_reader_concordance.py:19` works around birads_comparison via `MPLBACKEND`). Headless-failure risk only; minor.
- **requirements.txt:** 174 `==` pins; `openpyxl` and `open_clip_torch` (lines 175-176) remain unpinned — unchanged since the previous audit; `imagehash==4.3.2`, `opencv-python-headless==5.0.0.93`, `torch==2.13.0`, `timm==1.0.28` pinned.
- **Notebook:** `notebooks/01_eda.ipynb` — 5 executed cells with outputs; grep for `breast_poland|busi|gdph|sysucc|external` returns nothing.

**Verdict §10: CONCERN — minor.** No item changes a number. The previous audit's §10 list is essentially unchanged in code (hardcoded paths, duplicated frozen-path functions, `deploy_hf.py` mutating `src/inference.py`, unpinned `openpyxl`/`open_clip_torch`, missing Agg in six plotting scripts, no offline switch for weight download); the one code change made in response (`checkpoint_payload`) was applied to `train.py` but not to `v2_loco.py`. Evidence I would have expected for a FAIL — a bare `except: pass`, a `/Users/...` literal in tracked code, a script reading an external cohort into training, or credentials in code — was searched for and not found.

---

## PROVENANCE.md claim-by-claim

| § | Claim | Check | Result |
|---|---|---|---|
| 1 | private throughout, public since 2026-09-07 | GitHub API `private:false`, `pushed_at` 09-07 08:15Z; audit of 09-06 saw 404 | consistent (visibility history itself is not exposed by the API) |
| 1 | protocols internal only, not externally time-stamped | no OSF/Zenodo record before 09-07 (Zenodo `created` 09-07 07:30Z) | TRUE |
| 1 | identity rewrite 08-31, dates/trees/messages preserved | 11 pairs checked by tree hash + author date; fsck count 31 | TRUE |
| 1/5 | HF weights repo 2026-08-30 04:25:49Z, after external-v1 15:14:26Z | HF API | TRUE |
| 2 | tag table (commit hashes, dates) | every row matches `git log`/`git for-each-ref` | TRUE; `v1.0-audited` row shows "2026-09-07" with no time — tag object says 13:05:16 |
| 2 | "62 commits before the P4 commit" | `git rev-list --count 1f196a9^` = 62 | TRUE |
| 2 | 6c180ee 6-second amend; 8e058bd pre-amend draft unreachable | committer − author = 6 s; 8e058bd exists in author copy, tree differs | TRUE |
| 3 | 30 rewritten commits, 29 e-mails changed, d265b55 rewritten only for parent | d265b55 already gmail (checked) | TRUE |
| 3 | `git fsck --unreachable --no-reflogs` lists 31 commits | 31 (with `LC_ALL=C`) | TRUE, but reproducible only from the author's disk (stated) |
| 3 | wandb hashes d0cd52c/51e237c/1a02bd6/c9adedd/82bd016/402a193 map to 2d38e90/5e78373/0ae8efc/0c23853/a14990f/b213aa1 | tree-hash match for all six | TRUE (wandb dirs themselves checked by the §1 worker: 46 runs) |
| 4 | protocol append-only; RESULTS external-v1 never edited | 0 removed lines in every protocol commit; 0 removed lines in RESULTS.md since 2cbe1da | TRUE |
| 4 | Amendment 1 already contained the dedup counts | `git show 436ab56 -- data/external_protocol.md` contains 846→810 / 1559→1013 | TRUE |
| 4 | CoarseDropout rule 4 min before first cdrop run | a14990f 22:31:00; wandb run 22:35 (from author wandb dirs, §1 worker) | TRUE |
| 5 | all HF timestamps | HF API today | TRUE to the second |
| 5 | CHECKSUMS.txt verified against HF LFS metadata 28/28 | see §6 (worker) | see §6 |
| 7 | archive = `git archive --format=zip --prefix=breast-us-cad-v1.0/ v1.0-audited`, sha256 bb45b685…, 26,404,667 B, Zenodo MD5 07a7c1fe… | regenerated from this clone: **sha256 bb45b685fa80…182c1e, md5 07a7c1fe4f4daf15338d51ea9620d067, 26,404,667 bytes**; Zenodo file `breast-us-cad-v1.0.zip` size 26404667, `md5:07a7c1fe4f4daf15338d51ea9620d067` | TRUE — the deposited file is bit-identical to `git archive` of fa193a8 |
| 7 | superseded archive fcd270a5… (1f196a9) never uploaded | `git archive` of 1f196a9 → fcd270a5f6f9…; Zenodo record has exactly one file (the bb45… one) | TRUE |
| 7 | tagged commit fa193a8; re-pointed from 1f196a9 before any push/deposition | local/GitHub tag object 2e7da4d → fa193a8, tagger 13:05:16 = fa193a8's commit time; Zenodo created 15:30Z-8 = 15:30 local, push 16:15 local | TRUE |
| 7 | "an archive cannot contain its own hash, so this line lives in the commit after the tagged one" | TRUE — **but incomplete**: the deposited archive's own `docs/PROVENANCE.md` §7 says "Tagged commit: `1f196a9…`", "Archive SHA-256: `fcd270a5…`", "Zenodo DOI: pending", and its README.md:240–242 says "Zenodo DOI: pending (archive of tag `v1.0-audited` = commit 1f196a9, sha256 `fcd270a5f6f9e30e…`)"; its CITATION.cff has no `doi:` field. A reader who downloads the Zenodo file and follows its own PROVENANCE §7 will compute bb45… and find it does not match the fcd2… the file claims for itself, and will look for commit 1f196a9 which the archive is not. | **CONCERN — major** (provenance document contradicts the artifact it ships in; nothing in the Zenodo record description or in HEAD's PROVENANCE.md warns of this) |
| 7 | .zenodo.json in archive predates ORCID (063a25a) | archive .zenodo.json has no orcid | TRUE (disclosed) |
| 1/7 | DOI recorded in CITATION.cff / .zenodo.json / README / PROVENANCE | HEAD files: yes; **README.md:206 still says "Zenodo DOI pending upload"** while README:242–247 states the DOI; plan.md:151 heading "(prepared; DOI pending)" then "DONE … published" | minor internal inconsistency |
| 6 | forward commitment (OSF, no more rewrites, Zenodo per milestone) | a promise; not checkable | — |

**Verdict on PROVENANCE.md: PASS on every checkable factual claim (hashes,
dates, counts, HF and Zenodo metadata all reproduce). CONCERN — major — for
the self-referential defect in §7: the deposited archive describes itself as
a different commit with a different hash and "DOI pending".**

---

## Zenodo record 10.5281/zenodo.22630912 vs tagged commit and archive

`curl https://zenodo.org/api/records/22630912`: status published, concept DOI
10.5281/zenodo.22630911, version 1.0-audited, licence apache2.0, creator Wang
Thomas (ORCID 0009-0009-3066-3298), created 2026-09-07T07:30:33Z, one file
`breast-us-cad-v1.0.zip`, 26,404,667 B, `md5:07a7c1fe4f4daf15338d51ea9620d067`;
related identifiers: GitHub (isSupplementTo), HF weights + Space
(isSupplementedBy), cites 10.1002/mp.16812 and 10.5281/zenodo.8231412.

Local: `git archive --format=zip --prefix=breast-us-cad-v1.0/ v1.0-audited >
zip` → size 26404667, sha256 `bb45b685fa807fa14e83f01271f5256ab4cc933d5c43f53b176dd19b5b182c1e`,
md5 `07a7c1fe4f4daf15338d51ea9620d067`. `v1.0-audited` → fa193a8 (annotated,
tag object 2e7da4d, same on GitHub). **The Zenodo file is the archive of
fa193a8; PROVENANCE §7's sha256, size and MD5 prefix are all correct.** The
only defects are those in the table above (archive-internal metadata names
1f196a9/fcd270a5/"pending"; the Zenodo description does not mention it).

---

## consistency_sweep.py re-run

`python scripts/consistency_sweep.py` in the overlay (fresh venv) → "shared
values: 156; unverified: 0"; `diff` against the committed
`docs/consistency_check.md` → **IDENTICAL** (byte-for-byte, 192 lines).

But the check is weaker than its "147 verified" suggests. It calls a number
verified if the same token, or any source number that rounds to it, occurs
anywhere in RESULTS.md, any aggregate CSV, any JSON or `models/CHECKSUMS.txt`,
with no semantic link. Examples from the committed table:

| doc value | meaning in the document | "source" the sweep cites | what that line actually is |
|---|---|---|---|
| `55` | "台灣 55 歲以下女性" (report §1.1) | models/CHECKSUMS.txt:12 | a sha256 hex string containing "55" |
| `45` | "各約 45 秒" (interview beat timing) | external_protocol.md:21 | BrEaST normal-case ID 45 |
| `4.0` | "CC BY 4.0" | RESULTS.md:27 | effb0 parameter count 4.0 M |
| `0.5` | k_reliable 2.5th-pct recovery ≥ 0.5 | RESULTS.md:325 | U-Net Dice threshold 0.5 |
| `0.51` | κ (GDPH, true value 0.5149) | RESULTS.md:5 "0.505 rounded" | effb0 baseline Youden threshold 0.505 |
| `0.30` | Δspec upper end (true 0.3011) | RESULTS.md:458 "0.297 rounded" | a k=30 specificity CI lower bound |
| `95` | "95% CI" | RESULTS.md:137 | "95% CIs" (fine) |
| `0.85` | branch-B sens floor | RESULTS.md:467 | a SYSUCC recovery-ratio cell |

So "0 UNVERIFIED" means "no shared number is absent from the union of all
source numbers", which a wrong number that happens to exist elsewhere would
pass. The sweep did catch one real mismatch (latency), and the header states
the rule honestly, but plan.md P6 and docs/consistency_check.md:192 present
"0 UNVERIFIED shared values" as a result. The real cross-document
verification is P1's manual table and §2 of this audit.

**Verdict: PASS on reproducibility of the committed file; CONCERN — minor —
the tool's "verified" status is token coincidence, not verification, and
should not be cited as evidence of correctness.**

---

## 11. Resolution of every item in the 2026-09-06 audit

Legend: **RESOLVED** (evidence re-verified here) · **PARTIALLY RESOLVED** · **STILL OPEN** · **DISPUTED-BY-AUTHOR-CORRECTLY / -INCORRECTLY** · **PERMANENT** (cannot be fixed retroactively; disclosed).

### Audit §1 — split integrity
| Prior item | Author response | Status now | Evidence |
|---|---|---|---|
| Fold file not in repo; CLAUDE.md/data/README/README said it was | P3: committed `data/splits/busbra_official_5fold.csv`, hash-verified loader, docs corrected | **RESOLVED** | file present, sha256 8bed2d2e…, byte-identical to raw; `resolve_fold_file` raises on mismatch; 15/15 tests; README/CLAUDE.md/data/README.md correct — **but docs/report.md §2.1:44, §2.9:83, §4.5:172 still say the fold file is git-ignored / unpublished** (see §8) |
| Checkpoints embed `train_folds=[1,2,3,4]` regardless of k | P2: `checkpoint_payload` stores resolved folds; existing ckpts not rewritten | **PARTIALLY RESOLVED** | wired into `train_one_fold` (train.py:216-226); tests cover only the helper; `src/v2_loco.py:138` still saves raw cfg; all existing ckpts unchanged (disclosed RESULTS.md) |
| Best-epoch-on-the-same-fold selection bias, undisclosed | P1 disclosure + P2 quantification (fixed-epoch 0.9195 ± 0.0164, optimism 0.0111 ± 0.0073) | **RESOLVED (disclosed and quantified)** | README rows (a)/(a′), report abstract/§2.2/§3.1/§4.5, RESULTS.md POST-HOC 1; script re-run reproduces the CSV byte-for-byte; the bias itself is inherent to the frozen artifacts and remains |
| Tests need all raw datasets | README now says so | **RESOLVED** (statement) | README:155; but count "8" is stale (15) |

### Audit §2 — numbers (8 items)
| # | Prior item | Status now | Evidence |
|---|---|---|---|
| 2.1 | "~9.7M-pair" sweep | **RESOLVED** in README/report/protocol (12,056,505, re-derived and re-swept, §5) | only `docs/readme_skeleton.md:38` still says "~9.7M" (a tracked draft) |
| 2.2 | "sens ≥ 0.92 on all four" | **RESOLVED** in plan/app/deploy/report/README (≥ 0.918); RESULTS.md:485 and :1058 corrected by appended Errata only, in-place text still reads "≥ 0.92" | grep; RESULTS.md:1232-1237 |
| 2.3 | report "前三世代 CI 不重疊" | **RESOLVED** | report §3.9:132 now states BrEaST overlaps with the CIs |
| 2.4 | "≤ 10⁻³" vs "7×10⁻³" | **RESOLVED** | report §2.8/§4.5 say 7×10⁻³ / 8×10⁻⁴; grep for "≤ 10⁻³" → 0 |
| 2.5 | "1,879 BUS-BRA images" | **PARTIALLY RESOLVED** | plan.md/report corrected; RESULTS.md:403 via Errata; **`src/inference.py:153` and `deploy/inference.py:153` (and therefore the live Space) still say "1879 images"** |
| 2.6a | "121 malignant / 3" | **RESOLVED** | deleted; sourced 32/121 (POST-HOC 3) in report §3.1 |
| 2.6b | "six-item GO audit" | **RESOLVED** | rewritten as labelled unrecorded observation (report §2.5:67) |
| 2.6c | BUS-CoT | **RESOLVED** | grep → 0 hits outside the audit docs |
| 2.6d | "+0.01 screening gate" | **RESOLVED** | report §2.2:47 no longer states a gate |
| 2.7 | k_reliable "≈100–200" | **RESOLVED in README/report/app/summary/script** (GDPH k=200 only, POST-HOC labelled); **STILL OPEN in RESULTS.md:483** ("bands only become usable at k ≈ 100–200"), which the Errata does not list | grep `100[–-]200` |
| 2.8 | GDPH "within the readers' distribution" | **RESOLVED** | report abstract:22, §3.5:120, §4.2:163: below both readers on both axes; κ inter-reader only |
| new | `models/v2_biomedclip_operating_point.json` sens 0.9012 / tp 547 / fn 60 at thr 0.20070531964302063 | **NEW — STILL OPEN (minor)** | on the committed `v2_biomedclip_oof_preds.csv` the stored threshold gives 546/61, sens 0.8995 (< 0.90 rule); re-picked threshold is 3.7e-9 lower; report §3.10 / RESULTS Q2b inherit "sens 0.901" (§2) |

### Audit §3 — timeline
| Prior item | Status now | Evidence |
|---|---|---|
| Self-certified timeline, rewritten history, no public registry | **PERMANENT, disclosed** | README "Provenance" section, report §2.10, PROVENANCE.md §1/§4; Zenodo (09-07) is the first external timestamp; GitHub public 09-07 |
| Pre-rewrite hashes unmappable | **RESOLVED (documented)** | PROVENANCE §3 table; 11 pairs re-verified by tree hash + author date; 31 unreachable objects still present in the author's copy — and only there |
| HF postdates external-v1 | **PERMANENT, recorded** | PROVENANCE §5 timestamps match HF API to the second |
| Repository private | **RESOLVED** (public since 09-07) | GitHub API `private:false`; tags on GitHub = local |

### Audit §4 — protocol
| Prior item | Status now | Evidence |
|---|---|---|
| Q1 wording exceeds fired branch | **RESOLVED in every document + RESULTS Errata**; in-place RESULTS.md:1028-1033 unchanged by design | §4 table |
| Q2 abstract asserts forbidden claim | **RESOLVED** | NOT CLAIMED in abstract/§3.10/§4.1/§4.3/§5 |
| Abstention reframing | **RESOLVED** (labelled 事後詮釋) | report §3.9/§4.3 |
| CoarseDropout gate subjective | **RESOLVED** (caveat added) | README:91-92, report §4.3 |
| TTA not pre-registered | **RESOLVED** (disclosed) | README:195-197, report §2.2, plan.md, PROVENANCE §4 |
| M2 fit-failure fallback; patience 7 | **RESOLVED** (disclosed as run-time / under-specified) | report §2.7:73,:77; RESULTS.md:499-504,:875 |
| (New) protocol (g) adjudication figure | **NEW — STILL OPEN (minor)** | `reports/phash_cross_pairs.png` untracked in P4c; protocol text still cites it; only a table + BUS-BRA thumbnail is distributed |

### Audit §5 — data audit
| Prior item | Status now | Evidence |
|---|---|---|
| 9.7M count | RESOLVED (see 2.1) | |
| "zero overlap" > method | **RESOLVED** | README:89-90, report §2.4 "pHash d ≤ 8 無近重複…而非零重疊" |
| bus_0012-l/-r within-BUS-BRA d=8 pair not mentioned | **DISPUTED-BY-AUTHOR-CORRECTLY** | same patient, same fold, recorded in `phash_sweep_hits.csv`; no document claims anything about within-BUS-BRA duplicates |

### Audit §6 — parity
| Prior item | Status now | Evidence |
|---|---|---|
| README "exact inference module" | **RESOLVED** | README:96-99 states the preprocessing difference and ≤ 7×10⁻³ |
| Preprocessing not shared | **PERMANENT by design, now stated everywhere** | §6 table |
| app/ vs deploy/ example PNG encodings differ | **DISPUTED-BY-AUTHOR-CORRECTLY**, and the prior observation is not reproducible at HEAD (both md5 a8ebac44…, unchanged since 08-30) | §6(a) |
| Warm latency 6.15 vs 6.8 s | **RESOLVED** (range 6–9 s; today 7.18 s) | RESULTS Errata; §6(e) |
| (New) live Space ≠ deploy/ | **NEW — STILL OPEN (minor)** | live `inference.py`/`README.md` = P1 version (5652f0a); P3 `hf_repo_path` and P4b licence sentence never pushed; RESULTS.md:380 "Space repo verbatim" false at HEAD |

### Audit §7 — statistics (10 rows)
All ten rows: **RESOLVED** — units re-verified in code; every "non-overlapping/met/significant" sentence is now backed or carries the zero-crossing caveat in the same paragraph in README, report, project_summary and interview_script (§7 table). Residuals are new and minor (interview_script:26 "0.83–0.85" unqualified; report §3.8 "0.03–0.04" ambiguous; README "internal range" undefined).

### Audit §8 — overstatement (29 items)
29/29 **RESOLVED in their original form** (each re-grepped/re-read: items 1–11 README, 12–29 report). Three pre-registered negative results reported with original criteria in all four documents. **New overstatement/contradiction items** found in this audit: report not re-synced after P3/P4 (fold file / v2 weights "unpublished", Zenodo "future work" — §8 items 18–20); README:206 vs :242 DOI contradiction; README:95 "every number recomputes from committed artifacts" (false for wandb-derived and latency numbers); four uncited literature claims in report §1; §3.6 "7/8" unverifiable from the distributed repo; "gain is largely threshold placement" unlabelled interpretation in three documents.

### Audit §9 — reproducibility (8 rows)
| Prior row | Status now |
|---|---|
| 9.3 dataset layout undocumented | **RESOLVED** (README block) |
| 9.4 train.py ≠ frozen-v1; wandb login; MPS non-determinism | **RESOLVED** (README chain + statements); `train.py` itself unchanged (no offline flag) |
| 9.5 silent HF download | **RESOLVED** (stated); behaviour unchanged |
| 9.6 tests need datasets | **RESOLVED** (stated); count stale |
| 9.7 v2 checkpoints unpublished | **RESOLVED** (HF v2/, CHECKSUMS 28/28 verified) — except the report still says unpublished |
| 9.8 CLAUDE.md `--split test` | **RESOLVED** |

### Audit §10 — code quality
| Prior item | Status now |
|---|---|
| Hardcoded `data/raw`/`models/` literals | **STILL OPEN** (17 + 29 files; deferred "P2/P3", never done) |
| `deploy_hf.py` rewrites `src/inference.py` | **DISPUTED-BY-AUTHOR-INCORRECTLY** — "no-op when the id already matches" is true only for this account; the script still mutates the frozen source in place (`scripts/deploy_hf.py:114-118`); the hazard the audit named is unchanged |
| Duplicated frozen-path functions | **STILL OPEN** (`predict_hflip_pair` ×3, `probs_to_logits` ×3, `patient_bootstrap` ×2, `sens_spec` ×5) |
| No determinism flags / worker seeding | **STILL OPEN** (disclosed in README) |
| `evaluate.py` stale docstring, `cross_validate` default prefix, `compare_backbones` appends to RESULTS.md, `os.chdir` at import, missing Agg | **STILL OPEN** (6 plotting scripts without Agg) |
| `openpyxl`/`open_clip_torch` unpinned | **STILL OPEN** |
| `resolve_weight` network without offline switch | **STILL OPEN** (disclosed) |

### New items found by this audit (not in the 2026-09-06 list)
1. docs/report.md §2.1/§2.9/§2.10/§4.5/§4.6 stale-false on fold file, v2 weights, Zenodo (§8 items 18–20) — **major** (contradiction between the paper and README/PROVENANCE).
2. Zenodo archive's internal PROVENANCE §7 / README / CITATION.cff name commit 1f196a9, sha256 fcd270a5…, "DOI pending" (PROVENANCE section) — **major**.
3. README:206 "DOI pending" vs README:242 DOI — minor.
4. README:95 "every number in RESULTS.md recomputes from committed artifacts" — false for wandb-derived and latency numbers (§8 item 10) — major as a class with 5.
5. consistency_check.md "0 UNVERIFIED" is token coincidence (sweep section) — minor.
6. RESULTS.md:483 "k ≈ 100–200" unsupported, no erratum — minor.
7. `v2_biomedclip_operating_point.json` off by one image vs its own CSV (§2) — minor.
8. Live Space ≠ deploy/ (P3/P4b never pushed); RESULTS.md:380 false at HEAD (§6) — minor.
9. `src/inference.py:153` / `deploy/inference.py:153` "1879 images" (§6) — minor.
10. Report §1 four uncited literature claims, no bibliography (§8 items 15–17) — minor/major for a paper.
11. Report §3.6 "7/8" external-FP CAM reading no longer verifiable from the distributed repo (§8 item 25) — minor.
12. "Gain is largely threshold placement" stated as fact in report §4.1, project_summary:38, interview_script:26/48 (§8 item 29) — minor.
13. Unrecorded second LOCO-gdph training launch (wandb k896krp6 aborted at epoch 3; restart 3jn4cicy) (§1) — minor.
14. LOCO 85/15 slices not persisted in any artifact (§1) — minor.
15. `v2_loco.py:138` never adopted `checkpoint_payload` (§1) — minor.
16. protocol (g) adjudication figure untracked; protocol text still cites it (§4) — minor.
17. README:155 "8 split tests" (15); `verify_space.py` needs undocumented `--space` for third parties (§9) — minor.
18. `docs/readme_skeleton.md` (tracked) still carries "~9.7M", "safe direction", "≥ 0.92" — minor.

### Audit executive-summary findings
| Prior finding | Status now |
|---|---|
| 1. Internal headline optimistically biased and mixed, undisclosed | **RESOLVED as disclosure** (rows a/a′/b/b′ everywhere; POST-HOC 1–2 quantify 0.011 AUC optimism and the nested operating point). The bias is intrinsic to frozen-v1 and cannot be removed. |
| 2. Report/README overstate beyond RESULTS.md | **RESOLVED for the 2026-09-06 list**; **new contradictions introduced by P3/P4 not being propagated to the report** (this audit §8) |
| 3. Provenance rests on self-certified artifacts | **PERMANENT for external-v1, now fully disclosed** (PROVENANCE.md, README, report §2.10); fold file and v2 weights published; Zenodo DOI + public GitHub give the first external anchors (post hoc). One new provenance defect: the deposited archive's own PROVENANCE §7/README name the superseded commit 1f196a9, hash fcd270a5 and "DOI pending". |

---

## Executive summary

**Three most serious findings**

1. **The published record now contradicts itself on provenance (major).**
   docs/report.md — the paper — was corrected sentence-by-sentence in P1 but was
   not re-synchronised after P3/P4: §2.1:44 and §2.9:83 say the BUS-BRA fold file
   is git-ignored and "不在版本庫內", §2.9:83 and §4.5:172 say the nine v2
   checkpoints are unpublished, §2.10:90 and §4.6:175 list the Zenodo/OSF timestamp
   as future work. README:106–131/242, docs/project_summary.md:5 and PROVENANCE.md
   §7 say all three were done on 2026-09-07 — and they were (fold file committed
   and hash-verified; 28/28 checksums verified against HF LFS; Zenodo record live).
   README additionally contradicts itself (:206 "Zenodo DOI pending upload" vs
   :242 the DOI). Worst, the Zenodo deposit — the one external timestamp — ships a
   `docs/PROVENANCE.md` §7, README.md:240–242 and CITATION.cff that describe the
   archive as commit 1f196a9 with sha256 fcd270a5… and "DOI pending"; the file is in
   fact the archive of fa193a8 with sha256 bb45b685… (verified bit-for-bit: size
   26,404,667, md5 07a7c1fe… equal to Zenodo's). A reader who follows the archive's
   own instructions will conclude it has been tampered with. Nothing in the Zenodo
   description or in HEAD's PROVENANCE.md warns of this.
2. **The repository's self-verification claims are stronger than the
   verification performed (major).** README:95 "Every number in RESULTS.md
   recomputes from committed CSV/JSON artifacts" is false for the fixed-epoch AUCs
   (git-ignored wandb datastores), smoke/latency/epoch-time numbers and LOCO
   training-side metrics. `docs/consistency_check.md` "0 UNVERIFIED" is token
   coincidence — "55" (Taiwanese women under 55) is "verified" by a sha256 line in
   CHECKSUMS.txt, "4.0" (CC BY 4.0) by a parameter count, κ 0.51 by a Youden
   threshold. RESULTS.md still carries five wrong or unsupported sentences in place
   (:403, :483, :485, :1028–1033, :1058), four fixed only by an appendix 200–800
   lines later and one (:483 "bands only become usable at k ≈ 100–200") not at
   all. `models/v2_biomedclip_operating_point.json` records sens 0.9012 at its
   frozen threshold, but that threshold applied to the committed OOF CSV gives
   0.8995 — the v2 freeze violates its own ≥ 0.90 rule by one image. The report's
   §1 makes four literature claims with no citation and no bibliography, against
   its own P1 rule, and presents "the gain is largely threshold placement" as a
   finding in three documents without the "未檢定" label it uses elsewhere.
3. **Pre-registration remains self-certified and the deployed/derived artifacts
   have drifted (major, largely permanent).** Nothing can make the freeze-before-
   test ordering third-party-verifiable after the fact; the repository now says so
   plainly everywhere. But the only corroboration of the identity-rewrite mapping
   — 31 unreachable pre-rewrite commit objects — exists on one laptop and is one
   `git gc` from disappearing; the live HF Space runs the P1 version of
   `inference.py`/`README.md` (P3 `hf_repo_path` and the P4b licence text were never
   pushed), so RESULTS.md:380 "deploy/ … is the Space repo verbatim" is false; the
   `inference.py` docstring shipped to the Space still says "1879 images"; every
   existing checkpoint still embeds `train_folds=[1,2,3,4]`; `v2_loco.py` never
   adopted the P2 `checkpoint_payload` fix; and the audit-§10 code hygiene list
   (hardcoded paths, `deploy_hf.py` rewriting `src/inference.py`, duplicated
   frozen-path functions, unpinned deps) is unchanged despite "deferred to P2/P3".

**Three strongest verified claims**

1. **The external-v1 validation and everything downstream of it is exactly what
   the documents say, and reproduces in a fresh environment.** All four cohorts'
   AUC/sens/spec/CIs/confusions/medians recompute identically from CSVs with one
   adding commit and zero modifications; `y_pred == (p_cal ≥ 0.2683)` and
   `calibrate(raw, T) == p_cal` on all 12 prediction files; the 10 member
   probabilities reproduce the ensemble bitwise on 2454/2454 images (with the
   frozen reduction order); T, ECE, NLL, threshold, sens/spec and both CIs are
   bitwise equal to the JSONs; the self-check reproduces 0.9234433158791243
   byte-identically; the live Space matches to 1.07e-7; 28/28 sha256 match locally
   and on HF; the Zenodo file is bit-identical to `git archive v1.0-audited`.
2. **Split integrity, no v1 leakage, and the data audit hold under independent
   re-derivation.** 0 of 1064 patients span folds; the committed fold file is the
   official one (sha256 8bed2d2e…, byte-identical to the raw copy) and both OOF
   artifact sets sit on it 1875/1875; 46 wandb run configs prove fold k excluded
   from checkpoint k; no v1 code opens an external directory; 15/15 tests pass.
   Re-hashing all five image sets with the repo's own functions reproduces the
   keep-lists byte-for-byte (846→810, 1559→1013 = 507 + 39, 386→379), the 623-row
   hits file, exactly two d = 8 cross-set candidates, min d ≥ 10 elsewhere and the
   12,056,505 pair count; BUSI_WHU is absent from every registry and result.
3. **The response to the first audit was substantive and every numeric or
   overstatement item it lists is closed.** All 8 §2 discrepancies, all 10 §7 rows
   and all 29 §8 overstatements are gone in their original form; the three
   pre-registered negative results carry their original criteria in all four
   documents; the P2 post-hoc analyses (epoch-selection optimism 0.0111 ± 0.0073,
   nested operating point 0.9003 ± 0.0569 / 0.7802 ± 0.1315 with sens < 0.90 on
   3/5 folds, 32/121, 34/238) reproduce independently — the nested table from
   `oof_vit_preds.csv` alone, the epoch table from the raw wandb datastores.

**Overall verdict.** The repository **supports RESULTS.md and the README "Headline
results" table as written**, and supports docs/report.md's *results* sections
(§3, and the abstract's numbers) as written: every number traces, recomputes and
carries its caveat in the same paragraph. It **does not support docs/report.md's
provenance and reproducibility statements as written** (§2.1, §2.9, §2.10, §4.5,
§4.6 are stale-false in the direction *unfavourable* to the author, which is a
sync failure, not spin), and the Zenodo deposit misdescribes itself. The internal
headline remains optimistically biased and in-sample by construction — now
disclosed and quantified, not removed. Relative to 2026-09-06 the science has not
changed and the documentation has improved markedly; what a hostile reviewer
finds today is drift between five documents that were edited in sequence, a
self-verification layer (README:95, consistency sweep) that promises more than it
checks, and a provenance chain whose weakest links (rewritten private history,
unreachable git objects, an archive that names the wrong commit) are all
disclosed except the last. Required before this is citable as "audited": re-sync
docs/report.md and README:206 to the P3/P4 state; publish a Zenodo v1.0.1 (or
amend the record description) stating that the archive's internal PROVENANCE §7
predates the re-point and giving the correct commit/hash; add an erratum for
RESULTS.md:483; re-push deploy/ to the Space; fix the BiomedCLIP operating-point
JSON or record the one-image discrepancy; and either stop claiming "every number
recomputes" or scope it to the result tables.

*Not done (and why):* no re-training or new external inference (forbidden);
`v2_biomedclip_external.py --self-check` not re-run (adds nothing to v1 parity;
prior value 0.9230647908649297 on record); M2/M3 recalibration draws validated
from the committed per-draw CSVs (M1 re-derived in full); visual adjudication of
the two pHash candidates could not be repeated from the distributed repository
because the figure was untracked in P4c (the table and BUS-BRA-side thumbnail
were inspected); GitHub push history before 2026-09-07 is not exposed by the API,
so "nothing had been pushed" before the rewrite remains unverifiable.
