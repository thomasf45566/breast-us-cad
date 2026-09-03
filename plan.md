# plan.md — BreastUS-CAD task state
Read this at session start. Update the checkboxes when a task completes.

## Current phase: Step 8 (model improvement) → freeze → external validation

## Done
- [x] Env, repo, data (BUS-BRA / BrEaST / BUSI downloaded)
- [x] data.py + patient-level fold checks + EDA notebook
- [x] Baseline effb0 fold-5: AUC 0.8823 (tag: baseline-v1)
- [x] 5-fold CV effb0: AUC 0.891 ± 0.027 (tag: cv-v1)
- [x] Backbone screening fold-5: convnext_small 0.9257, vit_b16 0.9293
- [x] 5-fold CV convnext_small: AUC 0.930 ± 0.017 (summary in reports/cv_convnext_summary.csv)
- [x] 5-fold CV vit_b16: AUC 0.931 ± 0.016 (comparison table in RESULTS.md)
- [x] Backbone comparison table + interpretability audit (RESULTS.md)
- [x] Backbone decision: vit_base_patch16_224 (see standing decisions)
- [x] Pre-registered CoarseDropout ViT CV: cdrop discarded per pre-registered
      rule; plain cv_vit is the frozen backbone artifact set
      (models/cv_vit_fold{1-5}.pt). Details in RESULTS.md.
- [x] TTA (hflip only) on plain cv_vit CV ckpts: pooled OOF AUC 0.9231 →
      0.9254 with TTA (mean fold 0.9307 → 0.9323, 3/5 folds improved).
      src/tta_eval.py, `--tta` on evaluate.py, table in RESULTS.md, OOF
      preds (both settings) in reports/oof_vit_preds.csv.
- [x] calibrate.py: temperature scaling on pooled OOF TTA preds. T = 2.3644,
      ECE 0.0716 → 0.0401 (15 bins), NLL 0.4411 → 0.3237, AUC unchanged
      0.9254. models/calibration.json, reports/reliability_oof_vit_tta.png.
- [x] pick_threshold.py: threshold 0.2683 on calibrated OOF (sens ≥ 0.90,
      max spec). Sens 0.9028 (CI 0.874–0.928), spec 0.7713 (CI 0.745–0.798),
      PPV 0.654, NPV 0.943; patient-level bootstrap ×2000.
      models/operating_point.json, reports/roc_oof_operating_point.png.
- [x] FREEZE: tag frozen-v1 — 5 ckpts + hflip TTA + calibration.json
      (T=2.3644) + operating_point.json. NO model changes after this tag.
      Artifact list in RESULTS.md "FROZEN MODEL" section.
- [x] External validation PRE-FLIGHT (no external inference run):
      data/external_protocol.md (labels, exclusions, preprocessing, metrics
      — written before any scoring); dedup_busi.py → busi_clean.csv
      (386 → 379: 7 near-dups dropped, pHash d ≤ 8, histogram + pair grid
      in reports/, no BUSI↔BrEaST cross-dups, min d = 10); external_val.py
      frozen pipeline self-check PASSED (fold-5 TTA AUC 0.9234 reproduced
      digit-for-digit, reports/external_selfcheck.txt). Run needs --confirm.

- [x] Protocol AMENDMENT 1 (2026-08-29, verified before any external
      inference): GDPH + SYSUCC added as two separate cohorts (keep-lists
      after pHash dedup: 846→810, 1559→1013 — SYSUCC ~35% dups incl. 39
      label-conflicts); BUSI_WHU EXCLUDED (label mapping unverifiable:
      on-disk 756/171 vs published 560/367, no metadata, DSATNet loader is
      segmentation-only, HF re-release unmappable); full cross-set pHash
      sweep clean (2 d=8 candidates both visually refuted — NO BUS-BRA
      contamination); secondary BI-RADS reader comparison pre-registered
      (≥4a positive, run only after primary). Self-check re-passed after
      registry changes. Details: data/external_protocol.md Amendment 1.

- [x] External validation — THE RUN (2026-08-29, single-shot, tag
      external-v1): AUC BrEaST 0.8542 / BUSI 0.9339 / GDPH 0.9154 /
      SYSUCC 0.8380. Sensitivity held ≥ 0.92 on all four at the frozen
      threshold; specificity dropped to 0.41–0.63 (internal 0.77) —
      threshold does not transfer under domain shift. Full table +
      reading in RESULTS.md "External validation (single-shot,
      frozen-v1)"; preds/ROC/CM in reports/external_*. Recorded as-is;
      no re-tuning.

- [x] Pre-registered secondary BI-RADS reader comparison (protocol (h),
      2026-08-29, from saved predictions only): GDPH — model sens 0.9707 /
      spec 0.4529 vs reader1 0.9760/0.8943, reader2 0.9787/0.5126 (κ 0.51);
      SYSUCC — model 0.9309/0.4740 vs reader1 0.9130/0.6505, reader2
      0.9931/0.1315 (κ 0.22). src/birads_comparison.py, figures + table in
      RESULTS.md. Stray-'c' exclusion vacuous (row already dedup-dropped).

- [x] Grad-CAM gallery (2026-08-29, src/explain.py, fold-5 ckpt @
      blocks[-2].norm1, predicted-class target): internal 4×4 TP/TN/FP/FN
      from fold-5 OOF at the frozen threshold (moderate-confidence
      preferred) + external benign-FP study (top-4 SYSUCC/GDPH by
      calibrated prob). Reading: TP heat on lesion+margins; TN/FN diffuse
      and non-lesional; external FPs appearance-driven (heat on the
      lesion body in 7/8), not text/caliper driven. Figures in reports/,
      note in RESULTS.md "Grad-CAM gallery".

- [x] Segmentation demo (2026-08-30, src/train_seg.py + src/eval_seg.py,
      configs/seg.yaml): smp U-Net effb0-encoder, Dice+BCE, 40 epochs,
      BUS-BRA folds 1-4 → fold 5: mean Dice 0.9016 / IoU 0.8326
      (median 0.9325 / 0.8736, n=383). Single run, no CV/ensemble/
      external eval by design. models/seg_unet_effb0.pt, wandb
      seg_unet_effb0, reports/seg_examples.png, section in RESULTS.md.

- [x] Gradio demo app (2026-08-30, app/app.py): frozen-v1 pipeline on CPU
      (5-ckpt hflip-TTA ensemble, T + threshold read from models/*.json),
      U-Net contour + fold-5 Grad-CAM + calibrated-prob result card,
      research-prototype banner, About w/ limitations, 4 bundled BUS-BRA
      fold-5 examples (all classified correctly end-to-end). Warm CPU
      latency 0.53 s (target < 3 s). Section in RESULTS.md.

- [x] HF Spaces deployment (2026-08-30): src/inference.py = frozen
      inference path (no wandb/albumentations), app + external_val import
      it; self-check + four-example probs re-verified bitwise after every
      change. Weights at happytommy/breast-us-cad-weights; public Gradio
      Space happytommy/breast-us-cad (cpu-basic; free tier gone — needed
      HF PRO). cv2.resize INTER_LINEAR proved platform-unstable (Arm
      KleidiCV HAL locally vs x86) → ported to integer numpy
      (inference.resize_bilinear_frozen), bitwise-equal to local cv2 on
      all 1,879 BUS-BRA images. Live Space matches local to max delta
      1.07e-07 (cross-arch BLAS floor; identical at display precision).
      Details in RESULTS.md "Hugging Face Spaces deployment".

## Next (strict order)
1. README / slides / one-pager / rehearsal

## v2 line — site-specific recalibration study (secondary, post-hoc)
Strictly post-hoc to external-v1; must never alter any external-v1 number.
Protocol: data/external_protocol.md Amendment 2 (committed 2026-08-31
BEFORE any v2 computation). Inputs: saved external_*_preds.csv only — no
inference, no training, no model changes. Outputs: reports/v2_* + RESULTS.md
section "v2: Site-specific recalibration".
- [x] Amendment 2 written and committed alone, pre-computation (610e87f)
- [x] M1 (primary, 2026-08-31): src/v2_recalib_curve.py — sanity checks
      (a) k=0 reproduces external-v1 digit-for-digit, (b) oracle sens ≥
      0.90, (c) cal/eval disjoint — all passed; k grid per Amendment 2,
      R=500, seed 42, natural prevalence; held-out sens/spec + recovery,
      degenerate fractions (≤2.4%, k=10 only). k*: GDPH 10, SYSUCC 10,
      BrEaST 20, BUSI 20. All k reported. reports/v2_recalib_M1_{draws,
      summary}.csv + _curves.png; RESULTS.md "v2: Site-specific
      recalibration" (external-v1 sections untouched).
- [x] M2/M3 (2026-08-31, same harness/draws/seed; M1 determinism-checked
      against committed CSV): M2 refit-T (calibrate.py LBFGS; non-positive-T
      draws fall back to frozen, 7–20% at k=10 → 0% at k≥100) with (a)
      frozen thr / (b) local rule; M3 Platt + local rule. Findings: M2b/M3
      ≡ M1 (monotone maps + local rule pick the same rank boundary); M2a
      keeps sens highest but never reaches k* (shift is location, not
      scale); POST-HOC k_reliable (2.5th-pct recovery ≥ 0.5) reached only
      GDPH k=200 — no method is draw-level reliable at small k. Labeled
      POST-HOC in RESULTS.md + figure. reports/v2_recalib_methods_{draws,
      summary}.csv + v2_recalib_methods.png.
- [x] Amendment 3 written and committed alone, pre-computation (78f3991):
      ensemble disagreement as abstention signal. Authorized constrained
      re-run to persist the 10 member probs (must reproduce saved
      y_prob_raw to 1e-9 or abort; no metrics in the re-run script);
      signals U_std/U_range vs margin baseline |p_cal − 0.2683|;
      per-cohort error-AUROC + abstention curves q ∈ {5,10,20,30}%;
      success = error-AUROC(U_std) ≥ 0.65 AND q=10% spec +0.05 with sens
      not below frozen AND U_std > margin baseline. No internal OOF
      reference exists (one held-out member per image) — stated.
- [x] Amendment 3 execution (2026-08-31): src/v2_dump_members.py —
      reproduction check PASSED BITWISE (float32) on all 4 cohorts
      (252/379/810/1013 images), members in reports/v2_members_*.csv;
      src/v2_disagreement.py — error-AUROC U_std 0.77/0.86/0.80 on
      BrEaST/BUSI/GDPH (beats margin baseline) but 0.63 vs margin 0.75
      on SYSUCC; q=10% U_std abstention gains spec +4.4/+7.5/+5.1/+4.1
      pts with sens dipping 0.3–0.9 pts. VERDICT (mechanical, per (q)):
      NOT USEFUL on 0/4 cohorts — c2's sens-held clause fails everywhere
      (SYSUCC also fails c1/c3). reports/v2_abstention_{metrics,curves}
      .csv, v2_disagreement_auroc.png, v2_abstention_curves.png;
      RESULTS.md "v2: Ensemble disagreement as an abstention signal".
- [x] POST-HOC cross-site transfer (2026-08-31, src/v2_cross_site.py, not
      pre-registered — labeled descriptive): 5 sources (internal + 4
      cohorts, full data) × 4 targets, M1 thr transfer + M2 T transfer.
      Internal row reproduces external-v1 (asserted). Every external M1
      threshold beats the internal one on every target; GDPH's thr 0.4484
      drops cross-site sens to 0.80–0.87. reports/v2_cross_site.csv +
      _matrix.png.

## v2 line 4 — multi-source training + domain-pretrained backbone
Protocol: data/external_protocol.md Amendment 4 (committed alone at
3faa70b, BEFORE any v2-line-4 code). external-v1 is final and untouched;
external cohorts may enter TRAINING for v2 models only. Q2 first; Q1's
backbone fixed by rule (v). Artifacts: models/v2_*, reports/v2_*,
RESULTS.md sections "v2: Domain-pretrained backbone (Q2)" and
"v2: Multi-source LOCO training (Q1)".
- [x] Amendment 4 written and committed alone, pre-code (3faa70b)
- [x] Q2a (2026-09-01): USFM loading spike — NOT CLEAN. USFM_latest.pth
      is BEiT-style: no pos_embed (position = shared rel-pos-bias table,
      no slot in vanilla ViT) + LayerScale; 149/150 tensors map but 27
      checkpoint tensors (entire positional mechanism) unused
      (src/v2_load_usfm.py). Pre-registered fallback fired: BiomedCLIP
      ViT-B/16 (open_clip) substitution declared in protocol + RESULTS.md
      BEFORE any training. Q2 artifacts renamed v2_usfm_* →
      v2_biomedclip_*. Follow-through same day: BiomedCLIP export CLEAN
      (150/150 tensors, src/v2_export_biomedclip.py →
      models/pretrained/biomedclip_vitb16_timm.pt), train.py
      model.init_state_dict + configs/v2_biomedclip.yaml, 1-epoch fold-5
      smoke: val AUC 0.8434, 142.7 s/epoch (wandb
      v2_biomedclip_fold5_smoke). Q2b not launched.
- [x] Q2b (2026-09-02): 5-fold CV per v1 protocol (wandb group
      cv_biomedclip, mean best-val AUC 0.9170 ± 0.0220, trails v1 on all
      5 folds); pooled OOF hflip-TTA AUC 0.9109 (v1 0.9254); T = 2.8645;
      thr 0.2007 → sens 0.9012 / spec 0.7177 on calibrated OOF → FREEZE
      models/v2_biomedclip_fold{1-5}.pt +
      v2_biomedclip_{calibration,operating_point}.json
      (src/v2_biomedclip_freeze.py; v1 artifacts asserted untouched)
- [x] Q2c (2026-09-02, src/v2_biomedclip_external.py): self-check passed
      digit-for-digit, then single-shot on 4 cohorts. v2 AUC LOWER on all
      four (BrEaST 0.8441 / BUSI 0.9102 / GDPH 0.8821 / SYSUCC 0.8308 vs
      v1 0.8542/0.9339/0.9154/0.8380); benign-shift ratios 0.892/0.689/
      0.962/0.700163 — branch A 0/4, branch B 1/4 (SYSUCC misses 0.70 by
      0.0002). VERDICT: NOT CLAIMED. Table + preds in reports/v2_biomedclip_
      external_*, RESULTS.md "Q2 verdict". Tag v2-biomedclip.
- [x] Backbone decision for Q1 per rule (v): Q2 criterion NOT met →
      Q1 LOCO backbone = v1 ImageNet vit_base_patch16_224 (recorded with
      the Q2 verdict in RESULTS.md)
- [x] Q1a (2026-08-31, done ahead of Q2 as pure infrastructure — no
      training): src/v2_data.py build_multisource_df(hold_out) — BUS-BRA
      folds 1–4 train / fold 5 val (patient-level) + 85/15 image-level
      split of each external cohort, seed 42, drawn once (asserted
      identical across runs), held-out cohort asserted absent;
      tests/test_v2_splits.py (5 tests PASS). Preprocessing = same
      data.get_transforms path that produced external-v1. FINDING: the
      inference.py resize port matches cv2 only inside KleidiCV's
      size-dispatch envelope — 1421/2454 external images differ by
      ≤ 1 gray level (BrEaST 99/252, GDPH 309/810, SYSUCC 1013/1013,
      BUSI 0/379); parity check asserts load/normalize identical and
      the difference confined to that kernel dispatch
      (`python src/v2_data.py --sweep` reproduces the sweep)
- [ ] Q1b: four LOCO runs (hold out breast/busi/gdph/sysucc); per run:
      early stop + temperature + sens ≥ 0.90 threshold on validation
      only, single model + hflip TTA, then SINGLE-SHOT held-out eval
      (AUC + CI, sens/spec at run threshold, benign median p_cal)
      — breast DONE (2026-09-03, src/v2_loco.py, wandb v2_loco_breast):
      AUC 0.8657 vs v1-single 0.8467 (+0.019), spec 0.7532 vs 0.4740
      (+0.279) at sens 0.8571 — both (t) branches satisfied on this
      cohort. busi/gdph/sysucc pending.
- [ ] Q1c: comparator table — v1 fold-5 single + TTA derived from saved
      v2_members_*.csv (m5 cols, T=2.3644, thr 0.2683; no v1 re-run),
      v1 full ensemble as reference; mechanical verdict per (t) criterion
- [ ] RESULTS.md sections + plan.md checkboxes in the same commits

## Interview-day checklist
- Live demo: https://huggingface.co/spaces/happytommy/breast-us-cad
  (weights: https://huggingface.co/happytommy/breast-us-cad-weights)
- Cold start (sleeping/restart → first prediction): **9.7 s** — open the
  Space BEFORE the interview starts; a first-ever container build also
  downloads 1.7 GB of weights and takes minutes.
- Warm latency on the Space (2 vCPU): ~6.8 s/image. Local fallback:
  `python app/app.py` (0.54 s warm on the M4) — keep it ready in a
  terminal in case conference wifi or HF is down.
- Live-vs-local sanity: `python scripts/verify_space.py` (expects match
  at the 1e-5 platform tolerance; bitwise equality across CPU archs is
  impossible — talking point: KleidiCV resize + BLAS floor, see
  RESULTS.md deployment section).
- Four examples (calibrated p, threshold 0.2683): benign 16.6% / 22.9%,
  malignant 69.1% / 69.9% — all classify correctly on the live Space.

## Standing decisions (do not relitigate)
- TTA ADOPTED (2026-08-29): hflip TTA + 5-model ensemble is the frozen
  inference pipeline — all downstream evaluation, thresholding, external
  validation, and the demo app use it. Numbers in RESULTS.md TTA section.
- Patient-level splits only; BUS-BRA official folds
- BrEaST + BUSI are external-only, single-shot evaluation
- AUC is the primary metric; operating point from pooled OOF, frozen
- Numbers live in RESULTS.md; every result maps to a commit
- Winner backbone: vit_base_patch16_224 (AUC tied with convnext, 9x faster CPU
  inference, saliency usable at blocks[-2].norm1)
- Saliency method for demo/slides: Grad-CAM at blocks[-2].norm1. Attention
  rollout is analysis-only, never in the demo.
- Caliper/text inpainting: rejected for this project scope → future work
  (NTUH data collection will export annotation-free images)
- ONE pre-registered robustness experiment before TTA: CoarseDropout ViT CV.
  Adoption rule, decided BEFORE seeing results: adopt only if pooled OOF AUC
  >= (current ViT OOF AUC - 0.01) AND the saliency check shows visibly reduced
  caliper-adjacent heat on malignant TPs. Otherwise discard, no iteration,
  no second variant. Either way, next step is TTA on whichever ViT wins.