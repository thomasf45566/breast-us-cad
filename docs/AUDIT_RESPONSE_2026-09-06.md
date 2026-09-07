# Response to the independent audit of 2026-09-06 — phase P1 (documentation corrections)

Date: 2026-09-07. Scope: documentation only. Files changed in this phase:
`docs/report.md`, `README.md`, `CLAUDE.md`, `app/app.py` (About text),
`deploy/app.py` (byte-identical copy of `app/app.py`), `deploy/README.md`,
`plan.md`, and this file. **Not changed:** `RESULTS.md`,
`data/external_protocol.md`, anything under `reports/` or `models/`, any
code path (`src/`, `scripts/`, `tests/`, `configs/`). The live HF Space was
not re-pushed; `deploy/` is corrected locally only.

Rule applied to every factual sentence in the corrected documents: it must
trace to `RESULTS.md`, `data/external_protocol.md`, or a committed CSV/JSON
under `reports/` or `models/`. Sentences with no such source were deleted or
rewritten as an explicitly labeled "unrecorded in-session observation". The
one exception, required by the task itself, is the new "Provenance and
limitations of pre-registration" section, whose statements about the
history rewrite trace to git object metadata (commit `2c24500` message,
tag object `frozen-v1`), cited as such.

Line numbers in "original text" refer to the files at HEAD `fedb2f2`
(before this commit), as quoted by the audit. Source line numbers refer to
`RESULTS.md` / `data/external_protocol.md` at the same HEAD (unchanged by
this commit) and to CSV rows (1 = header).

Status legend: **corrected** (text rewritten; "no change needed" is noted
where the audit itself marked the sentence supported and it was retained
verbatim) · **deleted** · **disputed-with-reason** · **deferred-to-P2/P3**
(requires a change outside P1's file list or new computation) ·
**resolved-by-P2** (added 2026-09-07: closed by the P2 commit "P2: post-hoc
analyses for audit 2026-09-06 + train.py stores resolved folds" — RESULTS.md
section "Post-hoc analyses responding to the 2026-09-06 audit" and the
appended "Errata (2026-09-07)").

---

## A. Audit §2 — numbers that do not reproduce or differ between documents (8 items)

| # | Audit item | Original text | Corrected text | Source file:line | Status |
|---|---|---|---|---|---|
| 2.1 | "~9.7M-pair" pHash sweep (README:49; report abstract:14, §2.4:58) | "~9.7M-pair perceptual-hash sweep" / "970 萬對感知雜湊稽核確認與訓練集零重疊" | "12.06M-pair … all within- and cross-set unordered pairs among 1875/252/379/846/1559 images" (README); report §2.4 states the definition — Σ C(n,2) over the five sets + Σ n_a·n_b across sets = 12,056,505 — and notes the old figure had no source. "Zero overlap" → "no near-duplicate at pHash d ≤ 8" everywhere. | external_protocol.md:143–152 (set sizes, all-pairs definition, 2 candidates, min d ≥ 10); reports/phash_sweep_hits.csv (623 hit rows; 2 cross-set) | corrected |
| 2.2 | "sensitivity ≥ 0.92 on all four" (RESULTS.md:485, :1058; plan.md:54; app/app.py:67; deploy/README.md:31) | "sensitivity held ≥ 0.92 at the frozen threshold" | "sensitivity held ≥ 0.918 at the frozen threshold" in plan.md, app/app.py, deploy/app.py, deploy/README.md; report §3.11/§4.1/§4.2 "全域 ≥ 0.92" → "≥ 0.918". BrEaST sens = 90/98 = 0.9184. | RESULTS.md:185 (0.9184), :201 (90/91/8/63) | corrected (4 doc places); **resolved-by-P2** for RESULTS.md:485 and :1058 via the appended Errata section (existing lines untouched) |
| 2.3 | Report §3.9:117 "前三世代 CI 不重疊" | "(前三世代 CI 不重疊)" | "U_std 與 margin 之 CI 於 BUSI、GDPH 不重疊,於 BrEaST 重疊(0.710–0.831 vs 0.595–0.729)" | reports/v2_abstention_metrics.csv rows 2 & 4 (BrEaST U_std 0.7100–0.8314; margin 0.5945–0.7286); RESULTS.md:628–635 | corrected |
| 2.4 | Report §4.5:157 "≤ 10⁻³ 之 resize 分派差異" vs §2.8/RESULTS "max 7×10⁻³" | "≤ 10⁻³ 之 resize 分派差異(2/1,421 邊界翻轉)" | "校準機率最大 7×10⁻³、中位 8×10⁻⁴,2/1,421 邊界翻轉" (§4.5); §2.8 restated with per-cohort counts 99/0/309/1013 and flips case038/case200 | RESULTS.md:1083–1089; reports/v2_resize_dispatch_impact.csv (1421 rows; max abs_delta 0.00723, median 0.00077, decision_flip=1 on case038, case200) | corrected |
| 2.5 | RESULTS.md:403 "1,879 BUS-BRA images" (also plan.md:99, report §2.8) | "bitwise-equal to local cv2 on all 1,879 BUS-BRA images" | plan.md: "all 1,875 BUS-BRA images (713 sizes; RESULTS.md's '1,879' double-counts the 4 bundled examples)"; report §2.8: "BUS-BRA 尺寸範圔(1,875 張,713 種尺寸)" | models/operating_point.json `n_images: 1875`; models/calibration.json `n: 1875`; RESULTS.md:1084 (713 sizes); RESULTS.md:403 (the erroneous 1,879, which explicitly adds "+ the 4 examples") | corrected (plan.md, report); **resolved-by-P2** for RESULTS.md:403 via the appended Errata section |
| 2.6a | Report §3.1:84 "121 張惡性僅 3 張機率落 0.6–0.95" | "模型高度 overconfident(121 張惡性僅 3 張機率落 0.6–0.95)" | "校準前 ensemble 明顯 overconfident(T = 2.36);校準後中段機率仍殘餘輕度 overconfidence(reliability diagram)" | RESULTS.md:126–129; models/calibration.json (`temperature`, ECE before/after); **P2:** reports/posthoc_overconfidence.csv (fold-5 malignant raw TTA prob in [0.6, 0.95] = 32/121, not 3/121) | deleted (figure) / corrected (sentence) / **resolved-by-P2** (sourced count now in report §3.1) |
| 2.6b | Report §2.5:61 "六項唯讀 audit … 書面 GO" | "推論前六項唯讀 audit(git 狀態、protocol、keep-lists、registry、凍結產物、無外部預測檔)全數通過並留書面 GO" | Kept only the sourced parts (Amendment 1's pre-check that reports/ held no external predictions; self-check; single `--dataset all --confirm` run; one adding commit per preds CSV) and added, in italics: "未記錄之 session 內觀察:推論前另行執行之多項唯讀檢查…與其「GO」決定未留存於任何已提交檔案,故不作為事實陳述" | external_protocol.md:81–83; RESULTS.md:175–178, :387–389; reports/external_selfcheck.txt | corrected (rewritten as labeled unrecorded in-session observation) |
| 2.6c | Report §2.3:46 and §4.4:154 "BUS-CoT … 不適用 / 聚合污染" | "BUS-CoT 因聚合公開來源與訓練集重疊而不適用" | removed from §2.3 and §4.4 | (no source in RESULTS.md, protocol, plan.md or any committed artifact) | deleted |
| 2.6d | Report §2.2:41 "fold-5 篩選(門檻 +0.01)" | "fold-5 篩選(門檻 +0.01)後對 ConvNeXt-Small 與 ViT-B/16 行完整五折 CV" | "於 fold 5 篩選 ConvNeXt-Small(0.9257)與 ViT-B/16(0.9293)後對兩者行完整五折 CV" — the +0.01 gate is removed | RESULTS.md:7–8 (screening rows) | deleted (gate) / corrected (sentence) |
| 2.7 | k_reliable "≈100–200" (README:35; report abstract:16, §3.7:111, §4.1:145; app/app.py:77) | "draw-level reliability needs ≈100–200" / "可靠性需約 100–200 例" / "收斂需 k ≈ 100–200" / "variance stays wide until roughly 100–200 labels" | "the POST-HOC draw-level reliability bar (k_reliable) was reached only on GDPH at k=200 and not reached on the other three cohorts within the pre-registered grid" (README, app About, report abstract/§3.7/§4.1); report §3.7 adds the grid limits (BrEaST/BUSI to k=100, SYSUCC to k=200) and the k=10 median sens 0.830/0.846/0.866/0.909 | RESULTS.md:509–512 (POST-HOC definition), :516–531 (table: 200 for GDPH, "not reached" elsewhere), :556–558; reports/v2_recalib_methods_summary.csv `recovery_p2.5` (M1: BrEaST −0.62 at k=100, BUSI +0.03 at k=100, GDPH +0.60 at k=200, SYSUCC −0.27 at k=200); reports/v2_recalib_M1_summary.csv rows 2/7/12/18 (k=10 sens medians) | corrected |
| 2.8 | GDPH "model within the readers' distribution" (report abstract:16, §4.2:148) | "模型表現落於兩位放射科醫師 BI-RADS 判讀之分佈內(κ 0.22–0.52)" / "模型落於 inter-reader 分佈內而未達最佳專家" | "GDPH 模型於 sensitivity 與 specificity 兩軸皆低於兩位判讀者;SYSUCC 模型居兩位判讀者之間;κ(0.22 / 0.51)為判讀者間一致性,不涉及模型" | RESULTS.md:249–258 (GDPH: model 0.9707/0.4529 vs r1 0.9760/0.8943, r2 0.9787/0.5126; κ 0.5149), :262–270 (SYSUCC between readers; κ 0.2152) | corrected |

## B. Audit §7 — statistical claims (every row)

| # | Audit row | Original text | Corrected text | Source file:line | Status |
|---|---|---|---|---|---|
| 7.1 | RESULTS.md:634 "non-overlapping CIs on BUSI/GDPH" (U_std vs margin) — *supported* | (RESULTS.md, unchanged) | retained; report §3.9 now matches it exactly | reports/v2_abstention_metrics.csv rows 5,7,8,10 | corrected — no change needed (audit: supported) |
| 7.2 | Report §3.9:117 "前三世代 CI 不重疊" — *unsupported* | see A-2.3 | see A-2.3 | see A-2.3 | corrected |
| 7.3 | Report §3.5:105 "GDPH 之 Reader1 顯著高於模型 ROC" — *unsupported* (no test) | "Reader1 顯著高於模型 ROC;Reader2 落於曲線上" | "Reader1 之 specificity 遠高於模型(0.89 vs 0.45,點估計;未做任何檢定),Reader2 接近模型之 ROC 曲線" + "模型於兩軸皆低於兩位判讀者" | RESULTS.md:255–258 | corrected |
| 7.4 | RESULTS.md GDPH LOCO "CIs do not overlap" — *supported* | (RESULTS.md, unchanged) | retained in report §3.11 ("AUC CI 不重疊") | reports/v2_loco_q1c_table.csv rows 8–9 (0.9302–0.9601 vs 0.8668–0.9128) | corrected — no change needed (audit: supported) |
| 7.5 | LOCO "criterion MET" (README:36; report abstract:16 "判準成立") — met as registered; omissions | "multi-source LOCO training met its pre-registered criterion (ΔAUC +0.013 to +0.056 on 4/4 …)" / "ΔAUC +0.013 至 +0.056(判準成立)" | Same paragraph now states: criterion MET via branch A vs the v1 fold-5 **single** model; 2/4 paired ΔAUC CIs include zero (BrEaST −0.0155…+0.0550, SYSUCC −0.0099…+0.0346); branch B failed 2/4 (sens 0.8466, 0.7997 < 0.85); vs the deployed v1 ensemble ΔAUC = +0.0115/+0.0033/+0.0301/+0.0064, only 2/4 ≥ +0.01; the registered question concerned the specificity collapse (README, report abstract, §3.11, §4.1, §5) | RESULTS.md:1019–1033 (verdict), :1011–1013 (CI reading); external_protocol.md:390–392 (Q1 question), :459–463 (criterion), :451–458 (comparator); reports/v2_loco_q1c_table.csv (loco vs v1_ensemble `auc` per cohort → the four ensemble deltas) | corrected |
| 7.6 | Δspec "+0.27 to +0.30, paired CI 皆排除零" — *supported*, thresholds differ per model | "Δspec +0.27 至 +0.30(paired CI 皆排除零)" | retained, with "係於各模型不同閾值下比較(閾值定位混淆)" / "at different thresholds per model" added | reports/v2_loco_q1c_table.csv `delta_spec_lo/hi`; RESULTS.md:1008–1013 | corrected |
| 7.7 | Report abstract "BiomedCLIP … 於四世代 AUC 皆較低" — point estimates only | "但於四世代 AUC 皆較低" | "外部 AUC 點估計於四世代皆較低" (abstract); §3.10: "點估計皆較低,但各世代 v1/v2 之未配對 CI 皆重疊,未做配對檢定" | reports/v2_biomedclip_external_comparison.csv (`v1_auc_lo/hi`, `v2_auc_lo/hi` overlap on all four rows); RESULTS.md:834–839 | corrected |
| 7.8 | Internal sens 0.903 / spec 0.771 presented as "results" — in-sample, unlabeled | README:20 row; report abstract/§3.2/§3.3 | Labeled **in-sample** everywhere (README † footnote; report abstract, §3.2, §3.3 table row, §4.5; plan.md pick_threshold entry; app/deploy About "0.77 (internal, in-sample)") | models/operating_point.json (`fit_on: pooled OOF … y_prob_tta`, threshold and sens/spec on the same n=1875); external_protocol.md:250–252 (Amendment 2 (l) calls the identical operation an "in-sample oracle"); **P2:** reports/posthoc_nested_threshold.csv (out-of-sample 0.9003 ± 0.0569 / 0.7802 ± 0.1315) | corrected; **resolved-by-P2** (quantified) |
| 7.9 | Cross-site "皆優於" (abstract) — descriptive, in-sample thresholds, sens cost omitted | "外部世代彼此借用閾值皆優於訓練集閾值" | "外部世代彼此借用閾值之 specificity 皆高於訓練集閾值(post-hoc、in-sample 閾值、描述性;GDPH 閾值使跨場域 sens 降至 0.80–0.87)"; §3.8 lists all four thresholds and "無 CI" | RESULTS.md:565–599 (POST-HOC, in-sample diagonals, :589–592 GDPH sens 0.80–0.87, :597–598); reports/v2_cross_site.csv | corrected |
| 7.10 | "10–20 labels recover most specificity (median)" — caveats dropped in abstract | "10–20 例本地標註即可於中位數上恢復幾乎全部 specificity(可靠性需約 100–200 例)" | "…恢復接近 oracle 之 specificity(pre-registered k*),但 k=10 之中位 sensitivity 為 0.83–0.91;post-hoc … k_reliable 僅 GDPH 於 k=200 達標" (abstract, README, app About) | RESULTS.md:449–475, :481–486; reports/v2_recalib_M1_summary.csv rows 2/7/12/18 | corrected |

## C. Audit §8 — overstatement scan (29 items)

| # | Location | Original text | Corrected text | Source file:line | Status |
|---|---|---|---|---|---|
| 8.1 | README:20 headline row | "Internal (BUS-BRA, 5-fold patient-level CV) \| 0.931 ± 0.016 \| 0.903 \| 0.771" | Split into row (a) "single model, no TTA, 5-fold CV, best epoch on the reported fold \| 0.9307 ± 0.0161 \| — \| —" and row (b) "5-ckpt ensemble + hflip TTA + T, pooled OOF (n=1875) \| 0.9254 \| 0.903 † \| 0.771 †", with a note that (a) is optimistically biased (checkpoint at best val-AUC epoch on the same fold) and † marks in-sample sens/spec. Same split in report abstract and §3.2. | reports/cv_vit_summary.csv (`auc`, `best_epoch` 29/10/18/16/19; mean/sd rows 7–8); reports/tta_vit_summary.csv; RESULTS.md:100–101 (pooled OOF 0.9254), :165–167; models/operating_point.json; **P2:** reports/posthoc_epoch_selection.csv (fixed-epoch 0.9195 ± 0.0164), reports/posthoc_nested_threshold.csv | corrected; **resolved-by-P2** (README rows a′/b′ added) |
| 8.2 | README:30 "discrimination transfers" | "discrimination transfers; calibration does not" | "discrimination transfers reasonably (BUSI and GDPH within or near the internal range; BrEaST and SYSUCC lower, both CIs below the internal 0.9254); the operating point does not" | RESULTS.md:185–194; models/calibration.json `auc` 0.9254 | corrected |
| 8.3 | README:33 "the model fails in the safe direction" | "the model fails in the safe direction" | "the errors are false positives, not missed cancers — whether that direction is 'safe' was not assessed (no harm analysis)"; report §4.2 and §5 likewise ("此方向是否『可接受』未經任何危害分析") and note LOCO sens falls to 0.80 | RESULTS.md:195–198, :227–230 (37–59% of benigns over threshold); :967 (LOCO SYSUCC sens 0.7997) | corrected |
| 8.4 | README:35 "(draw-level reliability needs ≈100–200)" | as quoted | see A-2.7; POST-HOC label added | see A-2.7 | corrected |
| 8.5 | README:36–39 LOCO "met its pre-registered criterion" | as quoted | see B-7.5 | see B-7.5 | corrected |
| 8.6 | README:45 "Pre-registered protocols … committed before results" | "Pre-registered protocols (external protocol + 4 amendments) committed before results" | "Internally pre-registered protocols … committed before the computations they govern — in a private, version-controlled history with no external timestamp; see 'Provenance' below" + new README section "Provenance and limitations of pre-registration" (also report §2.10) noting Amendment 1 already contained the audit counts it registers | external_protocol.md:79–83, :174–179, :269–275, :376–383 (amendment preambles); git commit 2c24500 message (identity rewrite, dates preserved); RESULTS.md:372–375 (HF repo as only public artifact) | corrected |
| 8.7 | README:49 "~9.7M-pair" | as quoted | see A-2.1 | see A-2.1 | corrected |
| 8.8 | README:52 "Three pre-registered negative results" vs report §4.3:150 "兩個" | README "Three …"; report "### 4.3 兩個 pre-registered 失敗的方法學價值" | Both documents now say three (CoarseDropout, abstention 0/4, BiomedCLIP NOT CLAIMED); README adds that the CoarseDropout gate was a visual call on three images and no robustness metric was measured | RESULTS.md:61–77 (cdrop rule + gate 2 "1/3 improved…"), :663–680 (abstention 0/4), :846–859 (Q2 NOT CLAIMED) | corrected |
| 8.9 | README:55–56 "Every number traces to a commit; the deployed demo shares the exact inference module" | as quoted | "Every number in RESULTS.md recomputes from committed CSV/JSON artifacts; the deployed demo shares the same model, calibration and inference module as the validation pipeline, while its preprocessing differs from the validation path on non-BUS-BRA image sizes (documented; impact on calibrated probability ≤ 7×10⁻³)"; README "Not in the repository" paragraph states v2 checkpoints are published nowhere; report §2.8 adds that external-v1 (2cbe1da) predates inference.py (2859043) | RESULTS.md:377–385, :1079–1091; RESULTS.md:372–375 (HF repo contents = v1 only); git log (commit dates) | corrected |
| 8.10 | README:63–64 repository map "data/splits/ official BUS-BRA folds", "models/ checkpoints" | as quoted | "data/splits/ frozen external keep-lists (busi/gdph/sysucc_clean.csv) — the BUS-BRA fold file is NOT here"; "models/ calibration + operating-point JSONs only; the .pt checkpoints are git-ignored"; new paragraph says the folds are the dataset's own `5-fold-cv.csv` in git-ignored `data/raw/busbra/`, v1 weights on HF, v2 weights unpublished. Same fix in CLAUDE.md rule 1. | .gitignore (`data/*`, `models/*.pt`); `git ls-files data/splits models` (3 CSVs; JSONs only); external_protocol.md:39, :102–103 (keep-list files); RESULTS.md:373–375 | corrected (README, CLAUDE.md); **deferred-to-P3** for data/README.md:17 (same misstatement; file outside P1's list) and for committing/fetching the fold file |
| 8.11 | README:76–80 reproduce block | "python src/train.py --config configs/vit.yaml / python src/external_val.py --self-check … must reproduce AUC 0.9234 digit-for-digit" | New "Reproduce" section: exact dataset layout and filenames; self-check reproduces 0.9234433158791243 byte-identically; `inference.resolve_weight` auto-downloads missing weights from HF (network, no offline switch); pytest needs all five datasets; full regeneration chain `cross_validate.py --config configs/vit.yaml --prefix cv_vit` → `tta_eval.py` → `calibrate.py` → `pick_threshold.py`; `train.py` trains one fold only; `WANDB_MODE=offline`; MPS training not bitwise reproducible | reports/external_selfcheck.txt; RESULTS.md:387–391; external_protocol.md:15–19, :31–34, :88–94 (dataset dirs/filenames); RESULTS.md:84–90 (tta_eval), :113–118 (calibrate), :134–136 (pick_threshold), :157–158 (5 ckpts one per fold) | corrected |
| 8.12 | Report abstract:14 "970 萬對 … 零重疊" | as quoted | see A-2.1 | see A-2.1 | corrected |
| 8.13 | Report abstract:16 / §4.2 "落於兩位放射科醫師 … 分佈內(κ 0.22–0.52)" | as quoted | see A-2.8; §4.2 adds "κ … 為判讀者間一致性,說明一致性工具之需求,不涉及模型表現" | see A-2.8 | corrected |
| 8.14 | Report abstract:16 "10–20 例 … (可靠性需約 100–200 例)" | as quoted | see B-7.10 / A-2.7 (median claim kept; reliability range replaced; k=10 sens cost added) | see B-7.10 | corrected |
| 8.15 | Report abstract:16 "任何單調機率校準接本地閾值重選,其決策與直接重定閾值完全相同" | as quoted | "…為結構性等價(實跑曲線於 k ≤ 30 因擬合失敗回退與同分而有小差異)"; §3.7 lists the k* differences (BrEaST M1 20 / M2b 10; GDPH M1 10 / M2b 20; SYSUCC M1 10 / M2b 20) | RESULTS.md:535–542, :516–531 (k* column) | corrected |
| 8.16 | Report abstract:16 "外部世代彼此借用閾值皆優於訓練集閾值" | as quoted | see B-7.9 | see B-7.9 | corrected |
| 8.17 | Report abstract:16 "BiomedCLIP 預訓練縮小良性漂移(比值 0.69–0.96)…"; §4.1 "縮小漂移但犧牲判別力"; §5:165 "訓練端強化有效" | as quoted | Abstract: "BiomedCLIP 替代骨幹依 pre-registered 判準 **NOT CLAIMED**(分支 A 0/4、分支 B 1/4;外部 AUC 點估計於四世代皆較低)"; §3.10 keeps the ratios only as the criterion's own metric with "依規定不作為發現陳述" and drops "暗示穩定性—判別力之權衡"; §4.1 "BiomedCLIP 替代骨幹依判準 NOT CLAIMED"; §5 "訓練端修正中,多來源訓練達成其 pre-registered 判準(僅 AUC 分支,2/4 CI 含零),領域預訓練替代品未達" | RESULTS.md:846–859 (verdict NOT CLAIMED, 0/4, 1/4, SYSUCC 0.700163); external_protocol.md:486–493 (criterion); reports/v2_biomedclip_external_comparison.csv `shift_ratio` | corrected |
| 8.18 | Report abstract:16 / §3.11 "ΔAUC +0.013 至 +0.056(判準成立)" | as quoted | see B-7.5 | see B-7.5 | corrected |
| 8.19 | Report §2.5:61 "六項唯讀 audit … 書面 GO" | as quoted | see A-2.6b | see A-2.6b | corrected (labeled unrecorded in-session observation) |
| 8.20 | Report §2.3:46 "BUS-CoT … 不適用" | as quoted | see A-2.6c | — | deleted |
| 8.21 | Report §3.1:84 "121 張惡性僅 3 張 …" | as quoted | see A-2.6a; report §3.1 now states the sourced count 32/121 (26.4%) | reports/posthoc_overconfidence.csv | deleted; **resolved-by-P2** |
| 8.22 | Report §3.4:102 "Specificity 崩落完全歸因於良性分佈右移;漂移幅度與資料集風格距離同向" | as quoted | "Specificity 崩落與良性分佈右移同時出現(描述性觀察;無因果檢定)。*未檢定之詮釋:漂移幅度可能與資料集風格差異有關,但本研究未測量任何風格距離。*" | RESULTS.md:212–233 (descriptive, post-hoc) | corrected (restated as untested interpretation) |
| 8.23 | Report §3.5:105 "Reader1 顯著高於模型 ROC" | as quoted | see B-7.3 | see B-7.3 | corrected |
| 8.24 | Report §3.9:117 "前三世代 CI 不重疊" | as quoted | see A-2.3 | see A-2.3 | corrected |
| 8.25 | Report §3.9 / §4.3 "判準之概念缺陷(將轉介計為漏診)" | as quoted | §3.9: "*作者事後詮釋(非判準之一部分):失敗源於 sens 保留條款而非信號品質(SYSUCC 除外);任何放寬皆須新的 pre-registration。*"; §4.3 likewise labeled "事後詮釋"; the 0/4 outcome stated first, unchanged | RESULTS.md:676–687 | corrected (labeled post-hoc interpretation) |
| 8.26 | Report §4.2:148 "模型與判讀者被同一批非典型良性以同一方式誤導"; "SYSUCC 之 disagreement 反轉為第三方印證" | as quoted | "*未檢定之詮釋:外部 FP 之 Grad-CAM 熱區落於病灶本體,與「非典型良性外觀」相符;本研究未做逐影像之模型—判讀者一致性分析,故「模型與判讀者被同一批影像誤導」為假說而非結果。*" The "第三方印證" sentence is removed. | RESULTS.md:300–310 (external FP CAMs, "do not quantify"); **P2:** reports/posthoc_reader_concordance.csv (GDPH reader1 ≥ 4a on 34/238 model FPs; reader2 152/238; SYSUCC 81/152, 143/152) | corrected / deleted; **resolved-by-P2** (per-image cross-tab now in report §3.5, §4.2) |
| 8.27 | Report §4.5:157 "≤ 10⁻³ 之 resize 分派差異" | as quoted | see A-2.4 | see A-2.4 | corrected |
| 8.28 | Report §4.6:160 "外部共識保守閾值 0.33–0.38" | as quoted | "起始候選為四個外部 in-sample 閾值 0.331 / 0.376 / 0.385 / 0.448 之任一,須於本地資料上重選" | RESULTS.md:575–577; reports/v2_cross_site.csv `src_threshold_M1` (0.3314 / 0.3755 / 0.3851 / 0.4484) | corrected |
| 8.29 | Report §2.9:77 "每一數值對應一 commit" | as quoted | "RESULTS.md 每一節列出產生腳本與 … CSV/JSON 產物,所有表列數值可由該等已提交檔案重算。不在版本庫內者:所有模型檢查點 … 與 BUS-BRA fold 檔 … v2 之九個檢查點 … 尚未公開 … MPS 訓練無 bitwise 決定性" | RESULTS.md section artifact lists (e.g. :104–106, :135–138, :204–206, :446–447, :841–842, :1072–1075); RESULTS.md:373–375 (HF contents); .gitignore | corrected; publication of v2 weights **deferred-to-P3** |

Three pre-registered negative results, per audit §8's closing check: CoarseDropout — faithful, now with the "visual gate on three images, no robustness metric" caveat (README, report §3.1/§4.3); abstention — outcome faithful, reframing labeled post-hoc (8.25); BiomedCLIP — report abstract/§4.1/§5 brought to RESULTS.md's NOT CLAIMED wording (8.17); report §4.3 counts three (8.8).

## D. Audit §9 — reproducibility from scratch (every deviation)

| # | README step | Audit deviation | Corrected text | Source file:line | Status |
|---|---|---|---|---|---|
| 9.1 | `uv venv --python 3.12 && source .venv/bin/activate` | none (auditor relocated the venv for a read-only audit) | retained | — | corrected — no change needed |
| 9.2 | `uv pip install -r requirements.txt` | none | retained | — | corrected — no change needed |
| 9.3 | "datasets: download from original sources" | paths hardcoded; layout and xlsx filename nowhere stated | README "Dataset layout (hardcoded paths)" block listing `data/raw/{busbra,breast_poland,busi,gdph_sysucc}` with the required files (`bus_data.csv`, `5-fold-cv.csv`, `BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx`, `BIRADS&FOLD.xlsx`) | external_protocol.md:15–19, :31–34, :88–94; data/README.md:69 (local copy layout) | corrected |
| 9.4 | `python src/train.py --config configs/vit.yaml` | trains one fold only; does not regenerate frozen-v1; wandb login required; MPS not bitwise deterministic | README states `train.py` trains one model (folds 1–4 → fold 5, `models/vit_b16_fold5.pt`); gives the full chain `cross_validate.py --config configs/vit.yaml --prefix cv_vit` → `tta_eval.py` → `calibrate.py` → `pick_threshold.py`; `WANDB_MODE=offline` / `wandb offline`; "MPS training is not bitwise reproducible … only inference on the downloaded frozen weights is exact" | RESULTS.md:8 (vit_b16_fold5 screening run), :157–158, :84–91, :113–118, :134–138 | corrected |
| 9.5 | `python src/external_val.py --self-check` | needs BUS-BRA + fold-5 ckpt; `resolve_weight` silently downloads from HF | README: "If `models/cv_vit_fold5.pt` is absent, `inference.resolve_weight` downloads it from the HF weights repo automatically (network access; default local HF cache) — there is no offline switch" | RESULTS.md:373–375 (weights repo), :387–389 | corrected |
| 9.6 | tests (`pytest`) | need all five datasets | README: "`pytest` runs the 8 split tests (all five raw datasets required)"; "Why this repo" bullet amended | (tests/ unchanged; statement of requirement only) | corrected |
| 9.7 | v2 results | nine v2 checkpoints + BiomedCLIP export exist only on the author's disk | README and report §2.9 state this explicitly and that Q1/Q2 are reproducible only from committed prediction CSVs | RESULTS.md:373–375 (HF holds v1 only); reports/v2_biomedclip_external_*_preds.csv, v2_loco_*_preds.csv | corrected (disclosure); **deferred-to-P3** (publication under HF v2/ with CHECKSUMS.txt) |
| 9.8 | CLAUDE.md:29 `python src/evaluate.py --ckpt models/best.pt --split test` | `--split` accepts only `val` | `python src/evaluate.py --ckpt models/cv_vit_fold5.pt --split val [--tta]` with a note that external cohorts go through `src/external_val.py` | (command fix; evaluate.py unchanged) | corrected |

## E. Additional audit findings touched by P1 (for completeness)

| # | Audit section / item | Handling in P1 | Status |
|---|---|---|---|
| E.1 | §1 epoch-selection bias (best-epoch-on-the-same-fold; pooled OOF inherits it) | Disclosed in README, report, plan.md (P1). **P2:** quantified from the five cv_vit wandb datastores — best 0.9307 ± 0.0161 vs fixed epoch 18 0.9195 ± 0.0164, optimism 0.0111 ± 0.0073 (src/posthoc_epoch_selection.py, asserted to reproduce cv_vit_summary.csv); nested operating point 0.9003 ± 0.0569 / 0.7802 ± 0.1315 (src/posthoc_nested_threshold.py) | corrected (disclosure); **resolved-by-P2** (quantification) |
| E.2 | §1 misleading checkpoint metadata (`train_folds=[1,2,3,4]` stored in every CV ckpt) | **P2:** `train.checkpoint_payload` stores the resolved train/val folds; docstring records that existing checkpoints carry the raw YAML; tests/test_checkpoint_config.py (3 tests) | **resolved-by-P2** (existing checkpoints not rewritten; folds actually used remain documented in the wandb configs and RESULTS.md) |
| E.3 | §1 fold file location (CLAUDE.md:15, data/README.md:17, README:63) | CLAUDE.md and README corrected; data/README.md is outside P1's file list | corrected (2 of 3); **deferred-to-P3** (data/README.md) |
| E.4 | §3 self-certified pre-registration timeline; history rewrite; private repo; HF postdates external-v1 | New section "Provenance and limitations of pre-registration" in README and report §2.10 (internal, version-controlled; not externally time-stamped; 2026-08-31 identity rewrite with dates preserved; private at audit time; HF 2026-08-30 after external-v1 2026-08-29). External timestamping | corrected (disclosure); **deferred-to-P4** (Zenodo/OSF, docs/PROVENANCE.md) |
| E.5 | §4 TTA adoption not pre-registered | Stated in report §2.2/§3.2/§4.5, README Provenance section, plan.md standing decision | corrected |
| E.6 | §4 M2 fit-failure fallback and early-stop patience 7 not pre-specified | Stated in report §2.7 (Amendments 2 and 4 paragraphs). Source: RESULTS.md:499–504, :875–876; protocol (t) text lacks a patience value | corrected |
| E.7 | §4 Q1 wording exceeds the fired branch | See B-7.5; RESULTS.md:1028–1033 left untouched and corrected in the appended Errata section | corrected (docs); **resolved-by-P2** (Errata) |
| E.8 | §5 within-BUS-BRA d=8 pair bus_0012-l/-r (same patient, same fold) | Not added to the corrected documents: the pair is recorded in reports/phash_sweep_hits.csv row 2 and is a within-training-set same-patient pair with no leakage implication; no document made a claim about within-BUS-BRA duplicates | disputed-with-reason (no correction needed) |
| E.9 | §6 README "exact inference module"; app/deploy example PNG encodings differ | README wording corrected (8.9); app/deploy About and deploy/README now state the preprocessing difference. The example-file encoding difference produces identical probabilities (audit §6) and is not a documentation claim | corrected / disputed-with-reason (encoding difference: no doc claim to correct) |
| E.10 | §6 Space warm latency 6.15 s measured vs "6.8 s" recorded | RESULTS.md:416 records 6.8 s; report §3.12 and plan.md retain the recorded value with source | disputed-with-reason (recorded measurement stands; the audit's single re-measurement is not a committed artifact) |
| E.11 | §10 code-quality items (hardcoded paths, duplicated functions, deploy script rewriting src/inference.py, Agg backend, etc.) | Code — out of P1 scope | **deferred-to-P2/P3** |

## E2. Demo presentation (added to P1 at the author's request, 2026-09-07)

| # | Item | Original | Corrected | Source file:line | Status |
|---|---|---|---|---|---|
| E2.1 | README demo screenshot `docs/assets/demo.png` | Single row: benign example bus_0186-r (16.6%, LIKELY BENIGN); its benign-class Grad-CAM is diffuse and partly covers the burned-in annotation text | Two-row composite: row 1 = the existing benign capture, pixel-for-pixel unchanged (1500×655); row 2 = malignant example bus_0663-l (69.9%, SUSPICIOUS) captured from the local app (`python app/app.py`, headless Chromium via Playwright, 1500-px viewport, same region: page title through result card, all three panels + card). Thin left-hand label per row ("benign example" / "malignant example"). Nothing cropped out of either capture; CAM target unchanged (predicted class); no different benign example chosen. README caption (verbatim, author-supplied) states that benign-class CAMs are diffuse and can include the annotation text, that this is a documented training-set limitation whose effect on predicted probabilities the border-occlusion probe shows to be small, and that CAMs are single-model visualizations while decisions come from the ensemble. | reports/app_example_probs.json (bus_0186-r cal 0.1662; bus_0663-l cal 0.6986); RESULTS.md:362–365 (example probabilities), :287–294 (TP heat on lesion body/margins; TN/FN CAMs diffuse, partly over burned-in text), :45–47 (all inspected images carry burned-in annotations), :53–55 (border-occlusion probe: median drop < 0.01, heavy tail ~16% > 0.2), :276–278 (fold-5 single-model CAM, ensemble decision) | corrected (honest composite) |

## F. RESULTS.md errata carried forward (not editable in P1) — resolved-by-P2

These sentences in `RESULTS.md` are wrong or imprecise per the audit and
were corrected in every other document in P1. In P2 (2026-09-07) they were
recorded in the appended section "Errata (2026-09-07)" at the end of
RESULTS.md without editing any existing line; `git diff 2cbe1da HEAD --
RESULTS.md` removes zero lines (verified after the P2 commit):

- RESULTS.md:403 — "1,879 BUS-BRA images" → 1,875 (the 4 bundled examples
  are BUS-BRA images counted twice).
- RESULTS.md:485 and :1058 — "≥ 0.92 everywhere" / "held sens ≥ 0.92 on
  all four" → ≥ 0.918 (BrEaST 90/98 = 0.9184).
- RESULTS.md:1028–1033 — "multi-source training reduces the domain-shift
  specificity collapse, carried by the AUC branch" → should carry the
  caveats in B-7.5 in the same paragraph.
- RESULTS.md:165–167 and :3–12 — per-fold CV AUCs and the "headline"
  0.9307 ± 0.0161 are best-epoch-on-the-reported-fold; sens 0.9028 / spec
  0.7713 are in-sample — to be labeled when P2 quantifies them.

## G. P2 additions requested by the author (2026-09-07)

| # | Item | Handling | Source | Status |
|---|---|---|---|---|
| G.1 | Restore the BUS-BRA dataset description in report §2.1 (722 benign / 342 malignant patients, four scanners, Brazilian National Cancer Institute) | Restored with an explicit attribution to the dataset publication; dataset facts from the source publication are permitted sources | Gómez-Flores et al., *Med Phys* 2024 (BUS-BRA release paper); the audit (§1) independently confirmed 722/342 from bus_data.csv | corrected |
| G.2 | RESULTS.md errata without editing any existing line | Final section "Errata (2026-09-07)" appended: line 403 (1,879 → 1,875), line 485 and line 1058 (≥ 0.92 → ≥ 0.918), lines 1028–1033 (Q1 wording with the branch-A caveats), plus a context note on lines 165–167 / 3–12 pointing to POST-HOC 1–2 | RESULTS.md lines cited in each erratum; `git diff 2cbe1da HEAD -- RESULTS.md` → 0 removed lines | resolved-by-P2 |
