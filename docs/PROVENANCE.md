# Provenance of the BreastUS-CAD repository

Written 2026-09-07 (audit-response phase P4); §1, §4, §5 and §7 revised
2026-09-12 (P8', after the two independent re-audits of 2026-09-11,
docs/AUDIT2_claude_2026-09-11.md and docs/AUDIT2_astra_2026-09-11.md).
Research prototype — not a medical device. This document records, as plainly as possible, what the
repository's history can and cannot prove about the order of events
behind the pre-registered external validation, so that a reader does not
have to reconstruct it from git and the audit (docs/AUDIT_2026-09-06.md §3).

Every hash, date and timestamp below was read from the local git object
store, the local wandb run directories, or the Hugging Face API on
2026-09-07; dates are +08:00 unless marked Z (UTC). Nothing here has been
externally notarised except where §5 and §7 say so.

## 1. Summary

- The repository was **private** throughout development and at audit time
  (2026-09-06). It is **public as of 2026-09-07** at https://github.com/thomasf45566/breast-us-cad
  (pushed with tags after the Zenodo deposition below).
- All protocols and amendments were **internal, version-controlled
  pre-registrations** in `data/external_protocol.md`, and the file is
  append-only. Amendments 2/3/4 were each committed alone before the
  computation they govern; the base protocol was committed after the
  freeze and before the external run; Amendment 1 was committed together
  with the keep-lists and pHash artifacts it registers (436ab56, 10 files),
  i.e. after the data audit it describes and before any model metric on
  GDPH/SYSUCC. They were **not externally time-stamped** (no
  OSF/Zenodo/other registry entry) before the runs.
- On 2026-08-31 the author/committer identity of every earlier commit was
  rewritten with `git filter-branch`. Author and committer dates, trees
  and messages were preserved; §3 gives the complete pre→post mapping.
- The public artifacts created during the study — the Hugging Face
  weights repository (created 2026-08-30 04:25:49Z) and the Hugging Face
  Space (2026-08-30 09:24:53Z) — both postdate the external-v1 commit
  (2026-08-29 15:14:26Z). Neither can corroborate that the freeze preceded
  the external scoring.
- Consequently the freeze-before-test ordering rests on local, rewritten,
  self-assigned commit timestamps plus file mtimes and wandb metadata
  (all internally consistent, none third-party). This is a permanent
  limitation of the v1 line; §6 states what changes for every future study.

## 2. Tag and milestone timeline

| Tag / milestone | Commit | Author date | Note |
|---|---|---|---|
| `baseline-v1` (lightweight) | 2d38e90 | 2026-08-26 15:07:25 | data pipeline, effb0 baseline, split tests |
| `cv-v1` (lightweight) | 5e78373 | 2026-08-27 18:02:19 | 5-fold CV harness |
| calibration (T = 2.3644) | 25ef75a | 2026-08-29 18:33:03 | TTA adopted (not pre-registered) |
| **`frozen-v1`** (annotated; tagger 18:38:35) | f167665 | 2026-08-29 18:38:34 | operating point 0.2683; no model/T/threshold change after this |
| protocol (a)–(d) | a16a64f | 2026-08-29 18:49:09 | external protocol, BUSI keep-list, self-check |
| Amendment 1 | 436ab56 | 2026-08-29 21:44:18 | GDPH/SYSUCC added, BUSI_WHU excluded, pHash sweep (already contains its dedup counts) |
| **`external-v1`** (lightweight) | 2cbe1da | 2026-08-29 23:14:26 (15:14:26Z) | the single external run; the four preds CSVs have exactly one adding commit each and were never modified |
| Amendment 2 | 610e87f | 2026-08-31 16:46:51 | committed alone; first governed computation 17:55 (fbc4bba 17:56:44) |
| identity rewrite | 2c24500 | 2026-08-31 17:58:46 | see §3 |
| Amendment 3 | 78f3991 | 2026-08-31 20:27:20 | committed alone |
| Amendment 4 | 3faa70b | 2026-08-31 22:53:53 | committed alone |
| BiomedCLIP substitution note | 4fc3fb2 | 2026-09-01 18:38:49 | 20 min before the first BiomedCLIP run (wandb 830ejnrt 18:58:49) |
| `v2-biomedclip` (annotated) | 568fdac | 2026-09-02 14:34:40 | Q2 verdict NOT CLAIMED |
| `v2-loco` (lightweight) | 14c3bf2 | 2026-09-04 07:10:16 | Q1 verdict MET via branch A |
| independent audit archived | 7a40e27 | 2026-09-07 10:36:54 | docs/AUDIT_2026-09-06.md, unmodified |
| P1 documentation corrections | 5652f0a | 2026-09-07 10:59:35 | docs/AUDIT_RESPONSE_2026-09-06.md |
| P2 post-hoc analyses + errata | 35bdb43, f74647f | 2026-09-07 11:11:39, 11:13:21 | RESULTS.md append-only (0 lines removed vs 2cbe1da) |
| P3 provenance part 1 | 67915b0 | 2026-09-07 12:12:38 | v2 weights on HF under v2/, CHECKSUMS.txt, fold file committed |
| **`v1.0-audited`** (annotated, tagger 13:05:16; re-pointed once, §7) | fa193a8 | 2026-09-07 13:05:16 | Zenodo deposition 10.5281/zenodo.22630912 (§7.2) |
| P5–P6 (summary, interview script, consistency sweep) | … 4794ad6 | 2026-09-07 – 2026-09-11 | outside the v1.0 archive (14 paths changed after the tag) |
| independent re-audits archived | f7cbf38 | 2026-09-12 | docs/AUDIT2_claude_2026-09-11.md, docs/AUDIT2_astra_2026-09-11.md, unmodified |
| P8' corrections | (this document's commit and its successors) | 2026-09-12 | docs/AUDIT2_RESPONSE.md; POST-HOC 5–6; external_val overwrite guard |
| **`v1.1-audited`** (annotated) | hash in the archive's `.git-commit` | 2026-09-12 | Zenodo new version prepared, not yet published (§7.3) |

Git facts at HEAD of this document: 62 commits before the P4 commit, a
single author/committer identity, author date == committer date on all
but one commit (6c180ee, a 6-second amend; its pre-amend draft 8e058bd is
still an unreachable object, §3).

## 3. The 2026-08-31 identity rewrite

**What happened.** Between fbc4bba (2026-08-31 17:56:44) and 2c24500
(17:58:46) the author ran `git filter-branch` over all 30 prior commits to
replace the hostname-derived author/committer e-mail
(`…@Thomass-MacBook-Air.local`) with the author's real address. The
commit message of 2c24500 records this: "author/committer dates preserved,
tags carried over, frozen-v1 annotated tag re-created with the corrected
tagger identity at the same date. Nothing had been pushed."

**What changed:** author name/e-mail and committer name/e-mail on 29
commits (the 30th, d265b55 → fbc4bba, already carried the correct identity
and was rewritten only because its parent changed). Every commit hash
before fbc4bba therefore changed.

**What was preserved:** for all 30 commits the tree hash, the author date
and the committer date are identical between the pre-rewrite and
post-rewrite objects; the message is unchanged; the `frozen-v1` annotated
tag object was re-created with the tagger date 1 s after its commit.

**Evidence.**

1. The pre-rewrite commit objects are still present as *unreachable*
   objects in the author's working copy (`git fsck --unreachable
   --no-reflogs` lists 31 commits: the 30 rewritten ones plus the
   pre-amend draft of 6c180ee). They are **not** in any clone, archive or
   push, so this evidence is reproducible only from the author's disk.
2. The audit (§3) found pre-rewrite hashes in the local wandb run
   metadata (`wandb/run-*/files/wandb-metadata.json`, `git.commit`) for
   every run before 2026-08-31 and post-rewrite hashes after. The table
   below maps each of those hashes — and every other pre-rewrite commit —
   to its current counterpart by identical tree hash; author dates match
   to the second in all 30 cases (checked programmatically while
   generating the table).

| pre-rewrite hash | author date (preserved) | pre-rewrite author e-mail domain | current hash | subject |
|---|---|---|---|---|
| d0cd52c | 2026-08-26 15:07:25 +0800 | Thomass-MacBook-Air.local | 2d38e90 | Baseline: BUS-BRA data pipeline, effnet_b0 training/eval, EDA, split t |
| 51e237c | 2026-08-27 18:02:19 +0800 | Thomass-MacBook-Air.local | 5e78373 | 5-fold CV: cross_validate.py, train.py refactored into train_one_fold |
| e0e5650 | 2026-08-27 23:44:20 +0800 | Thomass-MacBook-Air.local | 12120c9 | Screening configs for convnext_small and vit_b16; warmup in build_sche |
| 1a02bd6 | 2026-08-27 23:44:29 +0800 | Thomass-MacBook-Air.local | 0ae8efc | Screening results: convnext_small and vit_b16 vs effb0 baseline on fol |
| b83090d | 2026-08-28 08:42:47 +0800 | Thomass-MacBook-Air.local | 322ffab | Prefix-aware CV summary path, ignore logs/; add plan.md tracker + CLAU |
| c9adedd | 2026-08-28 08:43:56 +0800 | Thomass-MacBook-Air.local | 0c23853 | CV results: convnext_small 5-fold AUC 0.930 ± 0.017 |
| 159a8f9 | 2026-08-28 20:41:07 +0800 | Thomass-MacBook-Air.local | cb53138 | CV results: vit_b16 5-fold AUC 0.931 ± 0.016; backbone comparison tabl |
| b1b3aae | 2026-08-28 20:42:54 +0800 | Thomass-MacBook-Air.local | f9b987d | plan.md: add convnext CV result to its checkbox |
| 0b77151 | 2026-08-28 20:52:50 +0800 | Thomass-MacBook-Air.local | df3f181 | Backbone comparison: merged table with params + measured CPU latency |
| b68e60e | 2026-08-28 22:11:19 +0800 | Thomass-MacBook-Air.local | 6cbed97 | Grad-CAM check: convnext vs vit fold-5 CV checkpoints |
| 914fc73 | 2026-08-28 22:26:19 +0800 | Thomass-MacBook-Air.local | a875641 | Interpretability audit: moderate-conf Grad-CAM, ViT rollout, border oc |
| a18733e | 2026-08-28 22:30:38 +0800 | Thomass-MacBook-Air.local | da718e6 | Interpretability audit: findings finalized, raw-image audit sheet adde |
| 82bd016 | 2026-08-28 22:31:00 +0800 | Thomass-MacBook-Air.local | a14990f | plan.md: backbone decision frozen — vit_base_patch16_224 |
| 60f3e07 | 2026-08-29 12:36:15 +0800 | Thomass-MacBook-Air.local | 5eefa3c | CoarseDropout experiment: DISCARDED per pre-registered rule — plain cv |
| fd00eda | 2026-08-29 17:18:14 +0800 | Thomass-MacBook-Air.local | 3c3407e | TTA (hflip only): --tta on evaluate.py + CV re-eval of cv_vit ckpts |
| 226af2e | 2026-08-29 18:33:03 +0800 | Thomass-MacBook-Air.local | 25ef75a | Calibration: temperature scaling on pooled OOF TTA preds; TTA adopted |
| dd1b9fd | 2026-08-29 18:38:34 +0800 | Thomass-MacBook-Air.local | f167665 | Operating point (sens>=0.90) + FREEZE frozen-v1 |
| 10bf7f4 | 2026-08-29 18:49:09 +0800 | Thomass-MacBook-Air.local | a16a64f | External validation pre-flight: protocol, BUSI dedup, pipeline self-ch |
| 3befcec | 2026-08-29 21:44:18 +0800 | Thomass-MacBook-Air.local | 436ab56 | Protocol Amendment 1: GDPH/SYSUCC cohorts, BUSI_WHU excluded, pHash sw |
| 7cc9414 | 2026-08-29 23:14:26 +0800 | Thomass-MacBook-Air.local | 2cbe1da | External validation single-shot: BrEaST/BUSI/GDPH/SYSUCC vs frozen-v1 |
| ed36292 | 2026-08-29 23:21:17 +0800 | Thomass-MacBook-Air.local | af7f003 | Prob-shift analysis of saved external preds: fig + table, RESULTS.md n |
| f8a435c | 2026-08-29 23:24:18 +0800 | Thomass-MacBook-Air.local | 5afac64 | Secondary analysis (protocol h): model vs BI-RADS readers on GDPH/SYSU |
| 402a193 | 2026-08-29 23:32:24 +0800 | Thomass-MacBook-Air.local | b213aa1 | Grad-CAM galleries (manual Step 10): internal TP/TN/FP/FN + external b |
| 6ec980f | 2026-08-30 01:11:09 +0800 | Thomass-MacBook-Air.local | a2fa18b | Lesion segmentation demo: U-Net effb0, fold 5 Dice 0.9016 / IoU 0.8326 |
| 663d91b | 2026-08-30 10:43:47 +0800 | Thomass-MacBook-Air.local | 27cd67c | Gradio demo app (Step 11): frozen-v1 pipeline, contour+CAM+result card |
| 9b664ff | 2026-08-30 12:16:25 +0800 | Thomass-MacBook-Air.local | 2859043 | Refactor frozen inference into src/inference.py + HF Spaces scaffoldin |
| 310f03d | 2026-08-30 17:51:09 +0800 | Thomass-MacBook-Air.local | f3170bc | Deploy demo to HF Spaces: live at happytommy/breast-us-cad |
| 79520e9 | 2026-08-31 16:46:51 +0800 | Thomass-MacBook-Air.local | 610e87f | Protocol Amendment 2: site-specific recalibration study (v2) — pre-reg |
| a7174f4 | 2026-08-31 16:47:08 +0800 | Thomass-MacBook-Air.local | d31d2dd | plan.md: add v2 line (site-specific recalibration study, per Amendment |
| d265b55 | 2026-08-31 17:56:44 +0800 | gmail.com | fbc4bba | v2 M1: site-specific recalibration learning curves (Amendment 2, prima |
| 8e058bd | 2026-09-02 04:26:23 +0800 | gmail.com | — (none) | v2 Q2b: BiomedCLIP 5-fold CV trained — mean val AUC 0.9170 ± 0.0220 |

The wandb-cited hashes are d0cd52c → 2d38e90, 51e237c → 5e78373,
1a02bd6 → 0ae8efc, c9adedd → 0c23853 (the five cv_vit runs that produced
the frozen checkpoints), 82bd016 → a14990f (the CoarseDropout runs) and
402a193 → b213aa1 (the segmentation run). The one unmapped unreachable
commit, 8e058bd (2026-09-02 04:26:23), is the pre-amend draft of 6c180ee:
same author date, committer date 6 s later, and the amend only added
`reports/v2_biomedclip_cv_summary.csv`.

**What the rewrite does not affect.** Rewriting identity does not alter
file contents (trees are identical), so no number, protocol text or
prediction CSV changed. It does, however, make every pre-2026-08-31
timestamp a self-assigned value in a history that was demonstrably
rewritten once; the preserved dates are consistent with the wandb run
directories' own timestamps (`wandb-metadata.json` `startedAt`, run
directory names) and with the file mtimes the audit inspected, none of
which are third-party either.

## 4. Pre-registration: internal only

- `data/external_protocol.md` was written before any external inference
  (a16a64f, 18:49 — after the freeze f167665 at 18:38) and amended four
  times; Amendments 2/3/4 were each committed alone before the computation
  they govern, Amendment 1 with its 10 data-audit artifacts (§2). The file
  is append-only: every amendment commit removes 0 lines, and
  `RESULTS.md`'s external-v1 section was never edited (`git diff 2cbe1da
  HEAD -- RESULTS.md` removes 0 lines; the P2 and P8' errata were appended,
  not inserted). A post-hoc compliance appendix was appended to the
  protocol on 2026-09-12 (deviations between the registered text and the
  implementation, all previously undeclared ones included).
- **No external timestamp exists for any of this before the Zenodo
  deposition of §7.** The ordering freeze → protocol → single run
  (f167665 18:38:34 → a16a64f 18:49:09 → 2cbe1da 23:14:26, all
  2026-08-29 +08) rests on the local commit dates (rewritten once, §3), the
  one-adding-commit history of the prediction CSVs, and the wandb metadata.
  `--confirm` in `src/external_val.py` is an intent flag: it authorizes
  execution and, at the time of the run, did NOT refuse existing outputs
  (a repeated command would have overwritten the four CSVs in place; the
  git history shows this never happened in any committed state, and cannot
  show what happened in uncommitted states). An output-exists refusal
  (`--overwrite` required) was added on 2026-09-12 as post-hoc enforcement.
  An adversary with the author's access could have produced all of these
  after the fact; the repository offers internal consistency, not proof.
- Amendment 1 is a pre-registration for model metrics only: at the time
  it was committed it already contained the dedup counts and the pHash
  outcome it registers (the data audit had been run first).
- TTA adoption (hflip) was not pre-registered; it was decided pre-freeze
  after seeing +0.0023 pooled OOF AUC (RESULTS.md TTA section).
- The CoarseDropout adoption rule entered plan.md at a14990f
  (2026-08-28 22:31:00), four minutes before the first CoarseDropout
  wandb run (22:35:07 local).

## 5. Hugging Face timestamps (third-party clock, read via the HF API on 2026-09-07)

| Event | Timestamp (UTC) | Local (+08:00) |
|---|---|---|
| external-v1 commit 2cbe1da (for comparison; local git clock) | 2026-08-29 15:14:26 | 2026-08-29 23:14:26 |
| weights repo `happytommy/breast-us-cad-weights` created | 2026-08-30 04:25:49 | 12:25:49 |
| first checkpoint uploads (cv_vit_fold1–4) | 2026-08-30 04:27:26 – 04:31:27 | 12:27 – 12:31 |
| cv_vit_fold5, seg_unet_effb0, calibration.json, operating_point.json | 2026-08-30 07:12:50 – 07:13:06 | 15:12 – 15:13 |
| Space `happytommy/breast-us-cad` created | 2026-08-30 09:24:53 | 17:24:53 |
| Space push (P1 About-text correction), sha b7878b3 | 2026-09-07 03:27:26 | 11:27:26 |
| weights repo commit 5ca5ba0: v2/ artifacts + CHECKSUMS.txt + card | 2026-09-07 04:11:35 | 12:11:35 |
| Space push (P8': inference.py with `hf_repo_path` and the 1,875 docstring fix, About text, licence sentence — the P3/P4b changes had never been pushed, so the live Space ran the P1 files until now), sha 7d41f22 | 2026-09-12 (HF commit time) | — |

The five frozen-v1 checkpoints uploaded on 2026-08-30 have the SHA-256
values recorded in `models/CHECKSUMS.txt` (verified against the HF LFS
metadata on 2026-09-07, 28/28 files; re-verified 28/28 by both re-audits).
This proves the weights that exist today are the ones uploaded on
2026-08-30 — 13 hours (fold1–4) to 16 hours (fold5, the segmentation
model and the two JSONs) after the external-v1 commit — not that they
existed before the external run.

## 6. Forward commitment

For every future protocol of this line — including the planned MAMA-MIA
work and the NTUH retrospective study (docs/report.md §4.6) — the
following applies, without exception:

1. The protocol text (task, cohorts, exclusions, preprocessing, metrics,
   decision rules, sample size where applicable) is **registered on OSF
   before any computation** on the data it governs, and the OSF
   registration DOI/URL and its timestamp are recorded in the protocol
   file and in RESULTS.md next to the first result it governs.
2. Every amendment is likewise registered on OSF before the computation
   it governs; the repository copy states the OSF timestamp.
3. Git history of this repository is not rewritten again; if an identity
   or message must be corrected, it is done in a new commit that says so.
4. Release states are deposited on Zenodo (this document's §7 is the
   first) so that each milestone has an external timestamp; checkpoints
   are published with SHA-256 checksums at the time of the freeze, not
   after the external run.

## 7. Deposition record (Zenodo)

### 7.1 How an archive identifies itself (changed 2026-09-12)

The v1.0 archive's own copy of this file, its README and its CITATION.cff
named a superseded commit (1f196a9), a superseded archive hash (fcd270a5…)
and "DOI pending": they were written before the tag was re-pointed, and
both 2026-09-11 re-audits flagged that the deposited file contradicts its
own provenance text. From v1.1 on the archive describes itself
differently:

- `.gitattributes` marks `.git-commit` with `export-subst`, so every
  `git archive` of a tag expands the file to the archived commit's full
  hash and committer date (in the working tree it holds the unexpanded
  `$Format:…$` placeholder). Check: `git rev-parse <tag>^{commit}` must
  equal the first line of `.git-commit` inside the archive.
- **This document does not state any archive checksum.** An archive cannot
  contain its own hash. The authoritative checksum of a deposited file is
  the one Zenodo computes and displays for that file (MD5 in the record's
  file list; the record API returns it too). Anyone can regenerate the
  archive and compare:

      git archive --format=zip --prefix=breast-us-cad-<version>/ <tag>

  (`git archive` is reproducible byte-for-byte for a given tag; the v1.0
  deposit was verified bit-identical to `git archive` of fa193a8 by both
  re-audits.)

### 7.2 v1.0-audited (published 2026-09-07)

- Tag `v1.0-audited` → `fa193a83aafc264c44a82a5e361900858585add4`
  (annotated, tag object 2e7da4d, tagger 2026-09-07 13:05:16; re-pointed
  once from 1f196a9 before any push or deposition, for the P4b/P4c
  licensing corrections; the earlier archive was never uploaded).
- Zenodo DOI **10.5281/zenodo.22630912** (concept DOI 10.5281/zenodo.22630911),
  created 2026-09-07T07:30:33Z; one file, `breast-us-cad-v1.0.zip`,
  26,404,667 B; checksum as displayed on the record.
- **Known defect of that deposit (cannot be edited in place):** its
  embedded `docs/PROVENANCE.md` §7, `README.md` and `CITATION.cff` describe
  the archive as commit 1f196a9 with archive SHA-256 fcd270a5… and "DOI
  pending". The file is in fact the archive of fa193a8. The erratum is
  recorded in the description of the v1.1 version of the record
  (docs/zenodo_v1.1_new_version.md) and in docs/AUDIT2_RESPONSE.md.
- The archive's `.zenodo.json` predates the ORCID addition (063a25a); the
  DOI record metadata is authoritative.
- 14 paths changed after the tag (P5, P6, the ORCID); those states are not
  in the v1.0 deposit — the DOI notarises fa193a8 only.

### 7.3 v1.1-audited (prepared 2026-09-12 — NOT yet published)

- Tag `v1.1-audited` (annotated): the P8' state — both 2026-09-11
  re-audits archived unmodified, the corrections in docs/AUDIT2_RESPONSE.md,
  POST-HOC 5–6, the external_val overwrite guard, the protocol compliance
  appendix, this section. Commit hash: `.git-commit` inside the archive;
  `git rev-parse v1.1-audited^{commit}` in the repository.
- Metadata: `CITATION.cff` and `.zenodo.json` (version 1.1-audited,
  2026-09-12); `docs/zenodo_v1.1_new_version.md` carries the description
  text for the new version and the one-sentence erratum for the v1.0
  description.
- The deposition is a NEW VERSION of the same concept record; it is
  uploaded only after the author reviews the archive. The version DOI is
  recorded here, in CITATION.cff and in README after publication; until
  then the concept DOI resolves to v1.0.
- Repository visibility: public since 2026-09-07
  (https://github.com/thomasf45566/breast-us-cad), pushed with tags.
