# Zenodo — new version 1.1-audited (PREPARED 2026-09-12; NOT PUBLISHED)

> **Status (2026-09-12): PUBLISHED** — version DOI 10.5281/zenodo.22725965
> (concept DOI 10.5281/zenodo.22630911). The checklist below is kept as
> written; the post-publication steps are recorded in docs/PROVENANCE.md §7.3.

Research prototype — not a medical device. This note is the author's
checklist for depositing tag `v1.1-audited` as a **new version** of the
existing Zenodo record (concept DOI 10.5281/zenodo.22630911; v1.0 version
DOI 10.5281/zenodo.22630912). Nothing has been uploaded. The steps that
publish are marked STOP.

## Archive

    git archive --format=zip --prefix=breast-us-cad-v1.1/ v1.1-audited > breast-us-cad-v1.1.zip

- Inside the archive `.git-commit` holds the archived commit hash and
  committer date (expanded by `export-subst`, `.gitattributes`); check
  that its first line equals `git rev-parse v1.1-audited^{commit}`.
- Do not record the archive's checksum in any file inside the archive; the
  checksum Zenodo displays for the uploaded file is authoritative
  (docs/PROVENANCE.md §7.1). Compare it against the local file before
  publishing.

## Metadata for the new version (mirrors .zenodo.json)

- Title: unchanged.
- Version: `1.1-audited`. Publication date: 2026-09-12 (or the upload date).
- Description: the `.zenodo.json` description text (it already contains
  the version-1.1 paragraph and the erratum sentence below).
- Related identifiers: unchanged (GitHub isSupplementTo; HF weights and
  Space isSupplementedBy; cites 10.1002/mp.16812 and 10.5281/zenodo.8231412).
- Licence apache-2.0; creator with ORCID 0009-0009-3066-3298.

## One-sentence erratum to add to the v1.0-audited version's description

> Erratum (2026-09-12): the files inside this archive — docs/PROVENANCE.md
> §7, README.md and CITATION.cff — describe the archive as commit 1f196a9
> with SHA-256 fcd270a5… and "DOI pending"; those lines were written before
> the tag was re-pointed, and the deposited file is in fact the archive of
> commit fa193a8 (verified bit-identical by two independent re-audits on
> 2026-09-11); version 1.1-audited supersedes it.

(Zenodo allows editing a published version's metadata without a new DOI;
the file itself cannot be edited.)

## After publishing (STOP — author's decision)

1. Upload the archive as a new version; copy the new version DOI.
2. Record the version DOI in docs/PROVENANCE.md §7.3, README "Citation"
   and CITATION.cff `identifiers` (a commit after the tag; the archive
   itself cannot contain it).
3. Add the erratum sentence above to the v1.0 version's description.
4. Push with tags.
