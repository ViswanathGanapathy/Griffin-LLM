# LoG 2026 Submission — LaTeX Skeleton

**Title:** Depth, Domain, and Reproducibility: A Multi-Seed Study of SMPNN in Relational Foundation Models
**Track:** Extended Abstract (4-page limit)
**Status:** Anonymous submission (review mode)

## Files

| File | Purpose |
|---|---|
| `main.tex`           | Main paper source |
| `reference.bib`      | Bibliography (5 core citations, 2 commented out for later) |
| `log_2026.sty`       | Official LoG 2026 style file (copied from conference template) |
| `log_2026_reference.pdf` | Reference PDF of the LoG 2026 template — sample output for comparison |

## Upload to Overleaf

1. Zip everything in this directory (`log_2026_submission.zip` in the parent folder is already prepared for you)
2. Overleaf → **New Project → Upload Project** → drop the zip
3. Overleaf will auto-detect `main.tex`. If not, right-click `main.tex` → *Set as Main Document*
4. Ensure the compiler is set to **pdfLaTeX** (Overleaf → Menu → Compiler)
5. Compile — the first run also runs BibTeX, so you may need to click *Recompile* twice for citations to resolve

## Track selection

Currently set for anonymous submission to the extended-abstract track:
```latex
\usepackage[review,eabstract]{log_2026}
```

To switch modes, edit `main.tex` and uncomment the appropriate line:
- `[review,eabstract]` — anonymous submission, extended-abstract track (current)
- `[eabstract]`         — camera-ready extended abstract
- `[review]`            — anonymous submission, full proceedings track (9 pages)
- `[preprint]`          — non-anonymous preprint version

## Author block

Currently anonymised. For camera-ready, replace the `\author` block near the top of `main.tex` — a template with real fields is provided directly below the anonymous version and just needs uncommenting + filling in.

## Page limits

- **Extended abstract track: 4 pages of main text** (excluding references and appendix, per LoG guidelines)
- Appendix is unlimited but reviewers may not read all of it — keep essential evidence in the main body

The current main body sections (1–5) should compile to ~4 pages. Verify after first compile — if it overflows, easiest cuts are:
1. Shorten §5 Discussion (currently 2 paragraphs + limitations)
2. Trim §3 Setup dense inline text
3. Move Table 1 into a `\small` size (currently already `\small`)

## Fill-in checklist before submission

- [ ] Real author names + affiliations (only for `[eabstract]` mode — do NOT reveal for `[review,eabstract]`)
- [ ] Verify all citations resolve (check compile log for undefined references)
- [ ] Update Appendix D pending experiments once results land
- [ ] Add α-trajectory figure to Appendix if experiment 3 completes
- [ ] Proofread abstract length — LoG guidance is "4-6 sentences" but extended abstracts often get away with a single dense paragraph (as ours does)

## Compile locally (optional)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Produces `main.pdf`.
