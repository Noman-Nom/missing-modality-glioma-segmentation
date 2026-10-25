# writing/

## Chapter map
- `proposal/proposal.tex` is the 7 page proposal (Option B two-tone journal style, burgundy #7A1F3D, all RQs). Build: `cd writing/proposal && latexmk -xelatex proposal.tex`.
- `tracer/tracer.tex` is the tracer bullet stub; it only shows a generated table builds. Build: `cd writing/tracer && latexmk -xelatex tracer.tex`.
- `tables/` holds LaTeX tables written by `experiments/*/make_table.py`. Never edit them by hand.
- Thesis chapters are not created yet; university template unknown, ask. Each chapter will state which RQs it answers.
- `refs.bib` is shared (19 entries, all UNVERIFIED). `figures/` holds generated figures only.

## Voice and style
Short plain sentences. "We" for method, passive allowed in results. British spelling (tumour, oedema) in all documents.

## Citations
Use keys from `refs.bib` only. No citation without a note in `docs/literature/`. Entries stay unverified until the `researcher` agent checks DOI/arXiv, authors, year, and that the claim is in the paper.

## Figures and tables
Generated from `results/`. Each has a caption, a label, and a sentence in the text referring to it.

## Terminology
Regions: WT (whole tumor), TC (tumor core), ET (enhancing tumor). Modalities: T1, T1ce, T2, FLAIR. "Missing pattern" means a fixed subset of available modalities. "Baseline" means the complete-input model unless stated.

## Do not
Pad, repeat the abstract verbatim in the conclusion, claim more than the evidence shows, leave `TODO` or `[CITE?]` in a submitted chapter, promise a metric value, call a cross-release test "external validation" without shown independence.
