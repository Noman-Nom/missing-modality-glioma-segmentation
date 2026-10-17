# writing/

## Chapter map
- No document exists yet. Format and template are unknown, ask the supervisor. Each chapter will state which RQs it answers.
- `refs.bib` is shared (empty now). `figures/` holds generated figures only.

## Voice and style
Short plain sentences. "We" for method, passive allowed in results. Pick one spelling variant per document, pending supervisor preference.

## Citations
Use keys from `refs.bib` only. No citation without a note in `docs/literature/`. Entries stay unverified until the `researcher` agent checks DOI/arXiv, authors, year, and that the claim is in the paper.

## Figures and tables
Generated from `results/`. Each has a caption, a label, and a sentence in the text referring to it.

## Terminology
Regions: WT (whole tumor), TC (tumor core), ET (enhancing tumor). Modalities: T1, T1ce, T2, FLAIR. "Missing pattern" means a fixed subset of available modalities. "Baseline" means the complete-input model unless stated.

## Do not
Pad, repeat the abstract verbatim in the conclusion, claim more than the evidence shows, leave `TODO` or `[CITE?]` in a submitted chapter, promise a metric value, call a cross-release test "external validation" without shown independence.
