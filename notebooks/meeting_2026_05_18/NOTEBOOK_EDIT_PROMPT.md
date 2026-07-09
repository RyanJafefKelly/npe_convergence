# Session prompt: edit the coauthor meeting notebook

Paste this into a new Claude Code session, or point the session at this file.
It is self-contained.

## Repo and task

Repo: `/Users/ryankelly/python_projects/npe_convergence`. Your task is to make
a set of edits to the coauthor meeting results-summary notebook, then
regenerate it. Work with Ryan; he confirms the exact edit list.

## Critical: the notebook is generated, do not hand-edit the .ipynb

`notebooks/meeting_2026_05_18/empirical_results_summary.ipynb` is produced by
`notebooks/meeting_2026_05_18/build_notebook.py`. A hand edit to the `.ipynb`
is wiped the next time the script runs. Every edit goes in `build_notebook.py`,
then you regenerate the notebook and the PDF.

## Files

- `notebooks/meeting_2026_05_18/build_notebook.py` - the generator. Edit this.
- `notebooks/meeting_2026_05_18/empirical_results_summary.ipynb` - the built
  notebook, 32 cells. A build output.
- `notebooks/meeting_2026_05_18/empirical_results_summary.pdf` - the PDF with
  code hidden. A build output.
- `notebooks/meeting_2026_05_18/data/` - staged and generated CSVs the notebook
  reads. The built notebook depends only on this folder.
- `notebooks/meeting_2026_05_18/discussion_notes.md` - Ryan's scratch notes,
  background only.

## How build_notebook.py is structured

Read the whole file first (about 616 lines):
- Constants and a `SOURCES` dict at the top, plus `GNK_OVERLAY_CELLS`.
- `stage_inputs()` copies upstream CSVs into `data/`.
- `build_gnk_overlay()`, `aggregate_ma2_b0_kl()`, `build_ma2_bivariate()`
  generate further CSVs into `data/` from result data in `res/`.
- `aggregate_flow_hexadecile()` is defined (near lines 102 to 118) but is not
  called in `main()`.
- `build_notebook()` assembles the notebook with `md()` and `code()` cell
  helpers, section by section: g-and-k, stereological, MA(2), summary.
- The `__main__` block runs the staging and generation functions, then
  `build_notebook()`.

## Regenerate after editing

From the repo root, with the project virtualenv:

```bash
.venv/bin/python notebooks/meeting_2026_05_18/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/meeting_2026_05_18/empirical_results_summary.ipynb
.venv/bin/jupyter nbconvert --to pdf --no-input \
  notebooks/meeting_2026_05_18/empirical_results_summary.ipynb
```

The virtualenv has `nbconvert`, `nbformat`, and the LaTeX `titling` package
installed.

## The edits

Ryan gives you the exact list. If this section has not been filled in, ask him
before changing anything. The candidates already identified are below; Ryan
may want some, all, none, or other changes from his own review.

**Candidate 1 - stereological bias boxplot, 12 panels down to lambda only.**
Near lines 387 to 388: change `params = ["lambda", "sigma", "xi"]` to
`params = ["lambda"]`, and change `plt.subplots(3, 4, figsize=(13, 8.2))` to a
single-row grid. Use `plt.subplots(1, 4, figsize=(13, 3.2), squeeze=False)` so
the existing `axes[r, c]` indexing and the `axes[0, 0]` legend lines keep
working. The intro markdown for that cell describes the rows by parameter, so
update its wording too.

**Candidate 2 - a g-and-k bivariate corner plot over the 6 parameter pairs.**
No code exists for this. Add a generator function modelled on
`build_ma2_bivariate()` (near lines 141 to 165): read g-and-k posterior draws,
write a CSV into `data/`. Call it from `main()`. Then add a notebook cell that
plots the 6 parameter pairs. The g-and-k posterior pickles are in `res/gnk/`;
`build_gnk_overlay()` and `GNK_OVERLAY_CELLS` show the path pattern and which
cells and seeds are clean to use.

**Candidate 3 - a faithful flow-NPE octile-vs-hexadecile figure.**
The current octile-vs-hexadecile cell (near lines 284 to 302) plots
Gaussian-NPE only. `aggregate_flow_hexadecile()` already exists to produce the
flow hexadecile CSV but is never called. Call it from `main()`, then add the
flow-NPE line to that cell. Watch the column names: the Gaussian CSV uses
`finite_kl_median`, while `aggregate_flow_hexadecile()` writes `kl_median`;
make the plotting code use the right column for each. The flow hexadecile
result data is in `res/gnk_hexadeciles/`.

## Constraints

- Plain language in any notebook text. No em-dashes or en-dashes, no
  LLM-cliche phrasing. Match the existing notebook's wording.
- Do not change the result numbers. The tables and figures report real
  experimental results; change presentation only, never values.
- Keep the scope to `build_notebook.py` and the notebook. Do not modify `res/`
  or other parts of the repo.
- `build_notebook.py` will later seed a reproducibility pipeline, so keep it
  clean and readable.

## When done

Confirm the notebook executes top to bottom with no errors and the PDF renders
with every figure. Show Ryan the rebuilt notebook or PDF and have him eyeball
the figures before wrapping up.
