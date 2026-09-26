# `src/info/`

Reference documentation used across the repo, plus a few data files the code reads. Check here
before writing up something that notebooks or code would otherwise each repeat.

## Documents

| File | What it covers |
|---|---|
| [WLASL_info.md](WLASL_info.md) | WLASL fields, facts about our copy of the data, gotchas, split/set naming, and what preprocessing did to each split |
| [plotting_conventions.md](plotting_conventions.md) | How to write and use `visualise2.py` plotting functions: signatures, colours, styling, legend placement, function catalog |
| [environment.md](environment.md) | The `wlasl` / `wlasl_cpu` conda envs, and the history of the detectron2 cleanup |

## Data files

Don't move or rename these without updating whatever reads them.

| File | Read by |
|---|---|
| `wlasl_class_list.json` | `run_types.CLASSES_PATH` (the class list for `configs.get_class_list`), and `src/media/disp_topk*.ipynb` |
| `wlasl_class_list.txt` | Nothing. A plain-text copy of the same list, for reading |

## Conventions for documents here

- **What goes here:** reference material used across the repo, such as dataset facts and repo-wide
  conventions. Docs about one package stay beside its code (e.g. `src/que/README.md`,
  `src/models/README.md`).
- **Say it once, link to it.** Notebooks and other docs point here with a one-line markdown cell
  or sentence, instead of copying the content. For a notebook in `src/results/<dir>/`:
  `See dataset information in [WLASL_info](../../info/WLASL_info.md).`
- **Numbers need a source and a date.** Any figure (counts, resolutions, versions) should say
  where it came from and when it was last checked, so a stale number can be spotted and
  refreshed.
- **Keep this index current.** Add a row when you add a document, and update it when one is
  renamed or its scope changes.
