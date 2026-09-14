# `src/visualise2.py` conventions

Reference for adding to or reusing [`visualise2.py`](visualise2.py). Read this before writing new
plotting code for a thesis figure, so the signature/colour/styling conventions below don't have to
be re-derived from the source (or from `git log`/old notebooks) each time.

## Before writing a new chart function

Check the [function catalog](#function-catalog) below first — a small parameter addition to an
existing function (e.g. a new `color`/`shade_by_density`-style toggle) is usually enough; a new
top-level function should be the exception, not the default. This mirrors the repo-wide rule (see
root `CLAUDE.md`, "Notebooks in `src/results` need updating") of checking for an existing
convention/notebook before writing new code.

Reference implementations (no `todo.txt` — these are the current convention, not legacy):
`src/results/aug_comparison`, `src/results/satnac_2026`. Directories *with* a `todo.txt` are still
mid-migration — don't copy patterns from them.

## Signature conventions

- **Operate on plain arrays, not DataFrames.** Take `ArrayLike` (or `dict[str, ArrayLike]` for
  multi-series charts, e.g. `plot_loss_curves`' `ys`), not a `df` + column-name pair. The caller
  does the data extraction/filtering (merges, groupbys, dict lookups); the plotting function stays
  reusable outside any one DataFrame schema.
- **`ax: ... = None` is always the last parameter.** If `None`, create
  `fig, ax = plt.subplots(figsize=figsize)`; otherwise reuse `fig = ax.figure`. This lets callers
  compose multiple charts onto one figure's subplots.
- **Always return `(fig, ax)`**, or `(fig, ax, *derived_values)` when the function also computes a
  reportable statistic the caller will want to log/print (e.g. `plot_metric_correlation` returns
  `(fig, ax, tau, p_value)` so the caller doesn't need to recompute Kendall's tau separately).
- **No `save_path` parameter.** Saving is a separate, explicit step at the call site via
  `save_fig(fig, path)` — keeps the plotting function itself pure and independent of I/O.
- **`fig.tight_layout()` right before returning.**
- **Docstring style:** one-paragraph description, then a plain bullet per non-obvious parameter
  (what it means / when to use it) — no numpy/Google-style `Args:`/`Returns:` headers.

## Colour conventions

- `DEFAULT_ACCENT` (`#0072B2`) — the default single-series colour.
- `LINE_PALETTE` — the Okabe-Ito colourblind-safe palette, for multi-series charts (lines, grouped/
  stacked bars). Cycle with `LINE_PALETTE[i % len(LINE_PALETTE)]` if there may be more series than
  colours.
- `CONTROL_COLORS` / `suggest_palette()` — only for bar charts that need to visually distinguish a
  "control" category (matched case-insensitively against `"baseline"`/`"no_aug"`) from treatment
  bars.
- Don't hardcode ad hoc hex colours in a notebook. Pick from the above, or add a new named constant
  to `visualise2.py` if there's a genuinely new, reusable need — don't invent a one-off colour
  inline.

## Formatting constants

Reuse `VALUE_FMT` (`"%.2f"`), `LOSS_FMT` (`"%.3f"`), `COUNT_FMT` (`"%d"`), and `FIGSIZE`
(`(8, 4.5)`) rather than re-deriving the same printf strings/figure size inline in a notebook.

## Styling

- Call `set_thesis_style()` exactly once per notebook/script, near the top, before any plotting.
- Grid: dashed line, `alpha=0.3`. Restrict to the axis with meaningful variation for bar-style
  charts (`axis="y"` for vertical bars, `axis="x"` for `horizontal=True`); scatter/correlation
  charts grid both axes.
- Legend: default `loc="upper left"`. For grouped/stacked bar charts where the legend could overlap
  many bars, place it outside via `bbox_to_anchor=(1.02, 1), borderaxespad=0.0`.

## Function catalog

| Function | Use for | Notes |
|---|---|---|
| `plot_bar_chart` | Single-series bar chart (e.g. config vs. test loss) | `horizontal=True` for long category labels |
| `plot_grouped_bar_chart` | Multiple series per category (e.g. top-1/5/10 acc per split) | |
| `plot_stacked_bar_chart` | Series summing to a per-category total (e.g. train/test/val instance counts) | `value_fmt` defaults to `COUNT_FMT` |
| `plot_loss_curves` | One line per series over a shared x-axis (e.g. train/val loss curves) | NaNs dropped per-series so lines stay continuous |
| `plot_metric_correlation` | Scatter of a metric against another variable, with linear fit + Kendall's tau (e.g. per-gloss F1 vs. instance/signer count) | `shade_by_density` toggles density-shaded vs. flat scatter |
| `save_fig` | Save any of the above figures | Creates parent dirs; use instead of `fig.savefig` directly |
| `suggest_palette` | Generate a per-category colour list for `plot_bar_chart` | Recognises `"baseline"`/`"no_aug"` as controls |
| `split_name_mapper` | Map a raw split key (`"asl100_cutoff_9"`) to its display name (`"WLASL-100"`) | Backed by `SPLIT_NAME_MAP` |
| `plot_bboxes_on_canvas` | Per-class average (or raw) bbox outlines on a blank frame-sized canvas | One colour per class via `tab20`; not a `suggest_palette` case (too many categories) |
| `plot_dimension_distributions` | Bbox width/height histograms with mean/median/quartile lines | Returns `(fig, axes)` with `axes` a length-2 array -- the one exception to the single-`ax` return convention |

Update this table whenever a function is added, renamed, or its purpose changes.
