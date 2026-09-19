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
- Legend: default `loc="upper left"`. **The legend must never occlude a bar/line/marker** — after
  placing it, check the rendered figure (not just the code) for overlap. If the default corner
  overlaps data, try another `loc` corner first; if no corner is clear (e.g. grouped/stacked bar
  charts with many bars, or a bar that spans the full plot height), place it outside via
  `bbox_to_anchor=(1.02, 1), borderaxespad=0.0`.

## Function catalog

| Function | Use for | Notes |
|---|---|---|
| `plot_bar_chart` | Single-series bar chart (e.g. config vs. test loss) | `horizontal=True` for long category labels |
| `add_iqr_lines` | Overlay mean±std and lower/upper quartile lines on a `plot_bar_chart` axes (e.g. variance across seeds) | Computes stats from raw values; calls `ax.legend(...)` itself, placed outside the axes |
| `plot_grouped_bar_chart` | Multiple series per category (e.g. top-1/5/10 acc per split) | |
| `plot_stacked_bar_chart` | Series summing to a per-category total (e.g. train/test/val instance counts) | `value_fmt` defaults to `COUNT_FMT` |
| `plot_loss_curves` | One line per series over a shared x-axis (e.g. train/val loss curves) | NaNs dropped per-series so lines stay continuous |
| `plot_metric_correlation` | Scatter of a metric against another variable, with linear fit + Kendall's tau (e.g. per-gloss F1 vs. instance/signer count) | `shade_by_density` toggles density-shaded vs. flat scatter; `y_clip` (e.g. `(0, 1)` for F1) truncates the fit line's x-range at the clip bound rather than clamping y (avoids a flat horizontal segment); legend is placed `loc="lower right"` **inside** the axes -- a deliberate exception to the outside-axes legend convention below, since a positive-trend scatter typically leaves that corner clear (verify against the rendered figure, don't assume); `legend_top_y` raises the legend so its top edge aligns with a given y-axis data value, to clear a cluster of low points still poking into the bottom-right corner |
| `save_fig` | Save any of the above figures | Creates parent dirs; use instead of `fig.savefig` directly |
| `suggest_palette` | Generate a per-category colour list for `plot_bar_chart` | Recognises `"baseline"`/`"no_aug"` as controls |
| `split_name_mapper` | Map a raw split key (`"asl100_cutoff_9"`) to its display name (`"WLASL-100"`) | Backed by `SPLIT_NAME_MAP` |
| `plot_bboxes_on_canvas` | Per-class average (or raw) bbox outlines on a blank frame-sized canvas | One colour per class via `tab20`; not a `suggest_palette` case (too many categories) |
| `plot_dimension_distributions` | Bbox width/height histograms with mean/median/quartile lines | Returns `(fig, axes)` with `axes` a length-2 array -- the one exception to the single-`ax` return convention |
| `plot_frame_grid` | Grid of evenly-sampled video frames (e.g. example clips, per-gloss prediction/misprediction comparisons) | Returns `(fig, axes)` with `axes` a 2D (rows x cols) array -- another exception to the single-`ax` return convention, since it's inherently a grid of subplots; no `ax` parameter for the same reason. Supersedes `utils.plt_display_grid` for new code -- that function predates this convention and lacks the `save_fig`-friendly `(fig, axes)` return (it saves directly via its own `output` param instead) |

Update this table whenever a function is added, renamed, or its purpose changes.
