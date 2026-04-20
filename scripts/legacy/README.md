# Legacy Scripts — Audit Archive

One-off migration scripts used during the Phase 0–1 notebook restructure (April 2026). Retained for audit trail; **not imported by any active code path** and not on `sys.path` for the notebook.

| Script | What it did |
|---|---|
| `restructure_notebook.py` | Rewrote `notebooks/forecasting_analysis.ipynb` structure once during Phase 0. |
| `inject_cells.py` | Inserted the initial ARIMAX-GARCH cells into the notebook. |
| `add_fred_cells.py` | Added Phase 1 extended FRED cells. |
| `update_arimax_cells.py` | Mass-updated the ARIMAX cells to match the new `src/arimax_models.py` API. |
| `patch_model_name.py` | Renamed model labels consistently across cells. |
| `exog_pipeline.py` | Early draft of what became `src/fred_pipeline.py`. Do not import. |
| `test_arimax.py` | Smoke test for early ARIMAX — superseded by notebook cell 82 diagnostics. |

See `.claude/journal/changelog.md` entries dated 2026-04-09 through 2026-04-10 for the corresponding refactor milestones.

**Delete when:** the audit window closes (≥ 6 months after Phase 2.5 ships to production).
