"""
Patch notebook cell 58 to:
  1. Add pca_en_pc_hist collector
  2. Track selected PCA components per step in the loop
  3. Append [] to pca_en_pc_hist in except block
  4. Replace naming block to use both AR lags and PCA component history

Run with:
    /Users/zimo/miniconda3/envs/bt3102/bin/python patch_model_name.py
"""

import json

NB_PATH = "/Users/zimo/mmforecasting/notebooks/forecasting_analysis.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Helper ────────────────────────────────────────────────────────────────────
def find_idx(marker: str) -> int:
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if marker in src:
            return i
    raise ValueError(f"Marker not found: {marker!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH cell 58 — PCA+EN backtest
# ═══════════════════════════════════════════════════════════════════════════════
idx58 = find_idx("pca_en_ar_hist  = []    # selected AR lag indices per step")
print(f"Patching cell {idx58} (PCA+EN backtest)...")

src58 = "".join(cells[idx58]["source"])

# 1. Add pca_en_pc_hist initialisation next to pca_en_ar_hist
src58 = src58.replace(
    "pca_en_ar_hist  = []    # selected AR lag indices per step",
    "pca_en_ar_hist  = []    # selected AR lag indices per step\n"
    "pca_en_pc_hist  = []    # selected PCA component indices per step",
)

# 2. Add PC tracking after AR tracking (inside the try block)
src58 = src58.replace(
    "        # Selected AR lags (first N_AR_LAGS_EX coefs in X_cand)\n"
    "        ar_coef = en.coef_[:N_AR_LAGS_EX]\n"
    "        sel_ar  = [j + 1 for j, c in enumerate(ar_coef) if abs(c) > 1e-10]\n"
    "        pca_en_ar_hist.append(sel_ar)",
    "        # Selected AR lags (first N_AR_LAGS_EX coefs in X_cand)\n"
    "        ar_coef = en.coef_[:N_AR_LAGS_EX]\n"
    "        sel_ar  = [j + 1 for j, c in enumerate(ar_coef) if abs(c) > 1e-10]\n"
    "        pca_en_ar_hist.append(sel_ar)\n"
    "\n"
    "        # Selected PCA components (next k coefs in X_cand)\n"
    "        pc_coef = en.coef_[N_AR_LAGS_EX:N_AR_LAGS_EX + k]\n"
    "        sel_pc  = [j + 1 for j, c in enumerate(pc_coef) if abs(c) > 1e-10]\n"
    "        pca_en_pc_hist.append(sel_pc)",
)

# 3. Add pca_en_pc_hist.append([]) in except block
src58 = src58.replace(
    "        mu_hat, sigma_hat, sel_ar = _last_mu_pe, _last_sig_pe, []\n"
    "        pca_en_ar_hist.append(sel_ar)",
    "        mu_hat, sigma_hat, sel_ar = _last_mu_pe, _last_sig_pe, []\n"
    "        pca_en_ar_hist.append(sel_ar)\n"
    "        pca_en_pc_hist.append([])",
)

# 4. Replace naming block at end of cell
OLD_NAMING = (
    "# Derive canonical model name from AR lag selection history\n"
    "_all_ar = [l for lags in pca_en_ar_hist for l in lags]\n"
    "_cnt    = Counter(_all_ar)\n"
    "_freq   = sorted(l for l, c in _cnt.items() if c > 0.5 * TEST_SIZE)\n"
    "_lag_str_pe = \",\".join(str(l) for l in _freq) if _freq else \"none\"\n"
    "PCA_EN_NAME = f\"ARX[{_lag_str_pe}]-GARCH(1,1) | PCA(k={PCA_K})+EN\""
)

NEW_NAMING = (
    "# Derive canonical model name from AR lag and PCA component selection history\n"
    "_all_ar  = [l for lags in pca_en_ar_hist for l in lags]\n"
    "_cnt_ar  = Counter(_all_ar)\n"
    "_freq_ar = sorted(l for l, c in _cnt_ar.items() if c > 0.5 * TEST_SIZE)\n"
    "\n"
    "_all_pc  = [p for pcs in pca_en_pc_hist for p in pcs]\n"
    "_cnt_pc  = Counter(_all_pc)\n"
    "_freq_pc = sorted(p for p, c in _cnt_pc.items() if c > 0.5 * TEST_SIZE)\n"
    "\n"
    "_ar_str    = f\"AR({','.join(str(l) for l in _freq_ar)})\" if _freq_ar else \"\"\n"
    "_pc_str    = f\"X({','.join(f'PC{p}' for p in _freq_pc)})\" if _freq_pc else \"\"\n"
    "_mean_part = (_ar_str + _pc_str) if (_ar_str or _pc_str) else \"intercept-only\"\n"
    "PCA_EN_NAME = f\"{_mean_part}-GARCH(1,1) | PCA(k={PCA_K})+EN\""
)

if OLD_NAMING in src58:
    src58 = src58.replace(OLD_NAMING, NEW_NAMING)
    print("  ✓ Naming block replaced")
else:
    print("  ✗ WARNING: naming block not found — manual check needed")
    # Show what's near the end for debugging
    print("  Last 500 chars of cell:", repr(src58[-500:]))

cells[idx58]["source"] = src58
print(f"  Cell {idx58} patched OK")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✓ Notebook saved.")
print("Verify by running the notebook cell 58, then check PCA_EN_NAME output.")
