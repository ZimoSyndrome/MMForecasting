"""
Gaussian HMM latent-regime pipeline.

Builds a small set of *learned* regime features from the target asset's own
(return, 5-day rolling volatility) process. Contrasts with
`regime_pipeline.py`, which derives a regime label from three exogenous
macro stress channels via a handcrafted tercile-cut composite.

Observables
-----------
2-D per day:
    r_t           : daily log-return of the target asset
    sigma_t^{(5)} : 5-day rolling standard deviation of r_t

Model
-----
hmmlearn.hmm.GaussianHMM with covariance_type="full". Each state gets its
own 2x2 covariance, so return-vol coupling (the leverage effect) can
differ by state. The number of states n in a caller-supplied grid is
chosen ONCE by BIC on the first training window. n is then FIXED for the
remainder of the walk-forward run. The reason is the feature-column
contract downstream. LSTM and XGBoost expect a stable input width across
rolling refits, and letting BIC flip n mid-run would break that.

Canonical state ordering
------------------------
hmmlearn state labels are arbitrary per fit. After every fit we sort
states by the fitted mean of the first observable (return) ascending, so
state 0 is the lowest mean ("bear") and state n-1 is the highest ("bull").
`fit_hmm_canonical` returns the fitted model paired with a permutation
vector. `run_hmm_rolling` applies that permutation so posterior columns
remain stable across refits.

Output features (all caller-shifted by 1)
-----------------------------------------
regime_state       : argmax of posterior, integer in {0, ..., n-1}
regime_prob_bear   : P(state = 0 | obs_{1:t})
regime_prob_bull   : P(state = n-1 | obs_{1:t})
regime_transition  : 1 if regime_state_t != regime_state_{t-1}, else 0

The middle-state probability (n=3) is omitted because it's redundant
under the simplex constraint sum_k P(state=k) = 1.

Validation note
---------------
`macro_utils.validate_features` is NOT applicable here. regime_state is
integer-valued and posteriors are bounded [0, 1]. Neither is meaningful
under an Augmented Dickey-Fuller stationarity gate. The HMM discipline
(rolling refit plus shift(1)) is the substitute invariant.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM

from macro_utils import _to_date_index


HMM_DEFAULT_STATE_GRID = (2, 3, 4)
HMM_DEFAULT_WINDOW = 504
HMM_DEFAULT_REFIT_EVERY = 20
HMM_DEFAULT_N_ITER = 100
HMM_OUTPUT_COLUMNS = [
    "regime_state",
    "regime_prob_bear",
    "regime_prob_bull",
    "regime_transition",
]


# ─── BIC selection ────────────────────────────────────────────────────────────

def _gaussian_hmm_bic(model: GaussianHMM, X: np.ndarray) -> float:
    """BIC = -2 * logL + k * log(n).

    Parameter count k for GaussianHMM(covariance_type="full"):
        means:       n_states * n_features
        covariances: n_states * n_features * (n_features + 1) / 2
        transmat:    n_states * (n_states - 1)       (rows sum to 1)
        startprob:   n_states - 1                    (sums to 1)
    """
    n_obs, n_feat = X.shape
    n = model.n_components
    k_means = n * n_feat
    k_cov = n * n_feat * (n_feat + 1) // 2
    k_trans = n * (n - 1)
    k_start = n - 1
    k = k_means + k_cov + k_trans + k_start
    logL = model.score(X)
    return -2.0 * logL + k * np.log(n_obs)


def select_n_states_by_bic(
    X_train: np.ndarray,
    state_grid: tuple = HMM_DEFAULT_STATE_GRID,
    covariance_type: str = "full",
    random_state: int = 42,
    n_iter: int = HMM_DEFAULT_N_ITER,
) -> int:
    """Return the n in state_grid that minimises BIC on X_train.

    Ties broken toward the smaller n (parsimony).  On fit failure for a
    candidate n, a warning is emitted and that n is skipped.  Returns
    the smallest grid entry if every candidate fails.
    """
    best_n = state_grid[0]
    best_bic = np.inf
    for n in state_grid:
        try:
            m = GaussianHMM(
                n_components=n,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state,
            )
            m.fit(X_train)
            bic = _gaussian_hmm_bic(m, X_train)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception as e:
            warnings.warn(f"HMM BIC: fit failed for n_states={n}: {e}")
    return int(best_n)


# ─── Fit + canonical ordering ─────────────────────────────────────────────────

def fit_hmm_canonical(
    X_train: np.ndarray,
    n_states: int,
    covariance_type: str = "full",
    random_state: int = 42,
    n_iter: int = HMM_DEFAULT_N_ITER,
) -> Tuple[GaussianHMM, np.ndarray]:
    """Fit a GaussianHMM and return (model, state_perm).

    state_perm[i] gives the raw hmmlearn state index that should be
    relabelled to canonical index i. Canonical ordering is by ascending
    mean of the first observable (return). State 0 is the lowest-mean
    (bear-like) and state n-1 is the highest-mean (bull-like).

    Callers apply state_perm to any state-indexed output (posteriors,
    predicted states) so downstream features are stable across refits.
    """
    m = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    m.fit(X_train)
    ret_means = m.means_[:, 0]
    state_perm = np.argsort(ret_means)  # ascending
    return m, state_perm


def _apply_perm_to_posteriors(posteriors: np.ndarray, state_perm: np.ndarray) -> np.ndarray:
    """Reorder posterior columns so canonical index i corresponds to the
    original state state_perm[i]. Shape preserved at (T, n_states)."""
    return posteriors[:, state_perm]


# ─── Walk-forward orchestration ───────────────────────────────────────────────

def run_hmm_rolling(
    X_obs: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    window: int = HMM_DEFAULT_WINDOW,
    refit_every: int = HMM_DEFAULT_REFIT_EVERY,
    state_grid: tuple = HMM_DEFAULT_STATE_GRID,
    covariance_type: str = "full",
    random_state: int = 42,
    n_iter: int = HMM_DEFAULT_N_ITER,
    verbose: bool = False,
) -> pd.DataFrame:
    """Walk-forward HMM fit + forward posteriors on the target's own obs.

    Parameters
    ----------
    X_obs : DataFrame with columns ['ret', 'vol5'] and a tz-naive daily
            index. NaN rows (e.g. from rolling().std() burn-in) are
            dropped internally. The returned DataFrame is reindexed to
            daily_index so early rows are NaN.
    daily_index : trading calendar to reindex against (typically df.index)
    window : rolling training-window length in rows of X_obs.
    refit_every : refit the HMM every N walk-forward steps. Between
            refits, the existing model is reused with predict_proba.
    state_grid : candidate n_states for one-time BIC selection.
    covariance_type : forwarded to GaussianHMM.
    random_state, n_iter : forwarded to GaussianHMM.
    verbose : if True, print the BIC-selected n and the number of refits.

    Returns
    -------
    DataFrame with columns HMM_OUTPUT_COLUMNS on daily_index. The first
    `window - 1` rows of X_obs carry NaN (no complete training window
    available yet). All subsequent rows are populated.
    """
    X_obs = X_obs.copy()
    X_obs.index = _to_date_index(X_obs.index)
    X_obs = X_obs.dropna().sort_index()

    if X_obs.shape[0] <= window:
        warnings.warn(
            f"HMM: insufficient observations ({X_obs.shape[0]}) for window={window}. "
            "Returning empty feature frame."
        )
        return pd.DataFrame(index=daily_index, columns=HMM_OUTPUT_COLUMNS, dtype=float)

    X_mat = X_obs.values
    T, _ = X_mat.shape

    # ── One-time BIC selection on the initial training window ─────────────
    X0 = X_mat[:window]
    n_states = select_n_states_by_bic(
        X0,
        state_grid=state_grid,
        covariance_type=covariance_type,
        random_state=random_state,
        n_iter=n_iter,
    )
    if verbose:
        print(f"HMM n_states selected by BIC: {n_states} "
              f"(grid={state_grid}, initial window rows={window})")

    # Containers over all feasible prediction steps t ∈ [window-1, T-1]
    pred_states = np.full(T, np.nan)
    pred_pbear  = np.full(T, np.nan)
    pred_pbull  = np.full(T, np.nan)

    model = None
    state_perm = None
    refit_count = 0
    carry_warned = False

    for t in range(window - 1, T):
        step_from_start = t - (window - 1)
        need_refit = (step_from_start % refit_every == 0)

        if need_refit:
            X_train = X_mat[t - window + 1 : t + 1]  # rolling 504d up to & including t
            try:
                model, state_perm = fit_hmm_canonical(
                    X_train,
                    n_states=n_states,
                    covariance_type=covariance_type,
                    random_state=random_state,
                    n_iter=n_iter,
                )
                refit_count += 1
            except Exception as e:
                if not carry_warned:
                    warnings.warn(f"HMM: refit failed at t={t}: {e}. Carrying forward last fit.")
                    carry_warned = True
                # fall through using prior model / state_perm

        if model is None:
            # cannot predict without a fit
            continue

        # Posteriors over the current window. Last row is P(state_t | obs_{1:t}).
        try:
            X_win = X_mat[t - window + 1 : t + 1]
            post_raw = model.predict_proba(X_win)           # (window, n_states)
            post = _apply_perm_to_posteriors(post_raw, state_perm)
            p_t = post[-1]                                   # (n_states,)
            pred_states[t] = int(np.argmax(p_t))
            pred_pbear[t]  = float(p_t[0])
            pred_pbull[t]  = float(p_t[-1])
        except Exception as e:
            if not carry_warned:
                warnings.warn(f"HMM: predict_proba failed at t={t}: {e}")
                carry_warned = True

    if verbose:
        print(f"HMM walk-forward: {refit_count} refits over {T - window + 1} steps")

    # ── Pack into DataFrame keyed by X_obs.index ─────────────────────────
    feat = pd.DataFrame(
        {
            "regime_state":     pred_states,
            "regime_prob_bear": pred_pbear,
            "regime_prob_bull": pred_pbull,
        },
        index=X_obs.index,
    )

    # regime_transition: change in state vs previous valid day. Defined only
    # where regime_state is populated. A NaN on either side produces NaN.
    prev = feat["regime_state"].shift(1)
    diff = (feat["regime_state"] != prev).astype(float)
    # Mask where either current or prior state is NaN
    mask_nan = feat["regime_state"].isna() | prev.isna()
    diff[mask_nan] = np.nan
    feat["regime_transition"] = diff

    # Reindex onto the caller's daily trading calendar
    out = feat.reindex(_to_date_index(daily_index))
    out = out[HMM_OUTPUT_COLUMNS]
    return out
