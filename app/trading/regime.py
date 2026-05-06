"""Market regime detection.

A core insight from the project's revised thesis: the market alternates
between two regimes that demand different decision policies.

* ``technical``  — quiet news flow, mean volatility, price reverts to
  micro-structure. Technical signal should dominate.
* ``narrative``  — strong qualitative pressure (Copom, fato relevante,
  guidance surprise, fiscal, commodity shock) and/or volatility spike.
  Qualitative signal should dominate or even *override* the technical one.
* ``mixed``     — neither pure regime; blend with cautious weights and prefer
  ``no_operate`` if signals diverge strongly.

This module is pure-function and stateless so it can be called from
``fusion.py``, ``paper.py`` and the API without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


VOL_NORMAL_PERCENTILE_HIGH = 0.80  # > p80 vol → narrative regime contribution
VOL_NORMAL_PERCENTILE_LOW = 0.20   # < p20 vol → strong technical regime
EVENT_MAGNITUDE_NARRATIVE = 0.45   # event_magnitude above this pushes narrative
SENTIMENT_OVERRIDE_THRESHOLD = 0.55  # |sentiment| × severity above which qualitative *overrides*


@dataclass(frozen=True)
class RegimeAssessment:
    regime: str                    # "technical" | "narrative" | "mixed"
    technical_weight: float        # ∈ [0, 1]
    qualitative_weight: float      # ∈ [0, 1]
    override_qualitative: bool     # if True, fusion replaces technical with qualitative direction
    volatility_percentile: float   # 0..1 within the ticker's own history
    event_magnitude: float         # 0..1, max severity * |sentiment|
    divergence: float              # |technical_up - 0.5 - sentiment/2|, 0..1
    notes: list[str]


def estimate_volatility_percentile(
    current_vol: float | None,
    historical_vols: list[float] | None,
) -> float:
    if current_vol is None or not historical_vols:
        return 0.5
    cleaned = sorted(v for v in historical_vols if v is not None)
    if not cleaned:
        return 0.5
    below = sum(1 for v in cleaned if v <= current_vol)
    return float(below) / float(len(cleaned))


def assess_regime(
    technical_probability_up: float,
    technical_confidence: float,
    sentiment_score: float,
    event_count: int,
    event_severity: float,
    event_magnitude: float,
    volatility_percentile: float,
) -> RegimeAssessment:
    """Combine vol regime + event load + signal divergence into a policy."""

    notes: list[str] = []

    narrative_pressure = 0.0
    # Volatility tail contributes to narrative pressure.
    if volatility_percentile >= VOL_NORMAL_PERCENTILE_HIGH:
        narrative_pressure += 0.4
        notes.append("volatility_above_p80")
    elif volatility_percentile <= VOL_NORMAL_PERCENTILE_LOW:
        narrative_pressure -= 0.2
        notes.append("volatility_below_p20")

    # Event load contributes to narrative pressure.
    if event_count > 0 and event_magnitude >= EVENT_MAGNITUDE_NARRATIVE:
        narrative_pressure += min(0.6, 0.3 + event_magnitude * 0.4)
        notes.append("strong_event_magnitude")
    elif event_count > 0:
        narrative_pressure += 0.15
        notes.append("event_present_low_magnitude")

    # Decide regime band
    if narrative_pressure >= 0.55:
        regime = "narrative"
        qualitative_weight = 0.70
        technical_weight = 0.30
    elif narrative_pressure <= 0.10:
        regime = "technical"
        qualitative_weight = 0.10
        technical_weight = 0.90
    else:
        regime = "mixed"
        qualitative_weight = 0.40
        technical_weight = 0.60

    # Override path: very strong, severe event with clear polarity beats whatever
    # the TF model says. This is the "Copom hike + negative tone" case.
    override = bool(
        event_count > 0
        and event_severity >= 0.8
        and abs(sentiment_score) * event_severity >= SENTIMENT_OVERRIDE_THRESHOLD
    )
    if override:
        notes.append("qualitative_override_triggered")
        qualitative_weight = 0.85
        technical_weight = 0.15

    # Divergence: how much do technical and qualitative disagree on direction?
    # technical_probability_up centred at 0.5 → +; sentiment_score centred at 0 → +.
    divergence = float(
        abs((technical_probability_up - 0.5) - (sentiment_score / 2.0))
    )
    if divergence >= 0.35:
        notes.append("strong_signal_divergence")

    return RegimeAssessment(
        regime=regime,
        technical_weight=float(technical_weight),
        qualitative_weight=float(qualitative_weight),
        override_qualitative=override,
        volatility_percentile=float(volatility_percentile),
        event_magnitude=float(event_magnitude),
        divergence=divergence,
        notes=notes,
    )

# ---------------------------------------------------------------------------
# HMM-based market regime overlay (P4)
# ---------------------------------------------------------------------------
# Heuristic narrative_pressure works on per-ticker context (vol percentile +
# event load). We complement it with a market-wide hidden-state model fitted
# on the cross-sectional median return + median 21d realised vol. The state
# with the lowest mean return is labelled ``bear``, the highest ``bull``,
# and the middle one ``chop``. The current state's posterior probability
# then biases the per-ticker assessment via :func:`apply_hmm_overlay`.
# hmmlearn is an optional dependency; if missing we silently skip the overlay.

_HMM_LABELS = ("bear", "chop", "bull")


def _build_market_observation_panel(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    needed = {"date", "ticker", "close"}
    if not needed.issubset(prices.columns):
        return pd.DataFrame()
    df = prices[["date", "ticker", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last"
    ).sort_index()
    returns = pivot.pct_change()
    median_return = returns.median(axis=1)
    realised_vol = returns.rolling(window=21, min_periods=10).std().median(axis=1)
    panel = pd.DataFrame(
        {
            "median_return": median_return,
            "realised_vol": realised_vol,
        }
    ).dropna()
    return panel


def fit_market_hmm(prices: pd.DataFrame, n_states: int = 3) -> dict | None:
    """Fit a Gaussian HMM on cross-sectional market dynamics.

    Returns a dict with the current state's label, posterior probabilities and
    the per-state mean return. Returns ``None`` when ``hmmlearn`` is missing,
    when the panel is too short, or when fitting fails.
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except Exception:
        return None
    panel = _build_market_observation_panel(prices)
    if len(panel) < max(80, n_states * 30):
        return None
    observations = panel.to_numpy(dtype="float64")
    try:
        model = GaussianHMM(
            n_components=int(n_states),
            covariance_type="diag",
            n_iter=200,
            random_state=42,
            tol=1e-3,
        )
        model.fit(observations)
        posteriors = model.predict_proba(observations)
    except Exception:
        return None
    means = model.means_[:, 0]
    order = np.argsort(means)
    if len(order) >= 3:
        labels = {int(order[0]): "bear", int(order[1]): "chop", int(order[-1]): "bull"}
    elif len(order) == 2:
        labels = {int(order[0]): "bear", int(order[1]): "bull"}
    else:
        labels = {int(order[0]): "chop"}
    last_post = posteriors[-1]
    state_probabilities = {
        labels.get(i, f"state_{i}"): float(last_post[i]) for i in range(len(last_post))
    }
    current_state = max(state_probabilities, key=state_probabilities.get)
    return {
        "current_state": current_state,
        "state_probabilities": state_probabilities,
        "state_means": {labels.get(i, f"state_{i}"): float(means[i]) for i in range(len(means))},
        "n_observations": int(len(panel)),
        "method": "gaussian_hmm_v1",
    }


def apply_hmm_overlay(
    assessment: RegimeAssessment,
    hmm_state: dict | None,
) -> RegimeAssessment:
    """Adjust the heuristic assessment with the HMM market state.

    * ``bear`` market with high posterior ? push toward narrative regime,
      down-weight the technical signal.
    * ``bull`` market with high posterior ? reinforce technical regime.
    * ``chop`` keeps the heuristic verdict.

    The assessment's :attr:`override_qualitative` flag is never lowered: if
    the heuristic already triggered an override, the HMM cannot retract it.
    """
    if not hmm_state:
        return assessment
    label = hmm_state.get("current_state")
    posterior = float(hmm_state.get("state_probabilities", {}).get(label, 0.0))
    if posterior < 0.55:
        return assessment

    notes = list(assessment.notes) + [f"hmm_state:{label}:{posterior:.2f}"]
    technical_weight = assessment.technical_weight
    qualitative_weight = assessment.qualitative_weight
    regime = assessment.regime
    if label == "bear":
        technical_weight = max(0.20, assessment.technical_weight - 0.20)
        qualitative_weight = 1.0 - technical_weight
        if regime == "technical":
            regime = "mixed"
    elif label == "bull":
        technical_weight = min(0.95, assessment.technical_weight + 0.10)
        qualitative_weight = 1.0 - technical_weight
        if regime == "narrative" and not assessment.override_qualitative:
            regime = "mixed"
    return replace(
        assessment,
        regime=regime,
        technical_weight=float(technical_weight),
        qualitative_weight=float(qualitative_weight),
        notes=notes,
    )