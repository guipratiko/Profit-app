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

from dataclasses import dataclass


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
