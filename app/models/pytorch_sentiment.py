"""Qualitative layer (PyTorch).

Phase 1 rewrite. The market is mostly narrative; this module is the system's
"perception" layer. It does three things in one pass per event:

1. Event tagging — classifies each headline/body into one or more market-moving
   tag families (Copom, fato_relevante, guidance, fiscal_policy,
   commodity_shock, ceo_speech, regulatory, geopolitical, earnings).
2. Magnitude scoring — beyond polarity, estimates how *severe* the event is
   (a CEO casual quote ≠ a Copom hike). Used downstream by fusion to override
   the technical signal when contextual evidence is strong.
3. Contextual encoding — when a transformer-based encoder is available
   (FinBERT-PT-BR / BERTimbau through `transformers` + `torch`), uses it to
   produce a real semantic embedding. Otherwise falls back to a deterministic
   lexical encoder so the pipeline never breaks on Python 3.14 / no-GPU envs.

Public API is intentionally preserved (analyze_text, build_qualitative_features,
generate_qualitative_features, get_qualitative_summary, evaluate_manual_sample)
so paper.py / fusion.py / E2E scripts keep working.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from app.data.database import (
    get_news_events,
    get_qualitative_features,
    initialize_database,
    save_qualitative_features,
)

try:
    import torch  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
except ImportError:  # Python 3.14 in the lightweight local venv may not have torch wheels yet.
    torch = None  # type: ignore[assignment]

# Optional contextual encoder. Only loaded when explicitly enabled and torch is
# available. We keep this opt-in via env var to avoid downloading 400MB weights
# during normal CLI usage / CI.
_TRANSFORMER_PIPELINE = None
_TRANSFORMER_NAME = os.environ.get(
    "PROFIT_APP_SENTIMENT_MODEL",
    "neuralmind/bert-base-portuguese-cased",  # BERTimbau base; FinBERT-PT-BR also works
)
_USE_TRANSFORMER = os.environ.get("PROFIT_APP_USE_TRANSFORMER_SENTIMENT", "0") == "1"


SENTIMENT_MODEL_NAME = "pytorch_contextual_sentiment_v2"
EMBEDDING_DIMENSION = 16  # kept for backwards-compat with stored rows
TRANSFORMER_EMBEDDING_DIMENSION = 768
TOKEN_RE = re.compile(r"[a-zA-Z_]+")

POSITIVE_TERMS = {
    "alta", "aumenta", "aumento", "cresce", "crescimento", "demanda",
    "dividendos", "forte", "ganho", "lucro", "melhora", "positivo",
    "recorde", "recuperacao", "supera", "valorizacao", "expansao",
    "aprovado", "acelera", "robusto", "otimista",
}
NEGATIVE_TERMS = {
    "baixa", "cai", "queda", "risco", "fraco", "perda", "prejuizo",
    "negativo", "reduz", "reducao", "volatilidade", "investigacao",
    "incerteza", "pressiona", "desacelera", "rebaixa", "recessao",
    "downgrade", "afundamento", "colapso", "demite",
}
NEUTRAL_CONTEXT_TERMS = {
    "BACEN", "COPOM", "MINERIO_DE_FERRO", "PETROLEO",
    "juros", "commodity", "commodities", "cenario",
}


# -------------------------------------------------------------------------
# Event tagging — the 5 narrative families that historically move B3 the most
# -------------------------------------------------------------------------
EVENT_TAG_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "copom": (
        re.compile(r"\bcopom\b", re.IGNORECASE),
        re.compile(r"\bbacen\b", re.IGNORECASE),
        re.compile(r"\bselic\b", re.IGNORECASE),
        re.compile(r"\btaxa\s+basica\b", re.IGNORECASE),
    ),
    "fato_relevante": (
        re.compile(r"fato\s+relevante", re.IGNORECASE),
        re.compile(r"comunicado\s+ao\s+mercado", re.IGNORECASE),
        re.compile(r"\bcvm\b", re.IGNORECASE),
    ),
    "guidance": (
        re.compile(r"\bguidance\b", re.IGNORECASE),
        re.compile(r"proje[cç][aã]o\s+(de\s+)?(lucro|receita|ebitda)", re.IGNORECASE),
        re.compile(r"revis[aã]o\s+de\s+meta", re.IGNORECASE),
        re.compile(r"\bbalanco\b", re.IGNORECASE),
        re.compile(r"resultado\s+trimestral", re.IGNORECASE),
    ),
    "fiscal_policy": (
        re.compile(r"reforma\s+(tributaria|fiscal)", re.IGNORECASE),
        re.compile(r"arcabouco\s+fiscal", re.IGNORECASE),
        re.compile(r"\bteto\s+de\s+gastos\b", re.IGNORECASE),
        re.compile(r"medida\s+provisoria", re.IGNORECASE),
        re.compile(r"\bpec\b", re.IGNORECASE),
        re.compile(r"\bimposto\b", re.IGNORECASE),
    ),
    "commodity_shock": (
        re.compile(r"PETROLEO|brent|wti", re.IGNORECASE),
        re.compile(r"MINERIO_DE_FERRO|minerio", re.IGNORECASE),
        re.compile(r"opep", re.IGNORECASE),
        re.compile(r"shock|choque\s+(de\s+)?(oferta|demanda)", re.IGNORECASE),
    ),
    "ceo_speech": (
        re.compile(r"\bceo\b", re.IGNORECASE),
        re.compile(r"presidente\s+da\s+(companhia|empresa)", re.IGNORECASE),
        re.compile(r"declar(a|ou|acao)", re.IGNORECASE),
        re.compile(r"pronunciamento", re.IGNORECASE),
    ),
    "regulatory": (
        re.compile(r"\bcade\b", re.IGNORECASE),
        re.compile(r"\banatel\b", re.IGNORECASE),
        re.compile(r"\baneel\b", re.IGNORECASE),
        re.compile(r"\banp\b", re.IGNORECASE),
        re.compile(r"\banvisa\b", re.IGNORECASE),
        re.compile(r"investigacao\s+regulatoria", re.IGNORECASE),
    ),
    "geopolitical": (
        re.compile(r"\bguerra\b", re.IGNORECASE),
        re.compile(r"\bsancoes?\b", re.IGNORECASE),
        re.compile(r"tarifa\s+(comercial|de\s+importacao)", re.IGNORECASE),
        re.compile(r"\bopep\b", re.IGNORECASE),
    ),
    "earnings": (
        re.compile(r"\bbalanco\b", re.IGNORECASE),
        re.compile(r"lucro\s+liquido", re.IGNORECASE),
        re.compile(r"ebitda", re.IGNORECASE),
        re.compile(r"trimestre", re.IGNORECASE),
    ),
}

# How much each tag family is allowed to weigh in fusion overrides.
# Calibrated to historical literature (Copom and fato_relevante are the
# classic "narrative regime" triggers on B3). These are *priors*, not truths.
EVENT_TAG_SEVERITY: dict[str, float] = {
    "copom": 1.0,
    "fato_relevante": 0.95,
    "fiscal_policy": 0.9,
    "geopolitical": 0.85,
    "commodity_shock": 0.8,
    "guidance": 0.75,
    "regulatory": 0.7,
    "ceo_speech": 0.55,
    "earnings": 0.7,
}


@dataclass(frozen=True)
class TextSentiment:
    sentiment_score: float
    sentiment_label: str
    positive_score: float
    negative_score: float
    neutral_score: float
    embedding: list[float]
    tokens: list[str]
    event_tags: list[str] = field(default_factory=list)
    event_severity: float = 0.0
    event_magnitude: float = 0.0  # |polarity| * severity ∈ [0, 1]
    encoder: str = "lexical_fallback"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def label_from_score(score: float) -> str:
    if score >= 0.20:
        return "positive"
    if score <= -0.20:
        return "negative"
    return "neutral"


def detect_event_tags(text: str) -> tuple[list[str], float]:
    """Return (matched_tag_list, severity_max) for a piece of text.

    Severity is the max of matched tag severities — one strong tag dominates,
    we don't average so a weak tag doesn't dilute a Copom hit.
    """
    if not text:
        return [], 0.0
    matched: list[str] = []
    severity_max = 0.0
    for tag, patterns in EVENT_TAG_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            matched.append(tag)
            severity_max = max(severity_max, EVENT_TAG_SEVERITY.get(tag, 0.5))
    return matched, severity_max


def _load_transformer_pipeline():
    """Lazy-load FinBERT-PT-BR / BERTimbau if explicitly enabled and torch present."""
    global _TRANSFORMER_PIPELINE
    if _TRANSFORMER_PIPELINE is not None:
        return _TRANSFORMER_PIPELINE
    if not _USE_TRANSFORMER or torch is None:
        return None
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(_TRANSFORMER_NAME)
        model = AutoModel.from_pretrained(_TRANSFORMER_NAME)
        model.eval()
        _TRANSFORMER_PIPELINE = (tokenizer, model)
        return _TRANSFORMER_PIPELINE
    except Exception:
        return None


def _transformer_embed(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float] | None:
    """Mean-pooled [CLS]-aware transformer embedding, projected to `dimension`.

    Projection is a deterministic hash-based linear map (no learned weights),
    so we don't add a fine-tuning dependency just to obtain a stable signature.
    """
    pipeline = _load_transformer_pipeline()
    if pipeline is None:
        return None
    tokenizer, model = pipeline
    try:
        with torch.no_grad():
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            out = model(**tokens).last_hidden_state.mean(dim=1).squeeze(0)
            full = out.cpu().tolist()
    except Exception:
        return None
    # Deterministic projection 768 → `dimension` so embedding column stays the same width.
    projected = [0.0] * dimension
    for index, value in enumerate(full):
        projected[index % dimension] += float(value)
    norm = math.sqrt(sum(v * v for v in projected))
    if norm > 0:
        projected = [v / norm for v in projected]
    return projected


def token_embedding(tokens: list[str], dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    if not tokens:
        return [0.0] * dimension
    vectors: list[list[float]] = []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector = [0.0] * dimension
        for index in range(dimension):
            raw_value = digest[index] / 255.0
            sign = -1.0 if digest[index + dimension] % 2 else 1.0
            vector[index] = sign * raw_value
        vectors.append(vector)

    if torch is not None:
        tensor = torch.tensor(vectors, dtype=torch.float32)
        mean_vector = tensor.mean(dim=0)
        norm = torch.linalg.vector_norm(mean_vector).item()
        if norm > 0:
            mean_vector = mean_vector / norm
        return [float(value) for value in mean_vector.tolist()]

    embedding = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]
    norm = math.sqrt(sum(value * value for value in embedding))
    if norm > 0:
        embedding = [value / norm for value in embedding]
    return [float(value) for value in embedding]


def analyze_text(text: str) -> TextSentiment:
    tokens = tokenize(text)
    positive_hits = sum(1 for token in tokens if token in POSITIVE_TERMS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_TERMS)
    neutral_hits = sum(1 for token in tokens if token in {term.lower() for term in NEUTRAL_CONTEXT_TERMS})
    directional_hits = positive_hits + negative_hits
    sentiment_score = (positive_hits - negative_hits) / max(directional_hits, 1)
    positive_score = positive_hits / max(directional_hits + neutral_hits, 1)
    negative_score = negative_hits / max(directional_hits + neutral_hits, 1)
    neutral_score = 1.0 - min(1.0, positive_score + negative_score)

    event_tags, severity = detect_event_tags(text)
    event_magnitude = float(min(1.0, abs(sentiment_score) * (severity if severity > 0 else 0.0)))

    transformer_embedding = _transformer_embed(text)
    if transformer_embedding is not None:
        embedding = transformer_embedding
        encoder = f"transformer:{_TRANSFORMER_NAME}"
    else:
        embedding = token_embedding(tokens)
        encoder = "lexical_fallback"

    return TextSentiment(
        sentiment_score=float(sentiment_score),
        sentiment_label=label_from_score(sentiment_score),
        positive_score=float(positive_score),
        negative_score=float(negative_score),
        neutral_score=float(neutral_score),
        embedding=embedding,
        tokens=tokens,
        event_tags=event_tags,
        event_severity=float(severity),
        event_magnitude=event_magnitude,
        encoder=encoder,
    )


def aggregate_event_group(group: pd.DataFrame) -> dict:
    texts = group["normalized_text"].fillna("").tolist()
    analyses = [analyze_text(text) for text in texts]
    event_count = len(analyses)
    sentiment_score = sum(item.sentiment_score for item in analyses) / max(event_count, 1)
    positive_score = sum(item.positive_score for item in analyses) / max(event_count, 1)
    negative_score = sum(item.negative_score for item in analyses) / max(event_count, 1)
    neutral_score = sum(item.neutral_score for item in analyses) / max(event_count, 1)
    embedding = [
        sum(item.embedding[index] for item in analyses) / max(event_count, 1)
        for index in range(EMBEDDING_DIMENSION)
    ]
    # Event tag aggregation: union of tags; severity = max; magnitude = max.
    aggregated_tags: list[str] = []
    for item in analyses:
        for tag in item.event_tags:
            if tag not in aggregated_tags:
                aggregated_tags.append(tag)
    severity_max = max((item.event_severity for item in analyses), default=0.0)
    magnitude_max = max((item.event_magnitude for item in analyses), default=0.0)
    encoders = sorted({item.encoder for item in analyses})

    ticker = str(group.iloc[0]["ticker"])
    aligned_date = str(group.iloc[0]["aligned_trading_date"])
    feature_payload = f"{SENTIMENT_MODEL_NAME}|{ticker}|{aligned_date}"
    feature_id = "qual_" + hashlib.sha256(feature_payload.encode("utf-8")).hexdigest()[:16]
    return {
        "feature_id": feature_id,
        "ticker": ticker,
        "aligned_trading_date": aligned_date,
        "event_count": int(event_count),
        "sentiment_score": float(sentiment_score),
        "sentiment_label": label_from_score(sentiment_score),
        "positive_score": float(positive_score),
        "negative_score": float(negative_score),
        "neutral_score": float(neutral_score),
        "embedding_json": json.dumps(embedding, ensure_ascii=True),
        "source_event_ids_json": json.dumps(group["event_id"].tolist(), ensure_ascii=True),
        "model_name": SENTIMENT_MODEL_NAME,
        "metadata_json": json.dumps(
            {
                "torch_available": torch is not None,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "method": "event_tagged_with_optional_transformer_encoder",
                "event_tags": aggregated_tags,
                "event_severity_max": float(severity_max),
                "event_magnitude_max": float(magnitude_max),
                "encoders_used": encoders,
                "transformer_enabled": bool(_USE_TRANSFORMER),
                "transformer_model": _TRANSFORMER_NAME if _USE_TRANSFORMER else None,
            },
            ensure_ascii=True,
        ),
    }


def build_qualitative_features(events: pd.DataFrame | None = None) -> pd.DataFrame:
    if events is None:
        events = get_news_events()
    if events.empty:
        return pd.DataFrame()

    records = [
        aggregate_event_group(group)
        for _key, group in events.groupby(["ticker", "aligned_trading_date"], sort=False)
    ]
    return pd.DataFrame(records)


def generate_qualitative_features() -> dict:
    initialize_database()
    events = get_news_events()
    features = build_qualitative_features(events)
    inserted = save_qualitative_features(features)
    labels = {} if features.empty else features["sentiment_label"].value_counts().to_dict()
    return {
        "events": int(len(events)),
        "generated": int(len(features)),
        "inserted": int(inserted),
        "labels": labels,
        "torch_available": torch is not None,
        "model_name": SENTIMENT_MODEL_NAME,
    }


def get_qualitative_summary() -> pd.DataFrame:
    features = get_qualitative_features()
    if features.empty:
        return pd.DataFrame()
    return features[
        [
            "ticker",
            "aligned_trading_date",
            "event_count",
            "sentiment_label",
            "sentiment_score",
            "positive_score",
            "negative_score",
            "neutral_score",
            "model_name",
        ]
    ]


def evaluate_manual_sample() -> list[dict]:
    examples = [
        ("lucro forte e dividendos positivos", "positive"),
        ("queda aumenta risco e pressiona resultado", "negative"),
        ("COPOM avalia cenario de juros", "neutral"),
    ]
    results: list[dict] = []
    for text, expected in examples:
        analysis = analyze_text(text)
        results.append(
            {
                "text": text,
                "expected": expected,
                "actual": analysis.sentiment_label,
                "passed": analysis.sentiment_label == expected,
                "sentiment_score": analysis.sentiment_score,
            }
        )
    return results
