from __future__ import annotations

import json
import os
from datetime import datetime
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from app.config import INITIAL_ASSETS
from app.data.company_branding import LOGO_CACHE_TTL, asset_branding_record, get_company_logo
from app.data.database import (
    get_fusion_predictions,
    get_latest_model_run_id,
    get_news_events,
    get_paper_trading_signals,
    get_paper_positions,
    get_price_counts,
    get_qualitative_features,
    get_real_positions,
    get_risk_alerts,
    initialize_database,
    read_model_predictions,
    read_ohlcv_prices,
    save_real_positions,
    delete_real_position,
    update_real_position,
    update_real_position_prices,
)
from app.models.fusion import run_fusion_predictions
from app.models.pytorch_sentiment import generate_qualitative_features
from app.pipelines.alpha_metrics import build_alpha_metrics
from app.pipelines.paper_validation import build_paper_validation_report
from app.pipelines.refresh import detect_staleness, run_refresh_pipeline
from app.trading.paper import generate_paper_trading_signals
from app.trading.risk_advisor import (
    audit_paper_portfolio,
    audit_paper_portfolio_with_conselheiro,
)


app = FastAPI(
    title="Profit App Alpha API",
    version="0.9.0",
    description="Experimental B3 alpha API for paper trading only. Not financial advice.",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("PROFIT_APP_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    clean = frame.where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def latest_row(frame: pd.DataFrame, ticker: str, date_column: str = "created_at") -> dict | None:
    if frame.empty:
        return None
    filtered = frame[frame["ticker"] == ticker].copy()
    if filtered.empty:
        return None
    if date_column in filtered.columns:
        filtered = filtered.sort_values(date_column)
    return dataframe_records(filtered.tail(1))[0]


def build_prediction_payload(
    ticker: str,
    run_id: str,
    technical_frame: pd.DataFrame,
    fusion_frame: pd.DataFrame,
    paper_frame: pd.DataFrame,
) -> dict:
    technical_latest = latest_row(technical_frame, ticker, date_column="date")
    fusion_latest = latest_row(fusion_frame, ticker)
    paper_latest = latest_row(paper_frame, ticker)
    if technical_latest is None and fusion_latest is None and paper_latest is None:
        raise HTTPException(status_code=404, detail=f"No prediction artifacts found for {ticker}")
    return {
        "ticker": ticker,
        "model_run_id": run_id,
        "technical_prediction": technical_latest,
        "fusion_prediction": fusion_latest,
        "paper_signal": paper_latest,
        "risk_notice": "Experimental paper-trading thesis only. Not financial advice.",
    }


class RealPositionCreateRequest(BaseModel):
    ticker: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    entry_at: datetime | None = None
    notes: str | None = None


def _real_portfolio_notice() -> str:
    return "Real positions are user-managed. This is not financial advice and no orders are executed."


def _coerce_limit(limit: int | object, default: int) -> int:
    if isinstance(limit, int):
        return limit
    return int(getattr(limit, "default", default))


DASHBOARD_HTML = """
<!doctype html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Profit App Alpha</title>
    <style>
        :root { color-scheme: light; --ink:#17202a; --muted:#5f6b7a; --line:#d9dee7; --bg:#f6f7f9; --panel:#ffffff; --accent:#0969da; --bad:#b42318; --ok:#067647; }
        * { box-sizing: border-box; }
        body { margin:0; font-family: Arial, sans-serif; background:var(--bg); color:var(--ink); }
        header { background:var(--panel); border-bottom:1px solid var(--line); padding:16px 24px; display:flex; align-items:center; justify-content:space-between; gap:16px; }
        h1 { font-size:20px; margin:0; }
        main { padding:20px 24px; display:grid; grid-template-columns: 280px 1fr; gap:18px; }
        section, aside { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
        button, select { min-height:36px; border:1px solid var(--line); border-radius:6px; background:#fff; padding:0 10px; color:var(--ink); }
        button { cursor:pointer; background:var(--accent); color:#fff; border-color:var(--accent); }
        table { width:100%; border-collapse:collapse; font-size:13px; }
        th, td { border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }
        th { color:var(--muted); font-weight:600; }
        .grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:12px; }
        .metric { border:1px solid var(--line); border-radius:8px; padding:12px; }
        .metric strong { display:block; font-size:18px; margin-top:4px; }
        .muted { color:var(--muted); }
        .risk { color:var(--bad); font-weight:600; }
        .ok { color:var(--ok); font-weight:600; }
        pre { white-space:pre-wrap; word-break:break-word; background:#f1f3f6; padding:12px; border-radius:6px; max-height:320px; overflow:auto; }
        @media (max-width: 900px) { main { grid-template-columns:1fr; } .grid { grid-template-columns:1fr; } }
    </style>
</head>
<body>
    <header>
        <div><h1>Profit App Alpha</h1><div class="muted">Paper trading experimental. Sem ordens reais.</div></div>
        <button onclick="auditRisk()">Auditar Risco</button>
    </header>
    <main>
        <aside>
            <label for="assetSelect">Ativo</label><br>
            <select id="assetSelect"></select>
            <p class="muted">As previsoes sao teses experimentais e nao recomendacao financeira.</p>
            <div id="assetList"></div>
        </aside>
        <section>
            <div class="grid" id="metrics"></div>
            <h2>Previsao e Tese</h2>
            <div id="prediction"></div>
            <h2>Historico recente</h2>
            <div id="prices"></div>
            <h2>Carteira simulada</h2>
            <div id="portfolio"></div>
            <h2>Alertas de risco</h2>
            <div id="alerts"></div>
            <h2>Explicacao</h2>
            <pre id="explanation">Selecione um ativo.</pre>
        </section>
    </main>
    <script>
        const fmt = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 4 });
        async function getJson(url, options) { const res = await fetch(url, options); if (!res.ok) throw new Error(await res.text()); return res.json(); }
        function table(rows, cols) {
            if (!rows || rows.length === 0) return '<p class="muted">Sem dados.</p>';
            return '<table><thead><tr>' + cols.map(c => `<th>${c[0]}</th>`).join('') + '</tr></thead><tbody>' +
                rows.map(r => '<tr>' + cols.map(c => `<td>${r[c[1]] ?? ''}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
        }
        async function loadAssets() {
            const data = await getJson('/assets');
            const select = document.getElementById('assetSelect');
            select.innerHTML = data.assets.map(a => `<option value="${a.ticker}">${a.ticker}</option>`).join('');
            document.getElementById('assetList').innerHTML = table(data.assets, [['Ticker','ticker'], ['Nome','name']]);
            select.onchange = () => loadTicker(select.value);
            await loadTicker(select.value);
        }
        async function loadTicker(ticker) {
            const [prediction, prices, explanation, metrics, portfolio, alerts] = await Promise.all([
                getJson(`/predictions/${ticker}`), getJson(`/prices/${ticker}?limit=8`), getJson(`/predictions/${ticker}/explanation`),
                getJson('/paper/metrics'), getJson('/portfolio/positions'), getJson('/portfolio/alerts')
            ]);
            document.getElementById('metrics').innerHTML = [
                ['Sinais', metrics.signals], ['Simulados', metrics.simulate_long], ['Bloqueados', metrics.no_operate]
            ].map(m => `<div class="metric"><span class="muted">${m[0]}</span><strong>${m[1]}</strong></div>`).join('');
            const paper = prediction.paper_signal || {};
            const fusion = prediction.fusion_prediction || {};
            document.getElementById('prediction').innerHTML = table([{
                ticker, direction: fusion.fused_direction, score: fmt.format(fusion.fused_score || 0), decision: paper.decision,
                reason: paper.block_reason, entry: paper.suggested_entry, stop: paper.stop_loss, target: paper.target_price, shares: paper.max_shares
            }], [['Ativo','ticker'], ['Direcao','direction'], ['Score','score'], ['Decisao','decision'], ['Motivo','reason'], ['Entrada','entry'], ['Stop','stop'], ['Alvo','target'], ['Qtd max','shares']]);
            document.getElementById('prices').innerHTML = table(prices.rows, [['Data','date'], ['Fechamento','close'], ['Volume','volume']]);
            document.getElementById('portfolio').innerHTML = table(portfolio.positions, [['Ativo','ticker'], ['Status','status'], ['Entrada','entry_price'], ['Atual','current_price'], ['Retorno','unrealized_return']]);
            document.getElementById('alerts').innerHTML = table(alerts.alerts, [['Ativo','ticker'], ['Acao','action'], ['Severidade','severity'], ['Motivo','reason'], ['Retorno','unrealized_return']]);
            document.getElementById('explanation').textContent = JSON.stringify(explanation.explanation, null, 2);
        }
        async function auditRisk() { await getJson('/portfolio/audit', {method:'POST'}); await loadTicker(document.getElementById('assetSelect').value); }
        loadAssets().catch(err => { document.getElementById('explanation').textContent = err.message; });
    </script>
</body>
</html>
"""


@app.on_event("startup")
def startup() -> None:
    initialize_database()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": "paper_trading_only",
        "risk_notice": "Experimental system. Not financial advice and not a real order router.",
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/assets")
def list_assets(request: Request) -> dict:
    return {
        "assets": [
            asset_branding_record(ticker, str(request.url_for("asset_logo", ticker=ticker)))
            for ticker in INITIAL_ASSETS
        ]
    }


@app.get("/assets/{ticker}/logo", name="asset_logo", include_in_schema=False)
def asset_logo(ticker: str) -> Response:
    if ticker not in INITIAL_ASSETS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} is not monitored by the current market data universe")
    logo = get_company_logo(ticker)
    return Response(
        content=logo.content,
        media_type=logo.media_type,
        headers={
            "Cache-Control": f"public, max-age={int(LOGO_CACHE_TTL.total_seconds())}",
            "X-Logo-Source": logo.source_url,
        },
    )


@app.get("/prices/{ticker}")
def price_history(ticker: str, limit: int = Query(default=120, ge=1, le=2000)) -> dict:
    prices = read_ohlcv_prices()
    filtered = prices[prices["ticker"] == ticker].sort_values("date").tail(limit)
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No prices found for {ticker}")
    return {"ticker": ticker, "rows": dataframe_records(filtered)}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <rect width="64" height="64" rx="16" fill="#020617"/>
      <path d="M16 42 29 29l7 7 13-18" fill="none" stroke="#38bdf8" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="49" cy="18" r="5" fill="#a78bfa"/>
    </svg>
    """.strip()
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/predictions/{ticker}")
def prediction_for_ticker(ticker: str) -> dict:
    run_id = get_latest_model_run_id()
    technical = read_model_predictions(run_id, split="test")
    return build_prediction_payload(
        ticker=ticker,
        run_id=run_id,
        technical_frame=technical,
        fusion_frame=get_fusion_predictions(),
        paper_frame=get_paper_trading_signals(),
    )


@app.get("/predictions")
def prediction_snapshot() -> dict:
    run_id = get_latest_model_run_id()
    technical = read_model_predictions(run_id, split="test")
    fusion = get_fusion_predictions()
    paper = get_paper_trading_signals()

    predictions: list[dict] = []
    for ticker in INITIAL_ASSETS:
        try:
            predictions.append(
                build_prediction_payload(
                    ticker=ticker,
                    run_id=run_id,
                    technical_frame=technical,
                    fusion_frame=fusion,
                    paper_frame=paper,
                )
            )
        except HTTPException:
            predictions.append({"ticker": ticker})

    return {"predictions": predictions}


@app.get("/predictions/{ticker}/explanation")
def prediction_explanation(ticker: str) -> dict:
    fusion_latest = latest_row(get_fusion_predictions(), ticker)
    if fusion_latest is None:
        raise HTTPException(status_code=404, detail=f"No fusion explanation found for {ticker}")
    explanation = json.loads(str(fusion_latest["explanation_json"]))
    return {"ticker": ticker, "explanation": explanation}


@app.post("/updates/retrain")
def update_and_recalculate(
    analyze_sentiment: bool = True,
    run_fusion: bool = True,
) -> dict:
    initialize_database()
    result: dict = {
        "updated_prices": False,
        "retrained_tensorflow": False,
        "risk_notice": "Endpoint recalculates local alpha artifacts only; it does not execute orders.",
    }
    if analyze_sentiment:
        result["sentiment"] = generate_qualitative_features()
    if run_fusion:
        result["fusion"] = run_fusion_predictions()
    return result


@app.post("/paper/signals")
def create_paper_signals() -> dict:
    return generate_paper_trading_signals()


@app.get("/paper/signals")
def paper_signals(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    signals = get_paper_trading_signals().head(limit)
    return {"signals": dataframe_records(signals)}


@app.get("/paper/blocked")
def blocked_signals(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    signals = get_paper_trading_signals()
    blocked = signals[signals["decision"] == "no_operate"].head(limit) if not signals.empty else signals
    return {"blocked_signals": dataframe_records(blocked)}


@app.get("/paper/metrics")
def paper_metrics() -> dict:
    signals = get_paper_trading_signals()
    if signals.empty:
        return {"signals": 0, "simulate_long": 0, "no_operate": 0, "blocked_by_reason": {}}
    blocked_reasons: dict[str, int] = {}
    for reason_value in signals["block_reason"].dropna().tolist():
        for reason in str(reason_value).split(","):
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
    return {
        "signals": int(len(signals)),
        "simulate_long": int((signals["decision"] == "simulate_long").sum()),
        "no_operate": int((signals["decision"] == "no_operate").sum()),
        "blocked_by_reason": blocked_reasons,
    }


@app.post("/portfolio/audit")
def audit_portfolio() -> dict:
    return audit_paper_portfolio()


@app.get("/portfolio/positions")
def portfolio_positions(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"positions": dataframe_records(get_paper_positions().head(limit))}


@app.get("/portfolio/alerts")
def portfolio_alerts(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"alerts": dataframe_records(get_risk_alerts().head(limit))}


@app.post("/portfolio/real/register")
def register_real_position(payload: RealPositionCreateRequest) -> dict:
    initialize_database()
    ticker = payload.ticker.strip().upper()
    if ticker not in INITIAL_ASSETS:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} is not monitored by the current market data universe")

    position_id = f"real_{uuid4().hex[:16]}"
    entry_at = (payload.entry_at or datetime.utcnow()).isoformat(timespec="seconds")
    notes = payload.notes.strip() if payload.notes else None
    save_real_positions(
        pd.DataFrame(
            [
                {
                    "position_id": position_id,
                    "ticker": ticker,
                    "quantity": int(payload.quantity),
                    "entry_price": float(payload.entry_price),
                    "entry_at": entry_at,
                    "cost_basis": float(payload.quantity) * float(payload.entry_price),
                    "current_price": float(payload.entry_price),
                    "last_updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                    "notes": notes,
                }
            ]
        )
    )
    return {
        "position_id": position_id,
        "status": "created",
        "ticker": ticker,
        "risk_notice": _real_portfolio_notice(),
    }


@app.get("/portfolio/real/positions")
def real_portfolio_positions(limit: int = Query(default=200, ge=1, le=1000)) -> dict:
    initialize_database()
    positions = get_real_positions().head(_coerce_limit(limit, 200))
    return {
        "positions": dataframe_records(positions),
        "risk_notice": _real_portfolio_notice(),
    }


@app.put("/portfolio/real/{position_id}")
def edit_real_position(position_id: str, payload: RealPositionCreateRequest) -> dict:
    initialize_database()
    ticker = payload.ticker.strip().upper()
    if ticker not in INITIAL_ASSETS:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} is not monitored by the current market data universe")

    entry_at = (payload.entry_at or datetime.utcnow()).isoformat(timespec="seconds")
    notes = payload.notes.strip() if payload.notes else None
    updated = update_real_position(
        position_id=position_id,
        ticker=ticker,
        quantity=int(payload.quantity),
        entry_price=float(payload.entry_price),
        entry_at=entry_at,
        notes=notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Real position {position_id} was not found")

    return {
        "position_id": position_id,
        "status": "updated",
        "ticker": ticker,
        "risk_notice": _real_portfolio_notice(),
    }


@app.post("/portfolio/mark-to-market")
def mark_to_market_portfolio() -> dict:
    initialize_database()
    updated_count = update_real_position_prices()
    return {
        "updated_count": int(updated_count),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "risk_notice": _real_portfolio_notice(),
    }


@app.delete("/portfolio/real/{position_id}")
def remove_real_position(position_id: str) -> dict:
    initialize_database()
    deleted = delete_real_position(position_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Real position {position_id} not found")
    return {
        "position_id": position_id,
        "status": "deleted",
        "risk_notice": _real_portfolio_notice(),
    }


@app.get("/qualitative/features")
def qualitative_features(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"features": dataframe_records(get_qualitative_features().head(limit))}


@app.get("/news/events")
def news_events(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"events": dataframe_records(get_news_events().head(limit))}


@app.get("/data/status")
def data_status() -> dict:
    return {
        "prices": dataframe_records(get_price_counts()),
        "qualitative_features": len(get_qualitative_features()),
        "fusion_predictions": len(get_fusion_predictions()),
        "paper_signals": len(get_paper_trading_signals()),
    }


@app.get("/refresh/status")
def refresh_status() -> dict:
    return detect_staleness()


@app.post("/refresh/run")
def refresh_run(
    max_staleness_days: int = Query(default=1, ge=0, le=90),
    refit_window_days: int = Query(default=180, ge=30, le=720),
    skip_price_update: bool = Query(default=False),
) -> dict:
    return run_refresh_pipeline(
        max_staleness_days=max_staleness_days,
        refit_window_days=refit_window_days,
        skip_price_update=skip_price_update,
    )


@app.post("/portfolio/audit-conselheiro")
def audit_portfolio_conselheiro() -> dict:
    return audit_paper_portfolio_with_conselheiro()


@app.get("/alpha/metrics")
def alpha_metrics() -> dict:
    return build_alpha_metrics()


@app.get("/validation/paper-gate")
def paper_validation_gate() -> dict:
    return build_paper_validation_report()


@app.get("/api/paper-validation", include_in_schema=False)
def paper_validation_compat() -> dict:
    return paper_validation_gate()
