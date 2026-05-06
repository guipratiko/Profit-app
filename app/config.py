import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


STORAGE_DIR = _env_path("PROFIT_APP_STORAGE_DIR", PROJECT_ROOT / "storage")

# Legacy SQLite path sentinel. The runtime backend is PostgreSQL (configured
# via the ``DATABASE_URL`` env var consumed by ``app.data.engine``). This
# placeholder is kept only so the many DAO functions in ``app.data.database``
# that historically defaulted ``database_path: Path = DATABASE_PATH`` can keep
# their signatures unchanged; the actual value is ignored by ``LegacyConnection``.
DATABASE_PATH: Path = STORAGE_DIR / ".unused_legacy_sqlite_sentinel"


def resolve_artifact_dir(artifact_path: str | Path) -> Path:
    candidate = Path(str(artifact_path))
    if candidate.exists():
        return candidate

    run_id = Path(str(artifact_path).replace("\\", "/")).name
    if not run_id:
        return candidate

    return STORAGE_DIR / "models" / run_id

INITIAL_ASSETS = {
    # Brazil — banking & financial
    "PETR4.SA": "Petrobras PN",
    "VALE3.SA": "Vale ON",
    "ITUB4.SA": "Itau Unibanco PN",
    "BBDC4.SA": "Bradesco PN",
    "BBAS3.SA": "Banco do Brasil ON",
    "SANB11.SA": "Santander Brasil Unit",
    "B3SA3.SA": "B3 ON",
    "BPAC11.SA": "BTG Pactual Unit",
    # Brazil — consumer & retail
    "ABEV3.SA": "Ambev ON",
    "MGLU3.SA": "Magazine Luiza ON",
    "LREN3.SA": "Lojas Renner ON",
    "RENT3.SA": "Localiza ON",
    "RAIL3.SA": "Rumo ON",
    # Brazil — industry & utilities
    "WEGE3.SA": "WEG ON",
    "EQTL3.SA": "Equatorial ON",
    "SBSP3.SA": "Sabesp ON",
    # Brazil — health, paper, steel
    "RDOR3.SA": "Rede D'Or ON",
    "HAPV3.SA": "Hapvida ON",
    "SUZB3.SA": "Suzano ON",
    "KLBN11.SA": "Klabin Unit",
    "CSNA3.SA": "CSN ON",
    "GGBR4.SA": "Gerdau PN",
    "USIM5.SA": "Usiminas PNA",
    "NVDC34.SA": "NVIDIA BDR",
    # United States — tech & semis
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Class A",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "AMD": "Advanced Micro Devices",
    "AVGO": "Broadcom",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    # United States — financial & consumer
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "WMT": "Walmart",
    "COST": "Costco",
    "HD": "Home Depot",
    # United States — energy & health
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth",
}

# Index/ETF tickers used as CROSS-ASSET CONTEXT (regime, breadth, relative
# strength) but NOT as trading universe. They are downloaded and joined into
# features per date but never receive predictions/signals.
CONTEXT_INDEX_TICKERS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IWM": "Russell 2000 ETF",
    "^GSPC": "S&P 500 Index",
    "^BVSP": "Ibovespa Index",
}

OFFICIAL_WEBSITES = {
    "PETR4.SA": "https://www.petrobras.com.br",
    "VALE3.SA": "https://vale.com",
    "ITUB4.SA": "https://www.itau.com.br",
    "BBDC4.SA": "https://banco.bradesco/html/classic/index.shtm",
    "BBAS3.SA": "https://www.bb.com.br",
    "SANB11.SA": "https://www.santander.com.br",
    "B3SA3.SA": "https://www.b3.com.br",
    "BPAC11.SA": "https://www.btgpactual.com",
    "ABEV3.SA": "https://www.ambev.com.br",
    "MGLU3.SA": "https://ri.magazineluiza.com.br",
    "LREN3.SA": "https://www.lojasrenner.com.br",
    "RENT3.SA": "https://ri.localiza.com",
    "RAIL3.SA": "https://ri.rumolog.com",
    "WEGE3.SA": "https://www.weg.net/institutional/BR/pt/",
    "EQTL3.SA": "https://www.equatorialenergia.com.br",
    "SBSP3.SA": "https://www.sabesp.com.br",
    "RDOR3.SA": "https://rededorsaoluiz.com.br",
    "HAPV3.SA": "https://www.hapvida.com.br",
    "SUZB3.SA": "https://www.suzano.com.br",
    "KLBN11.SA": "https://www.klabin.com.br",
    "CSNA3.SA": "https://www.csn.com.br",
    "GGBR4.SA": "https://www.gerdau.com",
    "USIM5.SA": "https://www.usiminas.com",
    "NVDC34.SA": "https://www.nvidia.com",
    "AAPL": "https://www.apple.com",
    "MSFT": "https://www.microsoft.com",
    "GOOGL": "https://abc.xyz",
    "AMZN": "https://www.amazon.com",
    "NVDA": "https://www.nvidia.com",
    "META": "https://about.meta.com",
    "TSLA": "https://www.tesla.com",
    "AMD": "https://www.amd.com",
    "AVGO": "https://www.broadcom.com",
    "ORCL": "https://www.oracle.com",
    "CRM": "https://www.salesforce.com",
    "JPM": "https://www.jpmorganchase.com",
    "BAC": "https://www.bankofamerica.com",
    "GS": "https://www.goldmansachs.com",
    "WMT": "https://corporate.walmart.com",
    "COST": "https://www.costco.com",
    "HD": "https://www.homedepot.com",
    "XOM": "https://corporate.exxonmobil.com",
    "CVX": "https://www.chevron.com",
    "JNJ": "https://www.jnj.com",
    "UNH": "https://www.unitedhealthgroup.com",
}

DEFAULT_PRICE_PERIOD = "10y"
DEFAULT_PRICE_INTERVAL = "1d"
