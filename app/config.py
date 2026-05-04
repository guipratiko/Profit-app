from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = PROJECT_ROOT / "storage"
DATABASE_PATH = STORAGE_DIR / "profit_app.sqlite3"

INITIAL_ASSETS = {
    "PETR4.SA": "Petrobras PN",
    "VALE3.SA": "Vale ON",
    "ITUB4.SA": "Itau Unibanco PN",
    "BBDC4.SA": "Bradesco PN",
    "BBAS3.SA": "Banco do Brasil ON",
    "ABEV3.SA": "Ambev ON",
    "WEGE3.SA": "WEG ON",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Class A",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
}

OFFICIAL_WEBSITES = {
    "PETR4.SA": "https://www.petrobras.com.br",
    "VALE3.SA": "https://vale.com",
    "ITUB4.SA": "https://www.itau.com.br",
    "BBDC4.SA": "https://banco.bradesco/html/classic/index.shtm",
    "BBAS3.SA": "https://www.bb.com.br",
    "ABEV3.SA": "https://www.ambev.com.br",
    "WEGE3.SA": "https://www.weg.net/institutional/BR/pt/",
    "AAPL": "https://www.apple.com",
    "MSFT": "https://www.microsoft.com",
    "GOOGL": "https://abc.xyz",
    "AMZN": "https://www.amazon.com",
    "NVDA": "https://www.nvidia.com",
    "META": "https://about.meta.com",
    "TSLA": "https://www.tesla.com",
}

DEFAULT_PRICE_PERIOD = "10y"
DEFAULT_PRICE_INTERVAL = "1d"
