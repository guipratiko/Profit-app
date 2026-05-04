export function tradingViewBaseTicker(ticker: string) {
  return ticker.toUpperCase().replace(/\.SA$/, "");
}

export function tradingViewExchange(ticker: string) {
  return ticker.toUpperCase().endsWith(".SA") ? "BMFBOVESPA" : "NASDAQ";
}

export function tradingViewSymbol(ticker: string) {
  return `${tradingViewExchange(ticker)}:${tradingViewBaseTicker(ticker)}`;
}

export function tradingViewSymbolSlug(ticker: string) {
  return `${tradingViewExchange(ticker)}-${tradingViewBaseTicker(ticker)}`;
}

export function tradingViewSymbolPageUrl(ticker: string) {
  return `https://www.tradingview.com/symbols/${tradingViewSymbolSlug(ticker)}/`;
}

export function tradingViewChartUrl(ticker: string) {
  return `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(tradingViewSymbol(ticker))}`;
}