"use client";

import { ExternalLink, LineChart } from "lucide-react";
import { tradingViewBaseTicker, tradingViewChartUrl, tradingViewExchange, tradingViewSymbolPageUrl } from "@/lib/tradingview";

type TradingViewSymbolInfoProps = {
  ticker: string;
  name: string;
};

export function TradingViewSymbolInfo({ ticker, name }: TradingViewSymbolInfoProps) {
  const exchange = tradingViewExchange(ticker);
  const baseTicker = tradingViewBaseTicker(ticker);

  return (
    <div className="min-h-[240px] overflow-hidden rounded-xl border border-white/10 bg-white/[0.02] p-4 backdrop-blur sm:min-h-[282px]">
      <div className="flex h-full flex-col justify-between gap-4" aria-label={`TradingView links for ${name}`}>
        <div className="space-y-3">
          <div className="inline-flex items-center gap-2 rounded-full border border-sky-400/20 bg-sky-400/10 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.24em] text-sky-100">
            <LineChart className="h-3.5 w-3.5" />
            TradingView
          </div>
          <div>
            <div className="text-lg font-semibold text-foreground">{name}</div>
            <div className="mt-1 text-sm text-muted-foreground">{baseTicker} · {exchange}</div>
          </div>
          <p className="max-w-md text-sm leading-6 text-muted-foreground">
            A ficha oficial e o Supercharts seguem disponiveis, mas sem embutir o script externo que gerava erro no console durante o ciclo de renderizacao em dev.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <a
            href={tradingViewSymbolPageUrl(ticker)}
            target="_blank"
            rel="noopener noreferrer nofollow"
            className="group rounded-2xl border border-white/10 bg-white/5 p-4 transition-colors hover:border-white/20 hover:bg-white/10"
          >
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-medium text-foreground">Ficha oficial</div>
                <div className="mt-1 text-xs text-muted-foreground">Resumo publico do ativo no TradingView.</div>
              </div>
              <ExternalLink className="h-4 w-4 text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </div>
          </a>

          <a
            href={tradingViewChartUrl(ticker)}
            target="_blank"
            rel="noopener noreferrer nofollow"
            className="group rounded-2xl border border-sky-400/25 bg-sky-400/10 p-4 transition-colors hover:border-sky-300/40 hover:bg-sky-400/15"
          >
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-medium text-sky-100">Abrir no Supercharts</div>
                <div className="mt-1 text-xs text-sky-100/70">Grafico completo com indicadores e layout do TradingView.</div>
              </div>
              <ExternalLink className="h-4 w-4 text-sky-100/80 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </div>
          </a>
        </div>
      </div>
    </div>
  );
}