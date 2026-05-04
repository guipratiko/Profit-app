"use client";

import { useEffect, useRef } from "react";
import { ColorType, createChart, LineSeries, type IChartApi, type ISeriesApi } from "lightweight-charts";
import type { PriceRow } from "@/lib/api";

export function PriceChart({ rows }: { rows: PriceRow[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      autoSize: true,
      height: 330,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "rgba(226,232,240,0.78)" },
      grid: { vertLines: { color: "rgba(148,163,184,0.08)" }, horzLines: { color: "rgba(148,163,184,0.08)" } },
      rightPriceScale: { borderColor: "rgba(148,163,184,0.15)" },
      timeScale: { borderColor: "rgba(148,163,184,0.15)", timeVisible: true },
      crosshair: { mode: 1 }
    });
    const series = chart.addSeries(LineSeries, { color: "#38bdf8", lineWidth: 2 });
    chartRef.current = chart;
    seriesRef.current = series;

    const resize = () => chart.applyOptions({ width: containerRef.current?.clientWidth || 0 });
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    const data = rows
      .filter((row) => row.date && typeof row.close === "number")
      .map((row) => ({ time: row.date, value: Number(row.close) }));
    series.setData(data);
    chartRef.current?.timeScale().fitContent();
  }, [rows]);

  return <div ref={containerRef} className="h-[330px] w-full overflow-hidden rounded-xl border border-white/10 bg-white/[0.02] backdrop-blur" />;
}
