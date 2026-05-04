"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";

export function GateChart({ gates }: { gates: Record<string, boolean> }) {
  const rows = Object.entries(gates || {}).map(([name, value]) => ({ name: name.replaceAll("_", " "), value: value ? 1 : 0, ok: !!value }));
  return (
    <div className="h-52 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 30, left: 0 }}>
          <defs>
            <linearGradient id="barOk" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#34d399" stopOpacity={0.95} />
              <stop offset="100%" stopColor="#34d399" stopOpacity={0.35} />
            </linearGradient>
            <linearGradient id="barBad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f87171" stopOpacity={0.85} />
              <stop offset="100%" stopColor="#f87171" stopOpacity={0.25} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
          <XAxis dataKey="name" angle={-24} textAnchor="end" interval={0} height={70} tick={{ fontSize: 11, fill: "rgba(226,232,240,0.65)" }} stroke="rgba(148,163,184,0.25)" />
          <YAxis domain={[0, 1]} ticks={[0, 1]} tick={{ fontSize: 11, fill: "rgba(226,232,240,0.65)" }} stroke="rgba(148,163,184,0.25)" />
          <Tooltip
            cursor={{ fill: "rgba(148,163,184,0.08)" }}
            contentStyle={{ background: "rgba(15,23,42,0.92)", border: "1px solid rgba(148,163,184,0.2)", borderRadius: 12, color: "#e2e8f0", backdropFilter: "blur(12px)" }}
            formatter={(value) => (value === 1 ? "OK" : "Bloqueado")}
          />
          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
            {rows.map((row, idx) => (
              <Cell key={idx} fill={row.ok ? "url(#barOk)" : "url(#barBad)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
