import * as React from "react";
import { cn } from "@/lib/utils";

type BadgeTone = "neutral" | "good" | "warn" | "bad" | "info";

const tones: Record<BadgeTone, string> = {
  neutral: "border-white/10 bg-white/5 text-foreground/80",
  good: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200 shadow-[0_0_18px_-6px_rgba(52,211,153,0.55)]",
  warn: "border-amber-300/30 bg-amber-300/10 text-amber-100 shadow-[0_0_18px_-6px_rgba(252,211,77,0.5)]",
  bad: "border-rose-400/35 bg-rose-400/10 text-rose-200 shadow-[0_0_18px_-6px_rgba(244,114,128,0.55)]",
  info: "border-sky-400/30 bg-sky-400/10 text-sky-200 shadow-[0_0_18px_-6px_rgba(56,189,248,0.55)]"
};

export function Badge({
  className,
  tone = "neutral",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return (
    <span
      className={cn(
        "inline-flex min-h-6 max-w-full items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 text-[11px] font-medium uppercase tracking-wide backdrop-blur",
        tones[tone],
        className
      )}
      {...props}
    />
  );
}
