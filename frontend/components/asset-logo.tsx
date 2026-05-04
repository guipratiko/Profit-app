"use client";

import { useMemo, useState } from "react";
import { assetLogoUrl, type Asset } from "@/lib/api";
import { cn } from "@/lib/utils";

type AssetLogoProps = {
  asset?: Asset | null;
  size?: "sm" | "md" | "lg";
  className?: string;
};

const sizeClasses = {
  sm: "h-10 w-10 rounded-xl",
  md: "h-12 w-12 rounded-2xl",
  lg: "h-14 w-14 rounded-[20px]"
} as const;

const fallbackTextClasses = {
  sm: "text-[10px] tracking-[0.18em]",
  md: "text-[11px] tracking-[0.18em]",
  lg: "text-xs tracking-[0.2em]"
} as const;

function fallbackLabel(asset?: Asset | null) {
  return asset?.ticker?.split(".")[0]?.slice(0, 5) || "--";
}

export function AssetLogo({ asset, size = "md", className }: AssetLogoProps) {
  const [hasImageError, setHasImageError] = useState(false);
  const logoSrc = useMemo(() => assetLogoUrl(asset), [asset]);

  if (!asset) {
    return (
      <div
        className={cn(
          "flex shrink-0 items-center justify-center border border-white/10 bg-white/10 font-semibold uppercase text-muted-foreground",
          sizeClasses[size],
          fallbackTextClasses[size],
          className
        )}
      >
        --
      </div>
    );
  }

  return (
    <div
      className={cn(
        "relative flex shrink-0 items-center justify-center overflow-hidden border border-white/10 bg-[radial-gradient(circle_at_30%_20%,rgba(56,189,248,0.18),transparent_48%),linear-gradient(145deg,rgba(15,23,42,0.94),rgba(17,24,39,0.78))] shadow-[0_20px_40px_-28px_rgba(15,23,42,0.85)]",
        sizeClasses[size],
        className
      )}
      title={`${asset.name} · logo oficial`}
    >
      <div className="absolute inset-[1px] rounded-[inherit] border border-white/6 bg-gradient-to-br from-white/[0.08] via-white/[0.03] to-transparent" />
      {!hasImageError && logoSrc ? (
        <img
          src={logoSrc}
          alt={`${asset.name} logo oficial`}
          className="relative z-[1] h-full w-full object-contain p-1.5 drop-shadow-[0_6px_18px_rgba(15,23,42,0.45)]"
          loading="lazy"
          referrerPolicy="no-referrer"
          onError={() => setHasImageError(true)}
        />
      ) : (
        <span
          className={cn(
            "relative z-[1] font-semibold uppercase text-slate-100",
            fallbackTextClasses[size]
          )}
        >
          {fallbackLabel(asset)}
        </span>
      )}
    </div>
  );
}