import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex h-9 items-center justify-center gap-2 rounded-xl border px-3.5 text-sm font-medium tracking-tight transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 backdrop-blur",
  {
    variants: {
      variant: {
        default:
          "border-sky-400/40 bg-gradient-to-br from-sky-400/30 to-sky-500/10 text-sky-50 hover:from-sky-400/45 hover:to-sky-500/20 shadow-[0_0_24px_-8px_rgba(56,189,248,0.7)] hover:shadow-[0_0_32px_-8px_rgba(56,189,248,0.9)]",
        secondary:
          "border-white/10 bg-white/5 text-foreground hover:bg-white/10 hover:border-white/20",
        quiet:
          "border-transparent bg-transparent text-muted-foreground hover:bg-white/5 hover:text-foreground",
        danger:
          "border-rose-400/40 bg-gradient-to-br from-rose-500/30 to-rose-600/10 text-rose-50 hover:from-rose-500/45"
      },
      size: { default: "h-9 px-3.5", icon: "h-9 w-9 px-0", sm: "h-7 px-2.5 text-xs rounded-lg" }
    },
    defaultVariants: { variant: "default", size: "default" }
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return <button className={cn(buttonVariants({ variant, size, className }))} {...props} />;
}
