"use client";

import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";

export const Tabs = TabsPrimitive.Root;

export function TabsList({
  className,
  ...props
}: React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      className={cn(
        "flex h-auto w-full max-w-full items-center gap-1 overflow-x-auto rounded-2xl border border-white/10 bg-white/5 p-1 scrollbar-thin backdrop-blur-xl shadow-[0_1px_0_0_rgba(255,255,255,0.06)_inset] sm:inline-flex sm:w-auto",
        className
      )}
      {...props}
    />
  );
}

export function TabsTrigger({
  className,
  ...props
}: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        "inline-flex h-9 shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-xl px-3 text-xs font-medium text-muted-foreground transition-all duration-200 sm:px-4 sm:text-sm",
        "hover:text-foreground hover:bg-white/5",
        "data-[state=active]:bg-gradient-to-br data-[state=active]:from-sky-400/30 data-[state=active]:to-purple-500/15",
        "data-[state=active]:text-foreground data-[state=active]:shadow-[0_0_24px_-8px_rgba(56,189,248,0.7),inset_0_1px_0_0_rgba(255,255,255,0.12)]",
        "data-[state=active]:border data-[state=active]:border-white/15",
        className
      )}
      {...props}
    />
  );
}

export function TabsContent({
  className,
  ...props
}: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>) {
  return <TabsPrimitive.Content className={cn("mt-5", className)} {...props} />;
}
