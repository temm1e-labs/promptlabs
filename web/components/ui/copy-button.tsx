"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/utils";

export function CopyButton({
  text,
  className,
  label = "Copy",
  iconOnly = false,
}: {
  text: string;
  className?: string;
  label?: string;
  iconOnly?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const onClick = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore — clipboard might be blocked
    }
  };

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className={cn(
        "inline-flex h-7 items-center gap-1.5 rounded-md px-2 font-mono text-[10px] uppercase tracking-widest",
        "text-muted-foreground transition-colors",
        "hover:bg-accent hover:text-foreground",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        className,
      )}
    >
      {copied ? <Check className="h-3 w-3 text-[var(--score-good)]" /> : <Copy className="h-3 w-3" />}
      {!iconOnly && <span>{copied ? "Copied" : label}</span>}
    </button>
  );
}
