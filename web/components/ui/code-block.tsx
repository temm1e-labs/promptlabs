"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { CopyButton } from "@/components/ui/copy-button";
import { cn } from "@/lib/utils";

export function CodeBlock({
  content,
  label,
  className,
  maxHeight = 360,
  defaultCollapsed = false,
}: {
  content: string;
  label?: string;
  className?: string;
  maxHeight?: number;
  defaultCollapsed?: boolean;
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <div className={cn("rounded-md border border-border bg-muted/30", className)}>
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <button
          type="button"
          onClick={() => setCollapsed((v) => !v)}
          className={cn(
            "inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest",
            "text-muted-foreground transition-colors hover:text-foreground",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            "rounded px-1.5 py-0.5",
          )}
          aria-expanded={!collapsed}
        >
          {collapsed ? (
            <ChevronRight className="h-3 w-3" />
          ) : (
            <ChevronDown className="h-3 w-3" />
          )}
          <span>{label ?? "prompt"}</span>
          <span className="text-muted-foreground/60">· {content.length} chars</span>
        </button>
        <CopyButton text={content} />
      </div>
      {!collapsed && (
        <pre
          className="overflow-auto px-3 py-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-words"
          style={{ maxHeight }}
        >
          {content}
        </pre>
      )}
    </div>
  );
}
