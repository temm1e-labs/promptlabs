"use client";

import { useState } from "react";
import { ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";

type Edit = {
  op: string;
  anchor?: string | null;
  new_text?: string | null;
  reason: string;
  targets_criterion?: string | null;
};

export function PromptDiff({
  current,
  edits,
  applied,
  skipped,
}: {
  current: string;
  edits: Edit[];
  applied?: number;
  skipped?: number;
}) {
  const [showFull, setShowFull] = useState(false);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          diff · {applied ?? 0} applied{skipped ? ` · ${skipped} skipped` : ""}
        </div>
        <button
          type="button"
          onClick={() => setShowFull((v) => !v)}
          className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
        >
          {showFull ? "hide prompt" : "view full prompt"}
        </button>
      </div>
      <div className="space-y-1.5">
        {edits.map((e, i) => (
          <EditCard key={i} edit={e} />
        ))}
        {edits.length === 0 && (
          <div className="rounded-md border border-dashed border-border px-3 py-4 text-center text-xs text-muted-foreground">
            no edits this iteration
          </div>
        )}
      </div>
      {showFull && (
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/30 p-3 font-mono text-[11px] leading-relaxed">
          {current}
        </pre>
      )}
    </div>
  );
}

function EditCard({ edit }: { edit: Edit }) {
  return (
    <div className="rounded-md border border-border bg-card/50 p-2.5">
      <div className="flex items-baseline gap-2">
        <span
          className={cn(
            "rounded-sm px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-widest",
            edit.op === "delete"
              ? "bg-[var(--score-bad)]/15 text-[var(--score-bad)]"
              : edit.op === "append" || edit.op.startsWith("insert")
              ? "bg-[var(--score-good)]/15 text-[var(--score-good)]"
              : "bg-primary/15 text-primary",
          )}
        >
          {edit.op}
        </span>
        {edit.targets_criterion && (
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            → {edit.targets_criterion}
          </span>
        )}
      </div>
      <p className="mt-1.5 text-xs text-muted-foreground">{edit.reason}</p>
      {edit.anchor && (
        <div className="mt-2 flex items-start gap-1.5 rounded bg-[var(--score-bad)]/8 px-2 py-1">
          <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-[var(--score-bad)]" />
          <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[11px] text-[var(--score-bad)]">
            {edit.anchor}
          </pre>
        </div>
      )}
      {edit.new_text && (
        <div className="mt-1 flex items-start gap-1.5 rounded bg-[var(--score-good)]/8 px-2 py-1">
          <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-[var(--score-good)]" />
          <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[11px] text-[var(--score-good)]">
            {edit.new_text}
          </pre>
        </div>
      )}
    </div>
  );
}
