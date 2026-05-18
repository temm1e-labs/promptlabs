"use client";

import { Check } from "lucide-react";

import { cn } from "@/lib/utils";
import type { OptimizationObjective } from "@/lib/api/types";

const OBJECTIVES: { value: OptimizationObjective; label: string; description: string }[] = [
  { value: "accuracy", label: "Accuracy", description: "Correctness against the task" },
  { value: "robustness", label: "Robustness", description: "Avoid hallucinations & instruction violations" },
  { value: "cost", label: "Cost", description: "Minimize tokens / $$$" },
  { value: "latency", label: "Latency", description: "Minimize response time" },
  { value: "format_adherence", label: "Format", description: "Strict output structure" },
  { value: "brevity", label: "Brevity", description: "Short, direct outputs" },
  { value: "tone", label: "Tone", description: "Match a target voice" },
];

export function ObjectivePicker({
  selected,
  onChange,
}: {
  selected: OptimizationObjective[];
  onChange: (next: OptimizationObjective[]) => void;
}) {
  const toggle = (v: OptimizationObjective) => {
    if (selected.includes(v)) {
      onChange(selected.filter((s) => s !== v));
    } else {
      onChange([...selected, v]);
    }
  };

  return (
    <div className="grid gap-2 md:grid-cols-2">
      {OBJECTIVES.map((o) => {
        const isActive = selected.includes(o.value);
        return (
          <button
            type="button"
            key={o.value}
            onClick={() => toggle(o.value)}
            className={cn(
              "flex items-start gap-3 rounded-md border px-3 py-2.5 text-left transition-colors",
              isActive
                ? "border-primary/50 bg-primary/5"
                : "border-border bg-card hover:border-border/80",
            )}
          >
            <div
              className={cn(
                "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-sm border",
                isActive ? "border-primary bg-primary text-primary-foreground" : "border-border",
              )}
            >
              {isActive && <Check className="h-3 w-3" />}
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium leading-tight">{o.label}</div>
              <div className="mt-0.5 text-xs text-muted-foreground">{o.description}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
