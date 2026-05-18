"use client";

import { Check } from "lucide-react";

import { cn } from "@/lib/utils";

export type RunPreset = {
  id: "quick" | "standard" | "thorough" | "custom";
  label: string;
  caption: string;
  description: string;
  budget_usd: number;
  max_iterations: number;
  eval_size: number;
  recommended?: boolean;
};

export const PRESETS: RunPreset[] = [
  {
    id: "quick",
    label: "Quick",
    caption: "$2 · 5 iter · 20 items",
    description: "Rapid prototyping. Smoke-test an approach in a few minutes.",
    budget_usd: 2,
    max_iterations: 5,
    eval_size: 20,
  },
  {
    id: "standard",
    label: "Standard",
    caption: "$10 · 10 iter · 50 items",
    description: "Real prompt development. Statistically meaningful, costs minutes.",
    budget_usd: 10,
    max_iterations: 10,
    eval_size: 50,
    recommended: true,
  },
  {
    id: "thorough",
    label: "Thorough",
    caption: "$25 · 20 iter · 100 items",
    description: "Production-bound prompts. High statistical power, slower.",
    budget_usd: 25,
    max_iterations: 20,
    eval_size: 100,
  },
];

export function PresetPicker({
  selectedId,
  onSelect,
}: {
  selectedId: RunPreset["id"];
  onSelect: (preset: RunPreset["id"]) => void;
}) {
  return (
    <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
      {PRESETS.map((p) => {
        const active = selectedId === p.id;
        return (
          <button
            key={p.id}
            type="button"
            onClick={() => onSelect(p.id)}
            className={cn(
              "relative flex h-full flex-col rounded-md border px-4 py-3 text-left transition-colors",
              active
                ? "border-primary/50 bg-primary/5"
                : "border-border bg-card hover:border-border/80",
            )}
          >
            {active && (
              <Check className="absolute right-3 top-3 h-3.5 w-3.5 text-primary" />
            )}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium tracking-tight">{p.label}</span>
              {p.recommended && (
                <span className="rounded-full bg-primary/15 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-widest text-primary">
                  recommended
                </span>
              )}
            </div>
            <div className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              {p.caption}
            </div>
            <p className="mt-2 text-xs text-muted-foreground">{p.description}</p>
          </button>
        );
      })}
      <button
        type="button"
        onClick={() => onSelect("custom")}
        className={cn(
          "relative flex h-full flex-col rounded-md border border-dashed px-4 py-3 text-left transition-colors",
          selectedId === "custom"
            ? "border-primary/50 bg-primary/5"
            : "border-border bg-card hover:border-border/80",
        )}
      >
        {selectedId === "custom" && (
          <Check className="absolute right-3 top-3 h-3.5 w-3.5 text-primary" />
        )}
        <div className="text-sm font-medium tracking-tight">Custom</div>
        <div className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          set your own
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Manually configure budget, max iterations, and eval size.
        </p>
      </button>
    </div>
  );
}
