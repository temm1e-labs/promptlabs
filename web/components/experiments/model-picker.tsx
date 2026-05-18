"use client";

import { Check } from "lucide-react";

import { cn } from "@/lib/utils";
import { useModels } from "@/lib/api/hooks";

export function ModelPicker({
  selected,
  onChange,
  label = "Target models",
}: {
  selected: string[];
  onChange: (next: string[]) => void;
  label?: string;
}) {
  const { data: groups, isLoading } = useModels();
  const toggle = (id: string) => {
    if (selected.includes(id)) onChange(selected.filter((s) => s !== id));
    else onChange([...selected, id]);
  };

  if (isLoading) {
    return (
      <div className="h-32 animate-pulse rounded-md border border-dashed border-border" />
    );
  }

  return (
    <div className="space-y-4">
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label} — pick one or more
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {groups?.map((g) => (
          <div key={g.provider} className="space-y-1.5">
            <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/70">
              {g.provider}
            </div>
            <div className="space-y-1">
              {g.models.map((m) => {
                const isActive = selected.includes(m.id);
                return (
                  <button
                    type="button"
                    key={m.id}
                    onClick={() => toggle(m.id)}
                    className={cn(
                      "flex w-full items-center gap-2.5 rounded-md border px-2.5 py-1.5 text-left transition-colors",
                      isActive
                        ? "border-primary/50 bg-primary/5"
                        : "border-border bg-card hover:border-border/80",
                    )}
                  >
                    <div
                      className={cn(
                        "flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-sm border",
                        isActive ? "border-primary bg-primary text-primary-foreground" : "border-border",
                      )}
                    >
                      {isActive && <Check className="h-2.5 w-2.5" />}
                    </div>
                    <span className="flex-1 truncate text-sm">{m.label}</span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {m.id.split("/")[0]}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
