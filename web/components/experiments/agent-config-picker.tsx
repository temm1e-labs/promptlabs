"use client";

import { ChevronDown, ChevronRight, Cpu } from "lucide-react";
import { useState } from "react";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useModels } from "@/lib/api/hooks";
import { cn } from "@/lib/utils";
import type { AgentConfig } from "@/lib/api/types";

type Role = "writer" | "evalgen" | "judge" | "optimizer";
const ROLES: { id: Role; label: string; help: string }[] = [
  { id: "writer", label: "Writer", help: "Drafts v0 from intent (or extracts vars from your existing prompt)" },
  { id: "evalgen", label: "EvalGen", help: "Generates the rubric + N test cases" },
  { id: "judge", label: "Judge", help: "Scores each output against the rubric" },
  { id: "optimizer", label: "Optimizer", help: "Produces surgical diffs based on failure samples" },
];

export function AgentConfigPicker({
  labDefault,
  onLabDefault,
  config,
  onConfig,
}: {
  labDefault: string;
  onLabDefault: (id: string) => void;
  config: AgentConfig;
  onConfig: (next: AgentConfig) => void;
}) {
  const { data: groups } = useModels();
  const [expanded, setExpanded] = useState(false);

  const allOptions = (groups ?? []).flatMap((g) => g.models);
  const optionsByProvider = groups ?? [];

  const roleValue = (role: Role): string => {
    const key = `${role}_model` as keyof AgentConfig;
    return (config[key] as string | undefined) || labDefault || "";
  };

  const setRole = (role: Role, value: string | null) => {
    const key = `${role}_model` as keyof AgentConfig;
    onConfig({ ...config, [key]: value });
  };

  return (
    <div className="space-y-4">
      <div>
        <div className="mb-2 flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          <Cpu className="h-3 w-3" />
          lab default model
          <span className="text-muted-foreground/60">
            · used for all agent roles unless overridden below
          </span>
        </div>
        <Select value={labDefault} onValueChange={onLabDefault}>
          <SelectTrigger className="max-w-md">
            <SelectValue placeholder="Pick a model…" />
          </SelectTrigger>
          <SelectContent>
            {optionsByProvider.map((g) => (
              <SelectGroup key={g.provider}>
                <SelectLabel>{g.provider}</SelectLabel>
                {g.models.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            ))}
          </SelectContent>
        </Select>
      </div>

      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className={cn(
          "flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest",
          "text-muted-foreground hover:text-foreground transition-colors",
          "rounded px-1 py-0.5",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        )}
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        advanced · per-role override
      </button>

      {expanded && (
        <div className="grid gap-3 rounded-md border border-dashed border-border bg-muted/30 p-3 md:grid-cols-2">
          {ROLES.map((r) => {
            const value = roleValue(r.id);
            const usingDefault =
              !(config[`${r.id}_model` as keyof AgentConfig] as string | null | undefined);
            return (
              <div key={r.id} className="space-y-1">
                <div className="flex items-baseline justify-between">
                  <div className="font-mono text-[10px] uppercase tracking-widest">
                    {r.label}
                  </div>
                  {!usingDefault && (
                    <button
                      type="button"
                      onClick={() => setRole(r.id, null)}
                      className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
                    >
                      reset
                    </button>
                  )}
                </div>
                <Select value={value} onValueChange={(v) => setRole(r.id, v)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Use lab default" />
                  </SelectTrigger>
                  <SelectContent>
                    {allOptions.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-[10px] text-muted-foreground/70">{r.help}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
