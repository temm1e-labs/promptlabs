"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { Plus } from "lucide-react";

import { Badge, badgeVariants } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useExperiments, useProject } from "@/lib/api/hooks";
import { formatCost, formatScore, scoreBand } from "@/lib/utils";
import type { ExperimentStatus } from "@/lib/api/types";

const STATUS_VARIANT: Record<ExperimentStatus, "default" | "good" | "mid" | "bad" | "outline"> = {
  pending: "outline",
  running: "default",
  paused: "outline",
  converged: "good",
  accepted: "good",
  overfit: "mid",
  exhausted: "mid",
  failed: "bad",
};

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const { data: project } = useProject(id);
  const { data: experiments, isLoading } = useExperiments(id);

  return (
    <div className="mx-auto max-w-6xl">
      <nav className="mb-6 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        <Link href="/" className="hover:text-foreground">
          projects
        </Link>
        <span className="mx-2">/</span>
        <span className="text-foreground">{project?.name ?? "…"}</span>
      </nav>

      <header className="mb-8 flex items-end justify-between border-b border-border pb-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{project?.name ?? "Project"}</h1>
          {project?.description && (
            <p className="mt-1 text-sm text-muted-foreground">{project.description}</p>
          )}
        </div>
        <Link href={`/projects/${id}/experiments/new`}>
          <Button>
            <Plus className="h-3.5 w-3.5" />
            New experiment
          </Button>
        </Link>
      </header>

      {isLoading ? (
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <Card key={i} className="h-16 animate-pulse" />
          ))}
        </div>
      ) : experiments && experiments.length === 0 ? (
        <section className="grid place-items-center rounded-lg border border-dashed border-border py-24">
          <div className="text-center">
            <Badge variant="outline" className="mb-4">
              no experiments yet
            </Badge>
            <p className="text-sm text-muted-foreground">
              Start your first experiment with a cold-start idea or an existing prompt.
            </p>
          </div>
        </section>
      ) : (
        <div className="overflow-hidden rounded-lg border border-border">
          <table className="w-full">
            <thead className="border-b border-border bg-card/50">
              <tr className="text-left">
                <th className="px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  name
                </th>
                <th className="px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  status
                </th>
                <th className="px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  iter
                </th>
                <th className="px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  best score
                </th>
                <th className="px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  cost
                </th>
                <th className="px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  targets
                </th>
              </tr>
            </thead>
            <tbody>
              {experiments?.map((e) => (
                <tr
                  key={e.id}
                  className="cursor-pointer border-t border-border transition-colors hover:bg-card/50"
                >
                  <td className="px-4 py-3">
                    <Link
                      href={`/projects/${id}/experiments/${e.id}`}
                      className="block text-sm font-medium tracking-tight"
                    >
                      {e.name}
                      <span className="mt-0.5 block text-xs text-muted-foreground line-clamp-1">
                        {e.intent}
                      </span>
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={STATUS_VARIANT[e.status]}>{e.status}</Badge>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                    {e.current_iteration}
                  </td>
                  <td className="px-4 py-3">
                    {e.best_score != null ? (
                      <Badge variant={scoreBand(e.best_score) as "good" | "mid" | "bad"}>
                        {formatScore(e.best_score)}
                      </Badge>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                    {formatCost(e.cost_usd)} / {formatCost(e.budget_usd)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1">
                      {e.target_models.slice(0, 3).map((m) => (
                        <span
                          key={m}
                          className={badgeVariants({ variant: "outline" })}
                          title={m}
                        >
                          {m.split("/")[1] ?? m}
                        </span>
                      ))}
                      {e.target_models.length > 3 && (
                        <span className={badgeVariants({ variant: "outline" })}>
                          +{e.target_models.length - 3}
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
