"use client";

import Link from "next/link";
import { FlaskConical } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { useAllExperiments } from "@/lib/api/hooks";
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
  cancelled: "outline",
};

function timeAgo(iso: string): string {
  const t = new Date(iso).getTime();
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function ExperimentsPage() {
  const { data: experiments, isLoading } = useAllExperiments();

  return (
    <div className="mx-auto max-w-6xl">
      <header className="mb-8 flex items-end justify-between border-b border-border pb-6">
        <div>
          <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            workspace
          </p>
          <h1 className="text-2xl font-semibold tracking-tight">All experiments</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Recent runs across every project. Most recent first.
          </p>
        </div>
      </header>

      {isLoading ? (
        <div className="space-y-2">
          {[0, 1, 2, 3].map((i) => (
            <Card key={i} className="h-16 animate-pulse" />
          ))}
        </div>
      ) : experiments && experiments.length === 0 ? (
        <section className="grid place-items-center rounded-lg border border-dashed border-border py-24">
          <div className="text-center">
            <FlaskConical className="mx-auto mb-4 h-6 w-6 text-muted-foreground" />
            <Badge variant="outline" className="mb-3">
              no experiments yet
            </Badge>
            <p className="text-sm text-muted-foreground">
              Create a project and start your first experiment.
            </p>
          </div>
        </section>
      ) : (
        <div className="overflow-hidden rounded-lg border border-border">
          <table className="w-full">
            <thead className="border-b border-border bg-card/50">
              <tr className="text-left">
                {["experiment", "project", "status", "iter", "best", "cost", "when"].map((h) => (
                  <th
                    key={h}
                    className="px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {experiments?.map((e) => (
                <tr
                  key={e.id}
                  className="border-t border-border transition-colors hover:bg-card/50"
                >
                  <td className="px-4 py-3">
                    <Link
                      href={`/projects/${e.project_id}/experiments/${e.id}`}
                      className="block text-sm font-medium tracking-tight"
                    >
                      {e.name}
                      <span className="mt-0.5 block text-xs text-muted-foreground line-clamp-1">
                        {e.intent}
                      </span>
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/projects/${e.project_id}`}
                      className="text-sm text-muted-foreground hover:text-foreground"
                    >
                      {e.project_name}
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
                  <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                    {timeAgo(e.created_at)}
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
