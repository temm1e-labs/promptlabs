"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useMemo } from "react";
import { Square, Trash2 } from "lucide-react";

import { ScoreTrajectory, type TrajectoryPoint } from "@/components/charts/score-trajectory";
import { TrainHoldoutGap, type GapPoint } from "@/components/charts/train-holdout-gap";
import { PromptDiff } from "@/components/lab/prompt-diff";
import { SSERail } from "@/components/lab/sse-rail";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { CodeBlock } from "@/components/ui/code-block";
import { Separator } from "@/components/ui/separator";
import {
  useCancelExperiment,
  useDeleteExperiment,
  useExperiment,
  useIterationStats,
  usePromptVersions,
  useRuns,
} from "@/lib/api/hooks";
import { formatCost, formatScore, scoreBand } from "@/lib/utils";
import type { ExperimentStatus, PromptVersion, Run } from "@/lib/api/types";
import type { IterationStats } from "@/lib/api/hooks";

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

const RUNNING_STATUSES: ExperimentStatus[] = ["pending", "running"];

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function ExperimentPage() {
  const { id: projectId, expId } = useParams<{ id: string; expId: string }>();
  const router = useRouter();
  const { data: exp } = useExperiment(expId);
  const { data: versions } = usePromptVersions(expId);
  const { data: runs } = useRuns(expId);
  const { data: stats } = useIterationStats(expId);
  const cancelMut = useCancelExperiment(expId);
  const deleteMut = useDeleteExperiment(expId, projectId);
  const isRunning = exp ? RUNNING_STATUSES.includes(exp.status) : false;
  const canDelete = exp && !isRunning;

  const onStop = async () => {
    if (!confirm("Stop this experiment? In-flight LLM calls will finish first.")) return;
    await cancelMut.mutateAsync();
  };

  const onDelete = async () => {
    if (!confirm("Delete this experiment and all its runs? This cannot be undone.")) return;
    await deleteMut.mutateAsync();
    router.push(`/projects/${projectId}`);
  };

  const targetModels = stats?.target_models ?? [];
  const isMultiModel = targetModels.length > 1;
  const trajectory = useMemo(
    () => buildTrajectory(stats?.iterations, targetModels),
    [stats, targetModels],
  );
  const gap = useMemo(() => buildGap(trajectory, targetModels), [trajectory, targetModels]);
  const bestPerModel = useMemo(() => bestHoldoutPerModel(stats), [stats]);
  const bestAggregated = useMemo(() => bestHoldoutAggregated(stats), [stats]);

  return (
    <div className="flex h-[calc(100vh-3rem)] gap-0 -m-6 lg:-mr-8">
      <div className="min-w-0 flex-1 overflow-y-auto px-8 py-6">
        <nav className="mb-6 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          <Link href="/" className="hover:text-foreground">
            projects
          </Link>
          <span className="mx-2">/</span>
          <Link href={`/projects/${projectId}`} className="hover:text-foreground">
            {exp ? "back" : "…"}
          </Link>
          <span className="mx-2">/</span>
          <span className="text-foreground">{exp?.name ?? "…"}</span>
        </nav>

        <header className="mb-8 flex items-start justify-between border-b border-border pb-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              {exp?.name ?? "Experiment"}
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              {exp?.intent ?? "Loading…"}
            </p>
            {exp?.failure_reason && (
              <p className="mt-2 text-xs text-[var(--score-bad)]">
                failure: {exp.failure_reason}
              </p>
            )}
          </div>
          <div className="flex items-center gap-4">
            {isMultiModel ? (
              <>
                {targetModels.map((m) => {
                  const best = bestPerModel[m];
                  return (
                    <Stat
                      key={m}
                      label={`best · ${shortModel(m)}`}
                      value={best != null ? formatScore(best) : "—"}
                      band={best != null ? (scoreBand(best) as "good" | "mid" | "bad") : undefined}
                    />
                  );
                })}
              </>
            ) : (
              <Stat
                label="best score"
                value={bestAggregated != null ? formatScore(bestAggregated) : "—"}
                band={
                  bestAggregated != null
                    ? (scoreBand(bestAggregated) as "good" | "mid" | "bad")
                    : undefined
                }
              />
            )}
            <Separator orientation="vertical" className="h-10" />
            <Stat label="iteration" value={exp ? `${exp.current_iteration} / ${exp.max_iterations}` : "—"} />
            <Separator orientation="vertical" className="h-10" />
            <Stat label="cost" value={exp ? `${formatCost(exp.cost_usd)} / ${formatCost(exp.budget_usd)}` : "—"} />
            <Separator orientation="vertical" className="h-10" />
            {exp && <Badge variant={STATUS_VARIANT[exp.status]}>{exp.status}</Badge>}
            {isRunning && (
              <Button
                variant="outline"
                size="sm"
                onClick={onStop}
                disabled={cancelMut.isPending}
              >
                <Square className="h-3 w-3" />
                {cancelMut.isPending ? "Stopping…" : "Stop"}
              </Button>
            )}
            {canDelete && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onDelete}
                disabled={deleteMut.isPending}
                aria-label="Delete experiment"
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            )}
          </div>
        </header>

        <section className="mb-8 grid gap-4 md:grid-cols-2">
          <Card className="p-5">
            <h3 className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              score trajectory
            </h3>
            <p className="mb-2 text-xs text-muted-foreground">
              Train (solid) vs holdout (dashed) per iteration. Higher is better.
            </p>
            <ScoreTrajectory data={trajectory} models={targetModels} />
          </Card>
          <Card className="p-5">
            <h3 className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              train − holdout gap
            </h3>
            <p className="mb-2 text-xs text-muted-foreground">
              Widening gap signals overfitting. Red threshold at 10pp.
            </p>
            <TrainHoldoutGap data={gap} models={targetModels} />
          </Card>
        </section>

        <section className="space-y-4">
          <h2 className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            iterations
          </h2>
          {versions && versions.length === 0 ? (
            <Card className="px-5 py-12 text-center">
              <p className="text-sm text-muted-foreground">
                Waiting for the writer agent to draft v0…
              </p>
            </Card>
          ) : (
            versions
              ?.slice()
              .reverse()
              .map((v) => (
                <IterationCard
                  key={v.id}
                  version={v}
                  runs={(runs ?? []).filter((r) => r.prompt_version_id === v.id)}
                />
              ))
          )}
        </section>
      </div>

      <SSERail experimentId={expId} apiUrl={API_URL} />
    </div>
  );
}

function Stat({
  label,
  value,
  band,
}: {
  label: string;
  value: string;
  band?: "good" | "mid" | "bad";
}) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div
        className={
          band === "good"
            ? "text-base font-medium text-[var(--score-good)]"
            : band === "mid"
            ? "text-base font-medium text-[var(--score-mid)]"
            : band === "bad"
            ? "text-base font-medium text-[var(--score-bad)]"
            : "text-base font-medium"
        }
      >
        {value}
      </div>
    </div>
  );
}

function IterationCard({ version, runs }: { version: PromptVersion; runs: Run[] }) {
  const trainRuns = runs.filter((r) => r.split === "train");
  const holdoutRuns = runs.filter((r) => r.split === "holdout");

  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="rounded-md bg-primary/10 px-2 py-0.5 font-mono text-[11px] tracking-tight text-primary">
            v{version.iteration}
          </span>
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            {version.source}
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs">
          {trainRuns.length > 0 && (
            <span className="text-muted-foreground">
              train: <span className="text-foreground">{avgScore(trainRuns)}</span>
            </span>
          )}
          {holdoutRuns.length > 0 && (
            <span className="text-muted-foreground">
              holdout: <span className="text-foreground">{avgScore(holdoutRuns)}</span>
            </span>
          )}
        </div>
      </div>
      {version.rationale && (
        <p className="mb-3 text-xs text-muted-foreground">{version.rationale}</p>
      )}
      {version.diff?.edits ? (
        <PromptDiff
          current={version.content}
          edits={version.diff.edits}
          applied={version.diff.applied}
          skipped={version.diff.skipped}
        />
      ) : (
        <CodeBlock content={version.content} label="prompt" />
      )}
      {runs.length > 0 && (
        <div className="mt-4 grid gap-2 md:grid-cols-2">
          {runs.map((r) => (
            <div
              key={r.id}
              className="flex items-center justify-between rounded-md border border-border bg-card/30 px-3 py-2 text-xs"
            >
              <div>
                <span className="font-mono text-muted-foreground">{r.split}</span>{" "}
                <span className="font-mono">{r.target_model.split("/")[1] ?? r.target_model}</span>
              </div>
              <div className="font-mono">
                {r.mean_score != null ? formatScore(r.mean_score) : "—"}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function avgScore(runs: Run[]): string {
  const scores = runs.filter((r) => r.mean_score != null).map((r) => r.mean_score as number);
  if (scores.length === 0) return "—";
  return formatScore(scores.reduce((a, b) => a + b, 0) / scores.length);
}

function buildTrajectory(
  iterations: IterationStats["iterations"] | undefined,
  targetModels: string[],
): TrajectoryPoint[] {
  if (!iterations) return [];
  return iterations
    .slice()
    .sort((a, b) => a.iteration - b.iteration)
    .map((it) => {
      const models: TrajectoryPoint["models"] = {};
      for (const m of targetModels) {
        const ms = it.by_model[m];
        if (!ms) continue;
        models[m] = {
          train: ms.train.mean,
          holdout: ms.holdout.mean,
          train_ci: ms.train.ci_half_width ?? null,
          holdout_ci: ms.holdout.ci_half_width ?? null,
          train_n: ms.train.n,
          holdout_n: ms.holdout.n,
        };
      }
      return {
        iteration: it.iteration,
        models,
        train: it.train.mean,
        holdout: it.holdout.mean,
        train_ci: it.train.ci_half_width ?? null,
        holdout_ci: it.holdout.ci_half_width ?? null,
        train_n: it.train.n,
        holdout_n: it.holdout.n,
      };
    });
}

function buildGap(trajectory: TrajectoryPoint[], targetModels: string[]): GapPoint[] {
  return trajectory
    .filter((p) => p.train != null && p.holdout != null)
    .map((p) => {
      const by_model: Record<string, number | null> = {};
      for (const m of targetModels) {
        const ms = p.models[m];
        if (ms && ms.train != null && ms.holdout != null) {
          by_model[m] = ms.train - ms.holdout;
        } else {
          by_model[m] = null;
        }
      }
      return {
        iteration: p.iteration,
        gap: (p.train as number) - (p.holdout as number),
        by_model,
      };
    });
}

/**
 * Lower-confidence-bound per model: argmax_iter (mean - 1.96 * SE).
 * Returns model → best holdout mean for the iteration with the highest LCB.
 * Prefers robust prompts over lucky high-variance ones.
 */
function bestHoldoutPerModel(stats: IterationStats | undefined): Record<string, number | null> {
  if (!stats) return {};
  const out: Record<string, number | null> = {};
  for (const m of stats.target_models) {
    let bestLcb: number | null = null;
    let bestMean: number | null = null;
    for (const it of stats.iterations) {
      const ms = it.by_model[m];
      if (!ms) continue;
      const h = ms.holdout;
      if (h.mean == null || h.n < 1) continue;
      const se = h.se ?? 0;
      const lcb = h.mean - 1.96 * se;
      if (bestLcb == null || lcb > bestLcb) {
        bestLcb = lcb;
        bestMean = h.mean;
      }
    }
    out[m] = bestMean;
  }
  return out;
}

function bestHoldoutAggregated(stats: IterationStats | undefined): number | null {
  if (!stats) return null;
  let bestLcb: number | null = null;
  let bestMean: number | null = null;
  for (const it of stats.iterations) {
    const h = it.holdout;
    if (h.mean == null || h.n < 1) continue;
    const se = h.se ?? 0;
    const lcb = h.mean - 1.96 * se;
    if (bestLcb == null || lcb > bestLcb) {
      bestLcb = lcb;
      bestMean = h.mean;
    }
  }
  return bestMean;
}

function shortModel(model: string): string {
  const lastSlash = model.lastIndexOf("/");
  return lastSlash >= 0 ? model.slice(lastSlash + 1) : model;
}
