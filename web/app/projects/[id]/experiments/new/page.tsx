"use client";

import { useParams, useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { AgentConfigPicker } from "@/components/experiments/agent-config-picker";
import { ModelPicker } from "@/components/experiments/model-picker";
import { ObjectivePicker } from "@/components/experiments/objective-picker";
import { PRESETS, PresetPicker, type RunPreset } from "@/components/experiments/preset-picker";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useCreateExperiment, useModels } from "@/lib/api/hooks";
import { cn } from "@/lib/utils";
import type { AgentConfig, OptimizationObjective } from "@/lib/api/types";

export default function NewExperimentPage() {
  const { id: projectId } = useParams<{ id: string }>();
  const router = useRouter();
  const mut = useCreateExperiment(projectId);
  const { data: groups } = useModels();

  const [mode, setMode] = useState<"cold" | "warm">("cold");
  const [name, setName] = useState("");
  const [intent, setIntent] = useState("");
  const [requirements, setRequirements] = useState("");
  const [existingPrompt, setExistingPrompt] = useState("");
  const [knownIssues, setKnownIssues] = useState("");
  const [objectives, setObjectives] = useState<OptimizationObjective[]>(["accuracy"]);
  const [targets, setTargets] = useState<string[]>([]);
  const [labDefault, setLabDefault] = useState("");
  const [agentConfig, setAgentConfig] = useState<AgentConfig>({});

  const [presetId, setPresetId] = useState<RunPreset["id"]>("standard");
  const preset = useMemo(() => PRESETS.find((p) => p.id === presetId), [presetId]);
  const [customBudget, setCustomBudget] = useState(10);
  const [customMaxIter, setCustomMaxIter] = useState(10);
  const [customEvalSize, setCustomEvalSize] = useState(50);

  // Default the lab model to first provider's default
  useMemo(() => {
    if (!labDefault && groups) {
      const defaultModel = groups.flatMap((g) => g.models).find((m) => m.is_default);
      if (defaultModel) setLabDefault(defaultModel.id);
    }
  }, [groups, labDefault]);

  const budget_usd = preset ? preset.budget_usd : customBudget;
  const max_iterations = preset ? preset.max_iterations : customMaxIter;
  const eval_size = preset ? preset.eval_size : customEvalSize;

  const canSubmit =
    name.trim() &&
    intent.trim() &&
    targets.length > 0 &&
    objectives.length > 0 &&
    labDefault.length > 0 &&
    (mode === "cold" || existingPrompt.trim().length > 0);

  const submit = async () => {
    if (!canSubmit) return;
    const finalAgentConfig: AgentConfig = {
      writer_model: agentConfig.writer_model || labDefault,
      evalgen_model: agentConfig.evalgen_model || labDefault,
      judge_model: agentConfig.judge_model || labDefault,
      optimizer_model: agentConfig.optimizer_model || labDefault,
    };
    const exp = await mut.mutateAsync({
      name: name.trim(),
      mode,
      intent: intent.trim(),
      requirements: requirements.trim() || null,
      existing_prompt: mode === "warm" ? existingPrompt : null,
      known_issues: mode === "warm" ? knownIssues.trim() || null : null,
      optimization_objectives: objectives,
      target_models: targets,
      agent_config: finalAgentConfig,
      budget_usd,
      max_iterations,
      eval_size,
      train_ratio: 0.7,
    });
    router.push(`/projects/${projectId}/experiments/${exp.id}`);
  };

  return (
    <div className="mx-auto max-w-3xl">
      <header className="mb-8 border-b border-border pb-6">
        <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          new experiment
        </p>
        <h1 className="text-2xl font-semibold tracking-tight">Configure the loop</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Cold start from intent, or warm start from an existing prompt. Pick what to optimize for,
          which models to test against, and how aggressively the loop should run.
        </p>
      </header>

      <div className="space-y-6">
        <Card className="p-6">
          <Label className="mb-3 block">1. Mode</Label>
          <div className="grid grid-cols-2 gap-3">
            {(["cold", "warm"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  "rounded-md border px-4 py-3 text-left transition-colors",
                  mode === m
                    ? "border-primary/50 bg-primary/5"
                    : "border-border bg-card hover:border-border/80",
                )}
              >
                <div className="text-sm font-medium tracking-tight">
                  {m === "cold" ? "Cold start" : "Warm start"}
                </div>
                <div className="mt-0.5 text-xs text-muted-foreground">
                  {m === "cold"
                    ? "From an intent. Writer drafts v0."
                    : "From an existing prompt. Skip to optimization."}
                </div>
              </button>
            ))}
          </div>
        </Card>

        <Card className="p-6 space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="exp-name">Experiment name</Label>
            <Input
              id="exp-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. iteration-1"
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="intent">2. Intent — what should the prompt do?</Label>
            <Textarea
              id="intent"
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              placeholder="e.g. Classify customer support emails into billing / technical / account / other."
              className="min-h-[80px]"
            />
          </div>

          {mode === "cold" ? (
            <div className="space-y-1.5">
              <Label htmlFor="requirements">Requirements (optional)</Label>
              <Textarea
                id="requirements"
                value={requirements}
                onChange={(e) => setRequirements(e.target.value)}
                placeholder="Constraints, format, audience, tone, do's and don'ts."
              />
            </div>
          ) : (
            <>
              <div className="space-y-1.5">
                <Label htmlFor="existing">Existing prompt</Label>
                <Textarea
                  id="existing"
                  value={existingPrompt}
                  onChange={(e) => setExistingPrompt(e.target.value)}
                  placeholder="Paste the prompt verbatim. Use {{variable_name}} for inputs that vary."
                  className="min-h-[160px] font-mono text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="issues">What&apos;s wrong with it? (seeds the optimizer)</Label>
                <Textarea
                  id="issues"
                  value={knownIssues}
                  onChange={(e) => setKnownIssues(e.target.value)}
                  placeholder="e.g. It hallucinates a category sometimes. Misclassifies multilingual inputs."
                />
              </div>
            </>
          )}
        </Card>

        <Card className="p-6">
          <Label className="mb-3 block">3. Optimize for</Label>
          <ObjectivePicker selected={objectives} onChange={setObjectives} />
        </Card>

        <Card className="p-6 space-y-5">
          <div>
            <Label className="mb-3 block">4. Target models — what we test against</Label>
            <ModelPicker selected={targets} onChange={setTargets} label="" />
          </div>

          <div className="border-t border-border pt-5">
            <Label className="mb-3 block">5. Agent models — what drives the loop</Label>
            <AgentConfigPicker
              labDefault={labDefault}
              onLabDefault={setLabDefault}
              config={agentConfig}
              onConfig={setAgentConfig}
            />
          </div>
        </Card>

        <Card className="p-6">
          <Label className="mb-3 block">6. Run preset</Label>
          <PresetPicker selectedId={presetId} onSelect={setPresetId} />
          {presetId === "custom" && (
            <div className="mt-4 grid gap-4 border-t border-border pt-4 md:grid-cols-3">
              <div className="space-y-1.5">
                <Label htmlFor="budget">Cost cap (USD)</Label>
                <Input
                  id="budget"
                  type="number"
                  min={0.1}
                  step={0.1}
                  value={customBudget}
                  onChange={(e) => setCustomBudget(Number(e.target.value))}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="maxiter">Max iterations</Label>
                <Input
                  id="maxiter"
                  type="number"
                  min={1}
                  max={50}
                  value={customMaxIter}
                  onChange={(e) => setCustomMaxIter(Number(e.target.value))}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="evalsize">Eval set size</Label>
                <Input
                  id="evalsize"
                  type="number"
                  min={4}
                  max={200}
                  value={customEvalSize}
                  onChange={(e) => setCustomEvalSize(Number(e.target.value))}
                />
              </div>
            </div>
          )}
        </Card>

        <div className="flex items-center justify-between gap-3">
          <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            <span>budget {budget_usd}$ · max {max_iterations} iter · {eval_size} items</span>
            {targets.length > 0 && (
              <span> · {targets.length} target model{targets.length === 1 ? "" : "s"}</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={() => router.back()}>
              Cancel
            </Button>
            <Button onClick={submit} disabled={!canSubmit || mut.isPending}>
              {mut.isPending ? "Starting..." : "Start loop"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
