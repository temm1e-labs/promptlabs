// Hand-written types for the v0 API surface. Will be replaced by
// openapi-typescript-generated types in `generated.ts` once stable.

export type Project = {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  experiment_count: number;
};

export type PromptVariable = {
  name: string;
  description: string;
  example_value: string;
};

export type RubricCriterion = {
  name: string;
  definition: string;
  weight: number;
  objective: string | null;
};

export type AgentConfig = {
  writer_model?: string | null;
  evalgen_model?: string | null;
  judge_model?: string | null;
  optimizer_model?: string | null;
  diversity_judge_model?: string | null;
};

export type ExperimentCreate = {
  name: string;
  mode: "cold" | "warm";
  intent: string;
  requirements?: string | null;
  existing_prompt?: string | null;
  known_issues?: string | null;
  optimization_objectives: OptimizationObjective[];
  target_models: string[];
  agent_config?: AgentConfig;
  budget_usd: number;
  max_iterations: number;
  eval_size: number;
  train_ratio: number;
};

export type Experiment = {
  id: string;
  project_id: string;
  name: string;
  intent: string;
  requirements: string | null;
  known_issues: string | null;
  optimization_objectives: OptimizationObjective[];
  target_models: string[];
  agent_config: AgentConfig;
  budget_usd: number;
  cost_usd: number;
  max_iterations: number;
  eval_size: number;
  train_ratio: number;
  current_iteration: number;
  accepted_iteration: number | null;
  status: ExperimentStatus;
  failure_reason: string | null;
  created_at: string;
  updated_at: string;
};

export type PromptVersion = {
  id: string;
  iteration: number;
  content: string;
  rationale: string | null;
  source: "cold" | "warm" | "optimizer";
  parent_id: string | null;
  diff: {
    edits?: Array<{
      op: string;
      anchor?: string | null;
      new_text?: string | null;
      reason: string;
      targets_criterion?: string | null;
    }>;
    applied?: number;
    skipped?: number;
    skip_reasons?: string[];
  } | null;
  created_at: string;
};

export type Run = {
  id: string;
  iteration: number;
  split: "train" | "holdout";
  target_model: string;
  prompt_version_id: string;
  status: "pending" | "running" | "completed" | "failed";
  mean_score: number | null;
  cost_usd: number;
  created_at: string;
};

export type OptimizationObjective =
  | "accuracy"
  | "cost"
  | "latency"
  | "robustness"
  | "format_adherence"
  | "brevity"
  | "tone";

export type ExperimentStatus =
  | "pending"
  | "running"
  | "paused"
  | "converged"
  | "overfit"
  | "exhausted"
  | "failed"
  | "accepted"
  | "cancelled";

export type Model = {
  id: string;
  label: string;
  provider: string;
  family: string | null;
  is_default: boolean;
};

export type ModelGroup = {
  provider: string;
  models: Model[];
};

export type ExperimentSummary = {
  id: string;
  name: string;
  intent: string;
  status: ExperimentStatus;
  current_iteration: number;
  cost_usd: number;
  budget_usd: number;
  target_models: string[];
  optimization_objectives: OptimizationObjective[];
  best_score: number | null;
};
