import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { api } from "./client";
import type {
  Experiment,
  ExperimentCreate,
  ExperimentSummary,
  ModelGroup,
  Project,
  PromptVersion,
  Run,
} from "./types";

export type CrossProjectExperiment = ExperimentSummary & {
  project_id: string;
  project_name: string;
  created_at: string;
};

export type SettingsStatus = {
  default_model: string;
  providers: Record<string, boolean>;
  cache_ttl_days: number;
  max_concurrent_requests: number;
  request_timeout_s: number;
  defaults: {
    eval_size: number;
    train_ratio: number;
    max_iterations: number;
    budget_usd: number;
  };
};

export type IterationStat = {
  mean: number | null;
  std: number | null;
  n: number;
  se: number | null;
  ci_half_width: number | null;
};

export type IterationModelStats = {
  train: IterationStat;
  holdout: IterationStat;
};

export type IterationStats = {
  /** Sorted list of target_models that appeared in any Run for this experiment. */
  target_models: string[];
  iterations: Array<{
    iteration: number;
    prompt_version_id: string | null;
    /** Aggregated across all target models (legacy shape; preserved for compat). */
    train: IterationStat;
    holdout: IterationStat;
    /** Per-target-model breakdown. Only models with data for this iteration appear. */
    by_model: Record<string, IterationModelStats>;
  }>;
};

export type BestPromptPerModel = Record<
  string,
  {
    iteration: number | null;
    mean: number | null;
    se: number | null;
    lcb: number | null;
    version_id?: string;
  }
>;

export function useProjects() {
  return useQuery({
    queryKey: ["projects"],
    queryFn: () => api.get<Project[]>("/projects"),
  });
}

export function useProject(id: string) {
  return useQuery({
    queryKey: ["projects", id],
    queryFn: () => api.get<Project>(`/projects/${id}`),
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: { name: string; description?: string | null }) =>
      api.post<Project>("/projects", input),
    onSuccess: (p) => {
      toast.success(`Created project "${p.name}"`);
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
    onError: (e) => toast.error(`Could not create project: ${(e as Error).message}`),
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.delete<void>(`/projects/${id}`),
    onSuccess: () => {
      toast.success("Project deleted");
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
    onError: (e) => toast.error(`Could not delete: ${(e as Error).message}`),
  });
}

export function useExperiments(projectId: string) {
  return useQuery({
    queryKey: ["projects", projectId, "experiments"],
    queryFn: () => api.get<ExperimentSummary[]>(`/projects/${projectId}/experiments`),
  });
}

export function useExperiment(id: string) {
  return useQuery({
    queryKey: ["experiments", id],
    queryFn: () => api.get<Experiment>(`/experiments/${id}`),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      // Keep polling while running, pending, or paused (paused can resume → running).
      const liveStatuses = ["running", "pending", "paused"];
      return liveStatuses.includes(data.status) ? 2000 : false;
    },
  });
}

export function useCreateExperiment(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ExperimentCreate) =>
      api.post<Experiment>(`/projects/${projectId}/experiments`, body),
    onSuccess: () => {
      toast.success("Experiment started");
      qc.invalidateQueries({ queryKey: ["projects", projectId, "experiments"] });
      qc.invalidateQueries({ queryKey: ["experiments", "all"] });
    },
    onError: (e) => toast.error(`Could not start experiment: ${(e as Error).message}`),
  });
}

export function useCancelExperiment(experimentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<Experiment>(`/experiments/${experimentId}/cancel`),
    onSuccess: () => {
      toast.success("Experiment cancelled — current step will finish then stop");
      qc.invalidateQueries({ queryKey: ["experiments", experimentId] });
    },
    onError: (e) => toast.error(`Could not cancel: ${(e as Error).message}`),
  });
}

export function usePauseExperiment(experimentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<Experiment>(`/experiments/${experimentId}/pause`),
    onSuccess: () => {
      toast.success("Experiment paused — current step will finish then pause");
      qc.invalidateQueries({ queryKey: ["experiments", experimentId] });
    },
    onError: (e) => toast.error(`Could not pause: ${(e as Error).message}`),
  });
}

export function useResumeExperiment(experimentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<Experiment>(`/experiments/${experimentId}/resume`),
    onSuccess: () => {
      toast.success("Experiment resumed");
      qc.invalidateQueries({ queryKey: ["experiments", experimentId] });
    },
    onError: (e) => toast.error(`Could not resume: ${(e as Error).message}`),
  });
}

export function useCloneExperiment(experimentId: string, projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<Experiment>(`/experiments/${experimentId}/clone`),
    onSuccess: (cloned) => {
      toast.success(`Restarted as ${cloned.name}`);
      qc.invalidateQueries({ queryKey: ["projects", projectId, "experiments"] });
      qc.invalidateQueries({ queryKey: ["experiments", "all"] });
    },
    onError: (e) => toast.error(`Could not restart: ${(e as Error).message}`),
  });
}

export function useDeleteExperiment(experimentId: string, projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.delete<void>(`/experiments/${experimentId}`),
    onSuccess: () => {
      toast.success("Experiment deleted");
      qc.invalidateQueries({ queryKey: ["projects", projectId, "experiments"] });
      qc.invalidateQueries({ queryKey: ["experiments", experimentId] });
      qc.invalidateQueries({ queryKey: ["experiments", "all"] });
    },
    onError: (e) => toast.error(`Could not delete: ${(e as Error).message}`),
  });
}

export function useAllExperiments() {
  return useQuery({
    queryKey: ["experiments", "all"],
    queryFn: () => api.get<CrossProjectExperiment[]>("/experiments"),
    refetchInterval: 3000,
  });
}

export function useSettingsStatus() {
  return useQuery({
    queryKey: ["settings", "status"],
    queryFn: () => api.get<SettingsStatus>("/settings/status"),
    staleTime: 60_000,
  });
}

export function useClearCache() {
  return useMutation({
    mutationFn: () => api.post<{ cleared: number }>("/settings/cache/clear"),
    onSuccess: (r) => toast.success(`Cleared ${r.cleared} cached responses`),
    onError: (e) => toast.error(`Could not clear cache: ${(e as Error).message}`),
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["models", "grouped"],
    queryFn: () => api.get<ModelGroup[]>("/models/grouped"),
    staleTime: 1000 * 60 * 10,
  });
}

export function usePromptVersions(experimentId: string) {
  return useQuery({
    queryKey: ["experiments", experimentId, "prompt-versions"],
    queryFn: () => api.get<PromptVersion[]>(`/experiments/${experimentId}/prompt-versions`),
    refetchInterval: 2000,
  });
}

export function useRuns(experimentId: string) {
  return useQuery({
    queryKey: ["experiments", experimentId, "runs"],
    queryFn: () => api.get<Run[]>(`/experiments/${experimentId}/runs`),
    refetchInterval: 2000,
  });
}

export function useIterationStats(experimentId: string) {
  return useQuery({
    queryKey: ["experiments", experimentId, "iteration-stats"],
    queryFn: () =>
      api.get<IterationStats>(`/experiments/${experimentId}/iteration-stats`),
    refetchInterval: 2000,
  });
}
