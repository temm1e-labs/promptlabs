import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

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
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"] }),
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.delete<void>(`/projects/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"] }),
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
      return data.status === "running" || data.status === "pending" ? 2000 : false;
    },
  });
}

export function useCreateExperiment(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ExperimentCreate) =>
      api.post<Experiment>(`/projects/${projectId}/experiments`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects", projectId, "experiments"] }),
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
