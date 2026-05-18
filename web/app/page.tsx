"use client";

import Link from "next/link";
import { FolderKanban } from "lucide-react";

import { NewProjectDialog } from "@/components/projects/new-project-dialog";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { useProjects } from "@/lib/api/hooks";

export default function HomePage() {
  const { data: projects, isLoading } = useProjects();

  return (
    <div className="mx-auto max-w-6xl">
      <header className="mb-10 flex items-end justify-between border-b border-border pb-6">
        <div>
          <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            workspace
          </p>
          <h1 className="text-2xl font-semibold tracking-tight">Projects</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            The closed prompt-engineering loop. Write, evaluate, surgically rewrite.
          </p>
        </div>
        <NewProjectDialog />
      </header>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Card key={i} className="h-32 animate-pulse" />
          ))}
        </div>
      ) : projects && projects.length === 0 ? (
        <section className="grid place-items-center rounded-lg border border-dashed border-border py-24">
          <div className="text-center">
            <Badge variant="outline" className="mb-4">
              no projects yet
            </Badge>
            <p className="text-sm text-muted-foreground">
              Create a project to start your first experiment.
            </p>
          </div>
        </section>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects?.map((p) => (
            <Link key={p.id} href={`/projects/${p.id}`}>
              <Card className="h-32 cursor-pointer p-5 transition-colors hover:border-primary/40">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <FolderKanban className="h-3.5 w-3.5" />
                  <span className="font-mono text-[10px] uppercase tracking-widest">
                    project
                  </span>
                </div>
                <h2 className="mt-3 text-sm font-medium tracking-tight">{p.name}</h2>
                <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                  {p.description ?? "No description"}
                </p>
                <p className="mt-auto pt-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/70">
                  {p.experiment_count} experiment{p.experiment_count === 1 ? "" : "s"}
                </p>
              </Card>
            </Link>
          ))}
        </div>
      )}

      <footer className="mt-10 flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-muted-foreground/60">
        <span>aesthetic science</span>
        <span>localhost · sqlite</span>
      </footer>
    </div>
  );
}
