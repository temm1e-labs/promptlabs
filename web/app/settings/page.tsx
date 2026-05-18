"use client";

import { Check, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useClearCache, useSettingsStatus } from "@/lib/api/hooks";
import { cn } from "@/lib/utils";

export default function SettingsPage() {
  const { data: status, isLoading } = useSettingsStatus();
  const clearMut = useClearCache();

  return (
    <div className="mx-auto max-w-3xl">
      <header className="mb-8 border-b border-border pb-6">
        <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          workspace
        </p>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          API key status, defaults, and provider cache. Edit{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">.env</code> at the
          repo root, then restart the API to apply changes.
        </p>
      </header>

      <div className="space-y-6">
        <Card className="p-6">
          <div className="mb-4 flex items-baseline justify-between">
            <div>
              <h2 className="text-sm font-medium tracking-tight">Provider API keys</h2>
              <p className="text-xs text-muted-foreground">
                Detected from environment variables.
              </p>
            </div>
          </div>
          {isLoading ? (
            <div className="h-32 animate-pulse rounded-md bg-muted/30" />
          ) : (
            <div className="grid gap-2 md:grid-cols-2">
              {status &&
                Object.entries(status.providers).map(([name, present]) => (
                  <div
                    key={name}
                    className={cn(
                      "flex items-center justify-between rounded-md border px-3 py-2",
                      present ? "border-border" : "border-dashed border-border bg-muted/30",
                    )}
                  >
                    <div className="font-mono text-xs uppercase tracking-widest">
                      {name.replace("_", " ")}
                    </div>
                    {present ? (
                      <Check className="h-3.5 w-3.5 text-[var(--score-good)]" />
                    ) : (
                      <X className="h-3.5 w-3.5 text-muted-foreground/60" />
                    )}
                  </div>
                ))}
            </div>
          )}
        </Card>

        {status && (
          <Card className="p-6">
            <h2 className="mb-4 text-sm font-medium tracking-tight">Defaults</h2>
            <dl className="grid gap-3 text-xs md:grid-cols-2">
              <Row label="default model" value={status.default_model} mono />
              <Row label="default budget" value={`$${status.defaults.budget_usd}`} />
              <Row label="default max iterations" value={status.defaults.max_iterations} />
              <Row label="default eval size" value={status.defaults.eval_size} />
              <Row label="default train ratio" value={status.defaults.train_ratio} />
              <Row label="concurrent requests" value={status.max_concurrent_requests} />
              <Row label="request timeout" value={`${status.request_timeout_s}s`} />
              <Row
                label="cache ttl"
                value={`${status.cache_ttl_days.toFixed(0)} days`}
              />
            </dl>
          </Card>
        )}

        <Card className="p-6">
          <div className="mb-4 flex items-baseline justify-between">
            <div>
              <h2 className="text-sm font-medium tracking-tight">Provider cache</h2>
              <p className="text-xs text-muted-foreground">
                LLM responses are content-addressed and cached locally. Clear to force fresh
                calls on the next run.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => clearMut.mutate()}
              disabled={clearMut.isPending}
            >
              {clearMut.isPending ? "Clearing…" : "Clear cache"}
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string | number;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between rounded-md border border-border bg-card/50 px-3 py-2">
      <dt className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </dt>
      <dd className={cn("text-xs", mono && "font-mono")}>{value}</dd>
    </div>
  );
}
