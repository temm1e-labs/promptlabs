"use client";

import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";

type SSEMessage = {
  type: string;
  timestamp: string;
  [key: string]: unknown;
};

const ICON: Record<string, string> = {
  "loop.started": "▶",
  "loop.finished": "■",
  "loop.failed": "!",
  "writer.completed": "✎",
  "writer.auto_wrapped": "⮊",
  "evalgen.completed": "𝛴",
  "iteration.started": "→",
  "iteration.completed": "✓",
  "run.started": "▸",
  "run.completed": "▪",
  "optimizer.completed": "Δ",
  "optimizer.noop": "·",
};

const TERMINAL_EVENTS = new Set(["loop.finished", "loop.failed"]);
const TRACKED_EVENTS = [
  "loop.started",
  "loop.finished",
  "loop.failed",
  "writer.completed",
  "writer.auto_wrapped",
  "evalgen.completed",
  "iteration.started",
  "iteration.completed",
  "run.started",
  "run.completed",
  "optimizer.completed",
  "optimizer.noop",
];

export function SSERail({ experimentId, apiUrl }: { experimentId: string; apiUrl: string }) {
  const [events, setEvents] = useState<SSEMessage[]>([]);
  const [closed, setClosed] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Reset between experiments so we don't carry events across IDs
    setEvents([]);
    setClosed(false);

    // Keep a SET of dedupe keys so re-replay on accidental reconnect doesn't pile up
    const seen = new Set<string>();

    const url = `${apiUrl}/experiments/${experimentId}/stream`;
    const es = new EventSource(url);
    esRef.current = es;

    const handler = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data) as SSEMessage;
        // Dedupe by (type + timestamp) — the server replays history on reconnect
        const key = `${data.type}::${data.timestamp}`;
        if (seen.has(key)) return;
        seen.add(key);
        setEvents((prev) => [...prev.slice(-200), data]);

        // Terminal event → close the stream so the browser doesn't reconnect & re-replay
        if (TERMINAL_EVENTS.has(data.type)) {
          es.close();
          esRef.current = null;
          setClosed(true);
        }
      } catch {
        // ignore parse errors
      }
    };

    TRACKED_EVENTS.forEach((t) => es.addEventListener(t, handler as EventListener));
    es.onmessage = handler;

    return () => {
      TRACKED_EVENTS.forEach((t) => es.removeEventListener(t, handler as EventListener));
      es.close();
      esRef.current = null;
    };
  }, [experimentId, apiUrl]);

  useEffect(() => {
    containerRef.current?.scrollTo({
      top: containerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [events]);

  return (
    <aside className="hidden h-full w-72 shrink-0 border-l border-border bg-card/30 lg:flex lg:flex-col">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          {closed ? "stream · ended" : "stream · live"}
        </div>
        <div
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            closed ? "bg-muted-foreground/40" : "animate-pulse bg-[var(--score-good)]",
          )}
        />
      </div>
      <div ref={containerRef} className="flex-1 overflow-y-auto px-3 py-2 font-mono text-[11px]">
        {events.length === 0 ? (
          <div className="px-2 py-1 text-muted-foreground/60">
            waiting for events…
          </div>
        ) : (
          events.map((e, i) => (
            <EventLine key={i} event={e} />
          ))
        )}
      </div>
    </aside>
  );
}

function EventLine({ event }: { event: SSEMessage }) {
  const t = new Date(event.timestamp).toLocaleTimeString([], { hour12: false });
  const icon = ICON[event.type] ?? "·";
  const color =
    event.type === "loop.failed"
      ? "text-[var(--score-bad)]"
      : event.type === "loop.finished" || event.type === "iteration.completed"
      ? "text-[var(--score-good)]"
      : "text-foreground";
  const detail = summarize(event);
  return (
    <div className="grid grid-cols-[auto_auto_1fr] items-baseline gap-2 px-2 py-1">
      <span className="text-muted-foreground/60">{t}</span>
      <span className={cn("w-3 text-center", color)}>{icon}</span>
      <span className="truncate text-muted-foreground" title={JSON.stringify(event)}>
        <span className={cn("mr-1", color)}>{event.type}</span>
        {detail}
      </span>
    </div>
  );
}

function summarize(e: SSEMessage): string {
  if (e.type === "iteration.completed") {
    return `train ${pct(e.train_mean)} · holdout ${pct(e.holdout_mean)} · $${num(e.cost_so_far)}`;
  }
  if (e.type === "run.completed") {
    return `${e.split} · ${e.target_model} · ${pct(e.mean_score)}`;
  }
  if (e.type === "optimizer.completed") {
    return `+${e.edits_applied} edits, -${e.edits_skipped} skipped`;
  }
  if (e.type === "evalgen.completed") {
    return `${e.n_train}+${e.n_holdout} items`;
  }
  return "";
}

function pct(v: unknown): string {
  if (typeof v !== "number") return "—";
  return (v * 100).toFixed(1) + "%";
}

function num(v: unknown): string {
  if (typeof v !== "number") return "—";
  return v.toFixed(3);
}
