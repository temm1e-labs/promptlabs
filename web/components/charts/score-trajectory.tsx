"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type ModelStats = {
  train: number | null;
  holdout: number | null;
  train_ci?: number | null;
  holdout_ci?: number | null;
  train_n?: number;
  holdout_n?: number;
};

export type TrajectoryPoint = {
  iteration: number;
  /** Per-target-model statistics for this iteration. Empty when no model data. */
  models: Record<string, ModelStats>;
  /** Legacy aggregated stats — kept for the single-model rendering path. */
  train: number | null;
  holdout: number | null;
  train_ci?: number | null;
  holdout_ci?: number | null;
  train_n?: number;
  holdout_n?: number;
};

// 6 distinct colors; experiments rarely benchmark more than this.
// First color matches the existing single-model accent so 1-model runs look unchanged.
const MODEL_COLORS = [
  "var(--primary)",
  "#a78bfa", // violet
  "#f59e0b", // amber
  "#10b981", // emerald
  "#ec4899", // pink
  "#22d3ee", // sky
];

function shortModelName(model: string): string {
  // gemini/gemini-3-flash-preview → gemini-3-flash-preview
  const lastSlash = model.lastIndexOf("/");
  return lastSlash >= 0 ? model.slice(lastSlash + 1) : model;
}

function colorFor(idx: number): string {
  return MODEL_COLORS[idx % MODEL_COLORS.length] ?? "var(--primary)";
}

// Per-row flat shape sent to Recharts. Bands are [lo, hi] tuples (Recharts
// renders Areas with a tuple dataKey as a filled envelope between the two).
type FlatRow = Record<string, number | [number, number] | null | undefined>;

// Skip the CI band when it would be a degenerate zero-width range — Recharts
// can handle [1.0, 1.0] inconsistently in a ComposedChart and that has been
// observed to suppress the Line render that sits on top of it.
function bandOrNull(value: number | null, ci: number | null | undefined): [number, number] | null {
  if (value == null || ci == null || ci <= 0) return null;
  return [Math.max(0, value - ci), Math.min(1, value + ci)];
}

function flattenSingle(data: TrajectoryPoint[]): FlatRow[] {
  return data.map((p) => ({
    iteration: p.iteration,
    train: p.train,
    holdout: p.holdout,
    train_n: p.train_n,
    holdout_n: p.holdout_n,
    train_band: bandOrNull(p.train, p.train_ci),
    holdout_band: bandOrNull(p.holdout, p.holdout_ci),
  }));
}

function flattenMulti(data: TrajectoryPoint[], models: string[]): FlatRow[] {
  return data.map((p) => {
    const row: FlatRow = { iteration: p.iteration };
    for (const m of models) {
      const s = p.models[m];
      row[`train::${m}`] = s?.train ?? null;
      row[`holdout::${m}`] = s?.holdout ?? null;
      row[`train_n::${m}`] = s?.train_n ?? undefined;
      row[`holdout_n::${m}`] = s?.holdout_n ?? undefined;
    }
    return row;
  });
}

function formatTooltipLabel(value: unknown, name: string, payload: FlatRow): [string, string] {
  // Single-model CI band: value arrives as [lo, hi]
  if (
    (name === "train_band" || name === "holdout_band") &&
    Array.isArray(value) &&
    value.length === 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number"
  ) {
    const lo = value[0] as number;
    const hi = value[1] as number;
    return [
      `${(lo * 100).toFixed(1)}% – ${(hi * 100).toFixed(1)}%`,
      `${name.replace("_band", "")} 95% CI`,
    ];
  }
  if (typeof value !== "number") return ["—", name];
  const parts = name.split("::");
  const kind = parts[0] ?? name;
  const model = parts[1];
  const nLookup =
    kind === "train" && model
      ? `train_n::${model}`
      : kind === "holdout" && model
      ? `holdout_n::${model}`
      : null;
  const n = nLookup ? (payload[nLookup] as number | undefined) : undefined;
  const suffix = n ? `  (N=${n})` : "";
  const label = model ? `${kind} · ${shortModelName(model)}` : kind;
  return [`${(value * 100).toFixed(1)}%${suffix}`, label];
}

export function ScoreTrajectory({
  data,
  models = [],
}: {
  data: TrajectoryPoint[];
  /** Target models to render. Empty or single → aggregated single-line layout. */
  models?: string[];
}) {
  const multi = models.length > 1;
  const flat = multi ? flattenMulti(data, models) : flattenSingle(data);

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={flat} margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="iteration"
            stroke="var(--muted-foreground)"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `v${v}`}
          />
          <YAxis
            domain={[0, 1]}
            stroke="var(--muted-foreground)"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            padding={{ top: 8, bottom: 4 }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelFormatter={(label) => `iteration v${label}`}
            formatter={(value, name, item) =>
              formatTooltipLabel(value, String(name), (item?.payload ?? {}) as FlatRow)
            }
          />
          {multi && (
            <Legend
              wrapperStyle={{ fontSize: 10, paddingTop: 8 }}
              formatter={(value) => {
                const [kind, model] = String(value).split("::");
                return model ? `${kind} · ${shortModelName(model)}` : kind;
              }}
            />
          )}

          {/*
            CRITICAL: Recharts 2.x uses React.Children.forEach to scan for Line/
            Area/Bar children. That iteration does NOT recurse into Fragments.
            Wrapping these in <>...</> made them invisible to Recharts and the
            chart silently rendered axes only. Every Area/Line below must be a
            direct child of ComposedChart — no fragments, no helper components.
          */}
          {!multi && (
            <Area
              type="monotone"
              dataKey="train_band"
              stroke="none"
              fill="var(--primary)"
              fillOpacity={0.12}
              connectNulls
              isAnimationActive={false}
              name="train_band"
            />
          )}
          {!multi && (
            <Area
              type="monotone"
              dataKey="holdout_band"
              stroke="none"
              fill="var(--primary)"
              fillOpacity={0.08}
              connectNulls
              isAnimationActive={false}
              name="holdout_band"
            />
          )}
          {!multi && (
            <Line
              type="monotone"
              dataKey="train"
              stroke="var(--primary)"
              strokeWidth={1.5}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              name="train"
              connectNulls
              isAnimationActive={false}
            />
          )}
          {!multi && (
            <Line
              type="monotone"
              dataKey="holdout"
              stroke="var(--primary)"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              name="holdout"
              connectNulls
              isAnimationActive={false}
            />
          )}

          {/* MULTI-MODEL PATH — one solid (train) + one dashed (holdout) per
              model. Arrays are fine as direct children (React.Children.forEach
              iterates arrays). Each Line must have a unique key. */}
          {multi &&
            models.map((m, idx) => (
              <Line
                key={`train::${m}`}
                type="monotone"
                dataKey={`train::${m}`}
                stroke={colorFor(idx)}
                strokeWidth={1.5}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                name={`train::${m}`}
                connectNulls
                isAnimationActive={false}
              />
            ))}
          {multi &&
            models.map((m, idx) => (
              <Line
                key={`holdout::${m}`}
                type="monotone"
                dataKey={`holdout::${m}`}
                stroke={colorFor(idx)}
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                name={`holdout::${m}`}
                connectNulls
                isAnimationActive={false}
              />
            ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
