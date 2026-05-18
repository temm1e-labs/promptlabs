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

function flatten(data: TrajectoryPoint[], models: string[]): FlatRow[] {
  return data.map((p) => {
    const row: FlatRow = { iteration: p.iteration };
    for (const m of models) {
      const s = p.models[m];
      row[`train::${m}`] = s?.train ?? null;
      row[`holdout::${m}`] = s?.holdout ?? null;
      row[`train_n::${m}`] = s?.train_n ?? undefined;
      row[`holdout_n::${m}`] = s?.holdout_n ?? undefined;
    }
    // Aggregated keys for the single-model rendering path
    row["train"] = p.train;
    row["holdout"] = p.holdout;
    if (p.train != null && p.train_ci != null) {
      row["train_band"] = [
        Math.max(0, p.train - p.train_ci),
        Math.min(1, p.train + p.train_ci),
      ];
    } else {
      row["train_band"] = null;
    }
    if (p.holdout != null && p.holdout_ci != null) {
      row["holdout_band"] = [
        Math.max(0, p.holdout - p.holdout_ci),
        Math.min(1, p.holdout + p.holdout_ci),
      ];
    } else {
      row["holdout_band"] = null;
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
  const flat = flatten(data, models);

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

          {/* SINGLE-MODEL PATH — preserves the existing aesthetic exactly. */}
          {!multi && (
            <>
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
            </>
          )}

          {/* MULTI-MODEL PATH — one solid (train) + one dashed (holdout) per model. */}
          {multi &&
            models.map((m, idx) => {
              const color = colorFor(idx);
              return (
                <Line
                  key={`train::${m}`}
                  type="monotone"
                  dataKey={`train::${m}`}
                  stroke={color}
                  strokeWidth={1.5}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                  name={`train::${m}`}
                  connectNulls
                  isAnimationActive={false}
                />
              );
            })}
          {multi &&
            models.map((m, idx) => {
              const color = colorFor(idx);
              return (
                <Line
                  key={`holdout::${m}`}
                  type="monotone"
                  dataKey={`holdout::${m}`}
                  stroke={color}
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                  name={`holdout::${m}`}
                  connectNulls
                  isAnimationActive={false}
                />
              );
            })}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
