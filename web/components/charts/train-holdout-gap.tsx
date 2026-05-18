"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type GapPoint = {
  iteration: number;
  gap: number; // train - holdout (aggregated, legacy)
  /** Per-target-model gap. Optional; when present and `models` is passed in,
   *  the chart renders one line per model instead of the aggregated area. */
  by_model?: Record<string, number | null>;
};

const MODEL_COLORS = [
  "var(--primary)",
  "#a78bfa",
  "#f59e0b",
  "#10b981",
  "#ec4899",
  "#22d3ee",
];

function shortModelName(model: string): string {
  const lastSlash = model.lastIndexOf("/");
  return lastSlash >= 0 ? model.slice(lastSlash + 1) : model;
}

function colorFor(idx: number): string {
  return MODEL_COLORS[idx % MODEL_COLORS.length] ?? "var(--primary)";
}

export function TrainHoldoutGap({
  data,
  models = [],
}: {
  data: GapPoint[];
  models?: string[];
}) {
  const multi = models.length > 1;
  // Flatten per-model gaps into top-level keys for Recharts
  const flat = data.map((p) => {
    const row: Record<string, number | null> = { iteration: p.iteration, gap: p.gap };
    if (p.by_model) {
      for (const [m, v] of Object.entries(p.by_model)) {
        row[`gap::${m}`] = v;
      }
    }
    return row;
  });

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        {multi ? (
          <LineChart data={flat} margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
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
              stroke="var(--muted-foreground)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}pp`}
              domain={["auto", "auto"]}
            />
            <ReferenceLine y={0.1} stroke="var(--score-bad)" strokeDasharray="2 4" />
            <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeOpacity={0.4} />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number, name: string) => {
                const [, model] = String(name).split("::");
                const label = model ? shortModelName(model) : "gap";
                return [`${(value * 100).toFixed(1)} pp`, label];
              }}
              labelFormatter={(label) => `iteration v${label}`}
            />
            <Legend
              wrapperStyle={{ fontSize: 10, paddingTop: 8 }}
              formatter={(value) => {
                const [, model] = String(value).split("::");
                return model ? shortModelName(model) : "gap";
              }}
            />
            {models.map((m, idx) => (
              <Line
                key={m}
                type="monotone"
                dataKey={`gap::${m}`}
                stroke={colorFor(idx)}
                strokeWidth={1.5}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                connectNulls
                isAnimationActive={false}
                name={`gap::${m}`}
              />
            ))}
          </LineChart>
        ) : (
          <AreaChart data={flat} margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
            <defs>
              <linearGradient id="gapFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--score-bad)" stopOpacity={0.4} />
                <stop offset="60%" stopColor="var(--score-bad)" stopOpacity={0.05} />
                <stop offset="100%" stopColor="var(--score-bad)" stopOpacity={0} />
              </linearGradient>
            </defs>
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
              stroke="var(--muted-foreground)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}pp`}
              domain={["auto", "auto"]}
            />
            <ReferenceLine y={0.1} stroke="var(--score-bad)" strokeDasharray="2 4" />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number) => `${(value * 100).toFixed(1)} pp`}
              labelFormatter={(label) => `iteration v${label}`}
            />
            <Area
              type="monotone"
              dataKey="gap"
              stroke="var(--score-bad)"
              strokeWidth={1.5}
              fill="url(#gapFill)"
            />
          </AreaChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
