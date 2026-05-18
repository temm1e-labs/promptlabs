"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type TrajectoryPoint = {
  iteration: number;
  train: number | null;
  holdout: number | null;
  // 95% CI half-widths — drawn as bands around the line. Null when n < 2.
  train_ci?: number | null;
  holdout_ci?: number | null;
  // sample sizes for the tooltip
  train_n?: number;
  holdout_n?: number;
};

// Recharts can render error bands by adding two extra series whose `Area`
// stacks form the CI envelope. We compute [lower, upper] tuples per point.
type Enriched = TrajectoryPoint & {
  train_band?: [number, number] | null;
  holdout_band?: [number, number] | null;
};

function enrich(data: TrajectoryPoint[]): Enriched[] {
  return data.map((p) => ({
    ...p,
    train_band:
      p.train != null && p.train_ci != null
        ? [Math.max(0, p.train - p.train_ci), Math.min(1, p.train + p.train_ci)]
        : null,
    holdout_band:
      p.holdout != null && p.holdout_ci != null
        ? [Math.max(0, p.holdout - p.holdout_ci), Math.min(1, p.holdout + p.holdout_ci)]
        : null,
  }));
}

export function ScoreTrajectory({ data }: { data: TrajectoryPoint[] }) {
  const enriched = enrich(data);
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={enriched} margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
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
            formatter={(value, name, item) => {
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
                  `${String(name).replace("_band", "")} 95% CI`,
                ];
              }
              if (typeof value === "number") {
                const p = item?.payload as Enriched | undefined;
                const n =
                  name === "train" ? p?.train_n : name === "holdout" ? p?.holdout_n : undefined;
                const suffix = n ? `  (N=${n})` : "";
                return [`${(value * 100).toFixed(1)}%${suffix}`, name];
              }
              return ["—", name];
            }}
          />
          {/* CI bands — drawn first so lines sit on top */}
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
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
