"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type TrajectoryPoint = {
  iteration: number;
  train: number | null;
  holdout: number | null;
};

export function ScoreTrajectory({ data }: { data: TrajectoryPoint[] }) {
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
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
            formatter={(value: number) =>
              value == null ? "—" : `${(value * 100).toFixed(1)}%`
            }
            labelFormatter={(label) => `iteration v${label}`}
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
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
