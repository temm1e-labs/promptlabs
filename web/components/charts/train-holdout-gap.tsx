"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type GapPoint = {
  iteration: number;
  gap: number; // train - holdout
};

export function TrainHoldoutGap({ data }: { data: GapPoint[] }) {
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
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
      </ResponsiveContainer>
    </div>
  );
}
