"use client";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface Props { data: { timestamp: string; consumption: number }[] }

export default function HistoricalChart({ data }: Props) {
  const sampled = data.filter((_, i) => i % Math.max(1, Math.floor(data.length / 500)) === 0);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={sampled} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="cGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1a2a3a" />
        <XAxis dataKey="timestamp" tick={{ fill: "#90a4ae", fontSize: 10 }}
          tickFormatter={v => v.slice(0, 10)} interval="preserveStartEnd" />
        <YAxis tick={{ fill: "#90a4ae", fontSize: 10 }} />
        <Tooltip contentStyle={{ background: "#0f2035", border: "1px solid #1a3a5a", color: "#e0e6ed" }}
          labelStyle={{ color: "#00d4ff" }} />
        <Area type="monotone" dataKey="consumption" stroke="#00d4ff" strokeWidth={1.5}
          fill="url(#cGrad)" name="kWh" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
