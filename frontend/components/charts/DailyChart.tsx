"use client";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface Props { data: { timestamp: string; consumption: number }[] }

export default function DailyChart({ data }: Props) {
  const display = data.slice(-60);
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={display} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1a2a3a" />
        <XAxis dataKey="timestamp" tick={{ fill: "#90a4ae", fontSize: 10 }} tickFormatter={v => v.slice(5, 10)} interval={6} />
        <YAxis tick={{ fill: "#90a4ae", fontSize: 10 }} />
        <Tooltip contentStyle={{ background: "#0f2035", border: "1px solid #1a3a5a", color: "#e0e6ed" }}
          formatter={(v: number) => [`${v.toFixed(2)} kWh`, "Daily"]} />
        <Bar dataKey="consumption" fill="#1e90ff" radius={[3, 3, 0, 0]} name="kWh" />
      </BarChart>
    </ResponsiveContainer>
  );
}
