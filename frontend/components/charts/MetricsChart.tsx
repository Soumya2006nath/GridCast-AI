"use client";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

interface ModelMetrics { MAE: number; RMSE: number; MAPE: number | null }
interface Props { metrics: Record<string, ModelMetrics> }

export default function MetricsChart({ metrics }: Props) {
  const data = Object.entries(metrics).map(([model, m]) => ({
    model,
    MAE: +(m.MAE ?? 0).toFixed(4),
    RMSE: +(m.RMSE ?? 0).toFixed(4),
    MAPE: m.MAPE != null ? +m.MAPE.toFixed(2) : 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1a2a3a" />
        <XAxis dataKey="model" tick={{ fill: "#90a4ae", fontSize: 12 }} />
        <YAxis tick={{ fill: "#90a4ae", fontSize: 11 }} />
        <Tooltip contentStyle={{ background: "#0f2035", border: "1px solid #1a3a5a", color: "#e0e6ed" }} />
        <Legend wrapperStyle={{ color: "#90a4ae", fontSize: 12 }} />
        <Bar dataKey="MAE" fill="#1e90ff" radius={[4, 4, 0, 0]} />
        <Bar dataKey="RMSE" fill="#00d4ff" radius={[4, 4, 0, 0]} />
        <Bar dataKey="MAPE" fill="#00e676" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
