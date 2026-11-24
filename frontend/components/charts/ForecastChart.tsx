"use client";
import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";

interface Props {
  data: { Date: string; Predicted_kWh: number }[];
  historicalAvg: number;
  model: string;
}

export default function ForecastChart({ data, historicalAvg, model }: Props) {
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  if (!ready) return <div style={{ height: 260 }} />;

  return (
    <div>
      <p className="text-sm font-semibold mb-3" style={{ color: "var(--cyan)" }}>7-Day Forecast — {model}</p>
      <div style={{ width: "100%", height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a2a3a" />
            <XAxis dataKey="Date" tick={{ fill: "#90a4ae", fontSize: 11 }} tickFormatter={v => v.slice(5)} />
            <YAxis tick={{ fill: "#90a4ae", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f2035", border: "1px solid #1a3a5a", color: "#e0e6ed" }}
              labelStyle={{ color: "#00d4ff" }} formatter={(v) => [`${Number(v).toFixed(2)} kWh`, "Forecast"]} />
            <ReferenceLine y={historicalAvg} stroke="#ff9100" strokeDasharray="5 5"
              label={{ value: `Hist avg: ${historicalAvg.toFixed(1)} kWh`, fill: "#ff9100", fontSize: 11 }} />
            <Bar dataKey="Predicted_kWh" fill="#00d4ff" radius={[4, 4, 0, 0]} name="Forecast kWh" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
