"use client";
import { useState } from "react";
import DailyChart from "../charts/DailyChart";
import HistoricalChart from "../charts/HistoricalChart";

interface Props {
  daily: { timestamp: string; consumption: number }[];
  weekly: { timestamp: string; consumption: number }[];
  monthly: { timestamp: string; consumption: number }[];
  historical: { timestamp: string; consumption: number }[];
  preview: Record<string, unknown>[];
}

const TABS = ["Preview", "Daily", "Weekly", "Monthly", "Full History"];

export default function Analysis({ daily, weekly, monthly, historical, preview }: Props) {
  const [tab, setTab] = useState("Daily");

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold" style={{ color: "var(--cyan)" }}>📊 Data Analysis</h1>

      <div className="flex gap-2 flex-wrap">
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            className="px-4 py-1.5 rounded-full text-sm border transition-all"
            style={tab === t
              ? { background: "var(--blue)", color: "#fff", borderColor: "var(--blue)" }
              : { background: "transparent", color: "var(--muted)", borderColor: "var(--border)" }}>
            {t}
          </button>
        ))}
      </div>

      <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
        {tab === "Preview" && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: "var(--muted)" }}>
                  {Object.keys(preview[0] ?? {}).map(k => <th key={k} className="text-left pb-2 pr-4 font-medium">{k}</th>)}
                </tr>
              </thead>
              <tbody>
                {preview.slice(0, 50).map((row, i) => (
                  <tr key={i} className="border-t" style={{ borderColor: "var(--border)" }}>
                    {Object.values(row).map((v, j) => <td key={j} className="py-1 pr-4" style={{ color: "var(--text)" }}>{String(v)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {tab === "Daily" && <><p className="text-sm font-semibold mb-3" style={{ color: "var(--blue)" }}>Daily Consumption</p><DailyChart data={daily} /></>}
        {tab === "Weekly" && <><p className="text-sm font-semibold mb-3" style={{ color: "var(--green)" }}>Weekly Trend</p><DailyChart data={weekly} /></>}
        {tab === "Monthly" && monthly.length >= 2
          ? <><p className="text-sm font-semibold mb-3" style={{ color: "#9c27b0" }}>Monthly Trend</p><DailyChart data={monthly} /></>
          : tab === "Monthly" && <p style={{ color: "var(--muted)" }}>Need at least 2 months of data.</p>}
        {tab === "Full History" && <><p className="text-sm font-semibold mb-3" style={{ color: "var(--cyan)" }}>Historical Consumption</p><HistoricalChart data={historical} /></>}
      </div>
    </div>
  );
}
