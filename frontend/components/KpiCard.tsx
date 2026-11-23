interface Props {
  label: string;
  value: string;
  unit?: string;
  color?: string;
}

export default function KpiCard({ label, value, unit, color = "var(--cyan)" }: Props) {
  return (
    <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
      <p className="text-xs uppercase tracking-widest mb-1" style={{ color: "var(--muted)" }}>{label}</p>
      <p className="text-2xl font-bold" style={{ color }}>{value}</p>
      {unit && <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>{unit}</p>}
    </div>
  );
}
