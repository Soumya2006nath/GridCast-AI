"use client";
import { useEffect, useState } from "react";

export default function ClientOnly({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div style={{ height: 260, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--muted)" }}>Loading chart…</div>;
  return <>{children}</>;
}
