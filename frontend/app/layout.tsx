import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GridCast AI — Smart Grid Forecasting",
  description: "AI-powered electricity consumption forecasting — Group 14",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen" style={{ background: "var(--bg)" }}>
        {children}
      </body>
    </html>
  );
}
