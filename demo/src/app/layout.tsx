import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GP Agent — AMD Hackathon",
  description: "A token-efficient, local-first general-purpose AI agent.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
