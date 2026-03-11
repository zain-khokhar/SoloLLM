import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "SoloLLM — Private Local AI Platform",
  description:
    "A self-hosted, privacy-first LLM platform with advanced RAG, auto-continuation, and context distillation. All processing happens locally — your data never leaves your machine.",
  keywords: ["LLM", "local AI", "privacy", "RAG", "self-hosted"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} antialiased`} style={{ fontFamily: "'Inter', sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
