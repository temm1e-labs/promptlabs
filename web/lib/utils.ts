import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCost(usd: number): string {
  if (usd < 0.01) return `$${(usd * 100).toFixed(2)}¢`;
  if (usd < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
}

export function formatScore(score: number | null | undefined): string {
  if (score == null) return "—";
  return (score * 100).toFixed(1) + "%";
}

export function scoreBand(score: number): "good" | "mid" | "bad" {
  if (score >= 0.8) return "good";
  if (score >= 0.5) return "mid";
  return "bad";
}
