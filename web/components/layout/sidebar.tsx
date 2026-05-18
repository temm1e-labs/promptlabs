"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FlaskConical, FolderKanban, Settings } from "lucide-react";

import { cn } from "@/lib/utils";

const nav = [
  { href: "/", label: "Projects", icon: FolderKanban, match: (p: string) => p === "/" || p.startsWith("/projects") },
  { href: "/experiments", label: "Experiments", icon: FlaskConical, match: (p: string) => p.startsWith("/experiments") },
  { href: "/settings", label: "Settings", icon: Settings, match: (p: string) => p.startsWith("/settings") },
] as const;

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden w-56 flex-col border-r border-border bg-card/50 px-3 py-5 md:flex">
      <Link href="/" className="mb-6 flex items-center gap-2 px-2">
        <div className="h-7 w-7 rounded-md bg-primary/15 ring-1 ring-primary/30" />
        <span className="font-mono text-sm font-medium tracking-tight">PromptLabs</span>
      </Link>
      <nav className="flex flex-col gap-1">
        {nav.map(({ href, label, icon: Icon, match }) => {
          const active = match(pathname ?? "");
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2.5 rounded-md px-2.5 py-1.5",
                "text-sm transition-colors",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                active
                  ? "bg-accent text-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              <Icon className="h-3.5 w-3.5" strokeWidth={1.75} />
              <span>{label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto px-2 pt-4">
        <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/70">
          v0.1.0 · local
        </p>
      </div>
    </aside>
  );
}
