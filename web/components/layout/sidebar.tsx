import Link from "next/link";
import { FlaskConical, FolderKanban, Settings } from "lucide-react";

import { cn } from "@/lib/utils";

const nav = [
  { href: "/", label: "Projects", icon: FolderKanban },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/settings", label: "Settings", icon: Settings },
] as const;

export function Sidebar() {
  return (
    <aside className="hidden w-56 flex-col border-r border-border bg-card/50 px-3 py-5 md:flex">
      <Link href="/" className="mb-6 flex items-center gap-2 px-2">
        <div className="h-7 w-7 rounded-md bg-primary/15 ring-1 ring-primary/30" />
        <span className="font-mono text-sm font-medium tracking-tight">PromptLabs</span>
      </Link>
      <nav className="flex flex-col gap-1">
        {nav.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-2.5 rounded-md px-2.5 py-1.5",
              "text-sm text-muted-foreground transition-colors",
              "hover:bg-accent hover:text-foreground",
            )}
          >
            <Icon className="h-3.5 w-3.5" strokeWidth={1.75} />
            <span>{label}</span>
          </Link>
        ))}
      </nav>
      <div className="mt-auto px-2 pt-4">
        <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/70">
          v0.1.0 · local
        </p>
      </div>
    </aside>
  );
}
