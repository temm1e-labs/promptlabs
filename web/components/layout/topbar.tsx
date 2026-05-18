"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

export function Topbar() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const active = resolvedTheme ?? theme;

  return (
    <header
      className={cn(
        "sticky top-0 z-30 flex h-12 items-center justify-end gap-3",
        "border-b border-border bg-background/70 px-6 backdrop-blur",
      )}
    >
      <button
        type="button"
        aria-label="Toggle theme"
        onClick={() => setTheme(active === "dark" ? "light" : "dark")}
        className={cn(
          "inline-flex h-7 w-7 items-center justify-center rounded-md",
          "text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
        )}
      >
        {mounted && active === "dark" ? (
          <Sun className="h-3.5 w-3.5" strokeWidth={1.75} />
        ) : (
          <Moon className="h-3.5 w-3.5" strokeWidth={1.75} />
        )}
      </button>
    </header>
  );
}
