"use client";

import Link from "next/link";
import { ThemeToggle } from "./ThemeToggle";
import { Target, Cpu, Map } from "lucide-react";

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/60 backdrop-blur-md supports-[backdrop-filter]:bg-background/60">
      <div className="w-full flex h-16 items-center justify-between px-4 sm:px-8 lg:px-12">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-90">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-primary to-indigo-500 text-primary-foreground shadow-md shadow-primary/20">
              <Target className="h-5.5 w-5.5 animate-pulse" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg leading-tight tracking-tight bg-gradient-to-r from-foreground via-foreground/90 to-muted-foreground/80 bg-clip-text text-transparent">
                AeroPoint AI
              </span>
              <span className="text-[10px] text-muted-foreground leading-none font-medium tracking-wide uppercase">
                GCP Pose Estimation
              </span>
            </div>
          </Link>

          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-muted-foreground">
            <Link href="#workspace" className="transition-colors hover:text-foreground">
              Detection Workspace
            </Link>
            <Link href="#how-it-works" className="transition-colors hover:text-foreground">
              Methodology
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/40 px-3 py-1 text-xs font-semibold text-muted-foreground">
            <Cpu className="h-3.5 w-3.5 text-primary" />
            <span>EfficientNet-B3 Cascade</span>
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
