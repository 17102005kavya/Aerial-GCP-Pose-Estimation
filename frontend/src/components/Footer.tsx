"use client";

import { Target, Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full border-t border-border/40 bg-muted/20 py-6 sm:py-8 mt-auto">
      <div className="w-full px-6 md:px-12 lg:px-16">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-muted-foreground tracking-tight">
              AeroPoint AI &copy; {new Date().getFullYear()}
            </span>
          </div>

          <p className="text-xs text-muted-foreground text-center sm:text-left flex items-center gap-1">
            Built for professional Computer Vision GCP Pose Estimation. Powered by Next.js &amp; PyTorch.
          </p>

          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <span>Engineering Portfolio Project</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
