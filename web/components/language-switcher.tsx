"use client";

import * as React from "react";
import { Languages } from "lucide-react";
import { LANGUAGES } from "@/lib/constants/languages";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

export function LanguageSwitcher({ className }: { className?: string }) {
  const { locale, setLocale } = useI18n();

  return (
    <label className={cn("relative inline-flex items-center", className)}>
      <Languages className="pointer-events-none absolute left-2.5 size-4 text-muted-foreground" aria-hidden />
      <select
        aria-label="Language"
        value={locale}
        onChange={(e) => setLocale(e.target.value)}
        className="h-10 cursor-pointer appearance-none rounded-full border bg-card pl-8 pr-4 text-sm font-semibold shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {LANGUAGES.map((l) => (
          <option key={l.code} value={l.code}>
            {l.native}
          </option>
        ))}
      </select>
    </label>
  );
}
