"use client";

import { motion } from "framer-motion";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

export function ConfidenceMeter({ value }: { value: number }) {
  const { t } = useI18n();
  const pct = Math.round(value * 100);
  const tone =
    pct >= 70 ? "bg-success" : pct >= 45 ? "bg-accent" : "bg-destructive";

  return (
    <div aria-label={`${t("result.confidence")}: ${pct}%`}>
      <div className="mb-1 flex items-center justify-between text-xs font-semibold text-muted-foreground">
        <span>{t("result.confidence")}</span>
        <span className="text-foreground">{pct}%</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-muted">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={cn("h-full rounded-full", tone)}
        />
      </div>
    </div>
  );
}
