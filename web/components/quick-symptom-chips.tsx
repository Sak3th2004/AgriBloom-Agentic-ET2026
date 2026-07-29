"use client";

import { motion } from "framer-motion";
import { QUICK_SYMPTOMS } from "@/lib/constants/symptoms";
import { useI18n } from "@/lib/i18n";

interface QuickSymptomChipsProps {
  onPick: (query: string) => void;
}

export function QuickSymptomChips({ onPick }: QuickSymptomChipsProps) {
  const { t } = useI18n();

  return (
    <div>
      <p className="mb-2 text-sm font-semibold text-muted-foreground">
        {t("home.quick")}
      </p>
      <div className="flex flex-wrap gap-2">
        {QUICK_SYMPTOMS.map((s, i) => (
          <motion.button
            key={s.id}
            type="button"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.04 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onPick(s.query)}
            className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full border bg-card px-4 py-2 text-sm font-semibold shadow-sm transition-colors hover:border-primary hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <span aria-hidden>{s.emoji}</span>
            {t(`symptom.${s.id}`)}
          </motion.button>
        ))}
      </div>
    </div>
  );
}
