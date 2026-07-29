"use client";

import { motion } from "framer-motion";
import {
  Bug,
  CheckCircle2,
  CircleDot,
  Cloudy,
  Leaf,
  TrendingDown,
  type LucideIcon,
} from "lucide-react";
import { QUICK_SYMPTOMS } from "@/lib/constants/symptoms";
import { useI18n } from "@/lib/i18n";

const ICONS: Record<string, LucideIcon> = {
  leaf_spots: CircleDot,
  yellow_leaves: Leaf,
  insects: Bug,
  white_fungus: Cloudy,
  wilting: TrendingDown,
  healthy_check: CheckCircle2,
};

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
        {QUICK_SYMPTOMS.map((s, i) => {
          const Icon = ICONS[s.id] ?? Leaf;
          return (
            <motion.button
              key={s.id}
              type="button"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.04 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => onPick(s.query)}
              className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full border bg-card px-4 py-2 text-sm font-semibold shadow-sm transition-colors hover:border-primary hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <Icon className="size-4 text-primary" aria-hidden />
              {t(`symptom.${s.id}`)}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
