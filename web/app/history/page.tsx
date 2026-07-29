"use client";

import * as React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ChevronRight, Leaf, ShieldAlert, Sprout, Trash2 } from "lucide-react";
import { clearHistory, listHistory, type HistoryEntry } from "@/lib/history";
import { useI18n } from "@/lib/i18n";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export default function HistoryPage() {
  const { t } = useI18n();
  const [entries, setEntries] = React.useState<HistoryEntry[] | null>(null);

  React.useEffect(() => {
    setEntries(listHistory());
  }, []);

  if (entries === null) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-extrabold tracking-tight">{t("history.title")}</h1>
        {entries.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              clearHistory();
              setEntries([]);
            }}
          >
            <Trash2 /> {t("history.clear")}
          </Button>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-lg border-2 border-dashed py-16 text-center">
          <Sprout className="size-10 text-muted-foreground/50" aria-hidden />
          <p className="max-w-xs text-sm font-semibold text-muted-foreground">
            {t("history.empty")}
          </p>
        </div>
      ) : (
        <ul className="space-y-2">
          {entries.map((e, i) => (
            <motion.li
              key={e.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
            >
              <Link
                href={`/result/${e.id}`}
                className="flex items-center gap-3 rounded-lg border bg-card p-4 shadow-sm transition-colors hover:border-primary"
              >
                <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  {e.status === "blocked" ? (
                    <ShieldAlert className="size-5 text-destructive" aria-hidden />
                  ) : (
                    <Leaf className="size-5" aria-hidden />
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-bold">{e.disease}</span>
                  <span className="block text-xs text-muted-foreground">
                    {new Date(e.date).toLocaleDateString()} ·{" "}
                    <span className="capitalize">{e.crop}</span>
                  </span>
                </span>
                {e.status === "blocked" && <Badge variant="destructive">!</Badge>}
                <ChevronRight className="size-5 shrink-0 text-muted-foreground" aria-hidden />
              </Link>
            </motion.li>
          ))}
        </ul>
      )}
    </div>
  );
}
