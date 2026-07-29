"use client";

import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, Check, FileCheck2, Loader2, ScanEye, Sparkles, BookOpenCheck } from "lucide-react";
import { diagnoseStreamUrl } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

const STAGES = [
  { id: "orchestrator", icon: Brain },
  { id: "vision", icon: ScanEye },
  { id: "knowledge", icon: BookOpenCheck },
  { id: "compliance", icon: FileCheck2 },
  { id: "output", icon: Sparkles },
] as const;

/**
 * Live "agents at work" overlay. Follows the SSE progress stream (§6.3);
 * if SSE is unavailable it falls back to a timed simulation.
 */
export function AgentStepper({ active }: { active: boolean }) {
  const { t } = useI18n();
  const [stageIdx, setStageIdx] = React.useState(0);

  React.useEffect(() => {
    if (!active) {
      setStageIdx(0);
      return;
    }

    let fallbackTimer: ReturnType<typeof setInterval> | null = null;
    let source: EventSource | null = null;

    const startFallback = () => {
      if (fallbackTimer) return;
      fallbackTimer = setInterval(
        () => setStageIdx((i) => Math.min(i + 1, STAGES.length - 1)),
        1000
      );
    };

    try {
      source = new EventSource(diagnoseStreamUrl("live"));
      source.addEventListener("step", (e) => {
        try {
          const data = JSON.parse((e as MessageEvent).data) as { stage: string };
          const idx = STAGES.findIndex((s) => s.id === data.stage);
          if (idx >= 0) setStageIdx(idx);
        } catch {
          // malformed event — keep current stage
        }
      });
      source.addEventListener("done", () => setStageIdx(STAGES.length - 1));
      source.onerror = () => {
        source?.close();
        startFallback();
      };
    } catch {
      startFallback();
    }

    return () => {
      source?.close();
      if (fallbackTimer) clearInterval(fallbackTimer);
    };
  }, [active]);

  return (
    <AnimatePresence>
      {active && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-background/90 backdrop-blur-sm"
          role="status"
          aria-live="polite"
        >
          <div className="w-full max-w-sm rounded-lg border bg-card p-6 shadow-xl">
            <div className="mb-5 flex items-center gap-2">
              <Loader2 className="size-5 animate-spin text-primary" aria-hidden />
              <h2 className="text-base font-bold">{t("stepper.title")}</h2>
            </div>
            <ol className="space-y-3">
              {STAGES.map((stage, i) => {
                const done = i < stageIdx;
                const current = i === stageIdx;
                const Icon = stage.icon;
                return (
                  <motion.li
                    key={stage.id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.06 }}
                    className={cn(
                      "flex items-center gap-3 rounded-md px-3 py-2 transition-colors",
                      current && "bg-primary/10",
                      done && "opacity-70"
                    )}
                  >
                    <span
                      className={cn(
                        "flex size-8 shrink-0 items-center justify-center rounded-full border-2",
                        done
                          ? "border-success bg-success text-success-foreground"
                          : current
                            ? "border-primary text-primary"
                            : "border-muted-foreground/30 text-muted-foreground/50"
                      )}
                    >
                      {done ? <Check className="size-4" /> : <Icon className="size-4" />}
                    </span>
                    <span
                      className={cn(
                        "text-sm font-semibold",
                        current ? "text-foreground" : "text-muted-foreground"
                      )}
                    >
                      {t(`stepper.${stage.id}`)}
                    </span>
                    {current && (
                      <motion.span
                        layoutId="stepper-dot"
                        className="ml-auto size-2 rounded-full bg-primary"
                        animate={{ scale: [1, 1.5, 1] }}
                        transition={{ repeat: Infinity, duration: 1 }}
                      />
                    )}
                  </motion.li>
                );
              })}
            </ol>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
