"use client";

import { motion } from "framer-motion";
import { CheckCircle2, ShieldAlert, TriangleAlert } from "lucide-react";
import type { TCompliance } from "@/lib/schemas";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

export function ComplianceCard({ compliance }: { compliance: TCompliance }) {
  const { t } = useI18n();
  const blocked = !compliance.allowed;
  const warning = compliance.status === "warning";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
    >
      <Card
        className={cn(
          "border-2",
          blocked
            ? "border-destructive/60 bg-destructive/5"
            : warning
              ? "border-accent/60 bg-accent/5"
              : "border-success/50 bg-success/5"
        )}
        role={blocked ? "alert" : undefined}
      >
        <CardHeader>
          <CardTitle
            className={cn(
              blocked
                ? "text-destructive"
                : warning
                  ? "text-accent-foreground dark:text-accent"
                  : "text-success"
            )}
          >
            {blocked ? (
              <ShieldAlert className="size-5" aria-hidden />
            ) : warning ? (
              <TriangleAlert className="size-5" aria-hidden />
            ) : (
              <CheckCircle2 className="size-5" aria-hidden />
            )}
            {t("result.safety")}: {blocked ? t("result.blocked") : t("result.safe")}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {compliance.blocked_substances.length > 0 && (
            <div>
              <p className="mb-1.5 text-sm font-bold text-destructive">
                {t("result.blockedSub")}:
              </p>
              <div className="flex flex-wrap gap-1.5">
                {compliance.blocked_substances.map((s) => (
                  <Badge key={s} variant="destructive">
                    {s}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {compliance.safe_alternatives.length > 0 && (
            <div>
              <p className="mb-1.5 text-sm font-bold text-success">
                {t("result.alternatives")}:
              </p>
              <ul className="space-y-1.5">
                {compliance.safe_alternatives.map((alt) => (
                  <li key={alt} className="flex items-start gap-2 text-sm">
                    <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-success" aria-hidden />
                    {alt}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <p className="text-xs italic text-muted-foreground">
            {compliance.disclaimer}
          </p>
        </CardContent>
      </Card>
    </motion.div>
  );
}
