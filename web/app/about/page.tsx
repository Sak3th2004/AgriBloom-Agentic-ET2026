"use client";

import { motion } from "framer-motion";
import { Code2, HeartHandshake, Phone, ShieldCheck } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useI18n } from "@/lib/i18n";

export default function AboutPage() {
  const { t } = useI18n();

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      <h1 className="text-xl font-extrabold tracking-tight">{t("about.title")}</h1>

      <Card>
        <CardHeader>
          <CardTitle>
            <HeartHandshake className="size-5 text-primary" aria-hidden />
            AgriBloom
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed text-muted-foreground">{t("about.body")}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            <Phone className="size-5 text-primary" aria-hidden />
            Helpline
          </CardTitle>
        </CardHeader>
        <CardContent>
          <a
            href="tel:18001801551"
            className="text-base font-bold text-primary hover:underline"
          >
            {t("about.helpline")}
          </a>
        </CardContent>
      </Card>

      <Card className="border-accent/50 bg-accent/5">
        <CardHeader>
          <CardTitle>
            <ShieldCheck className="size-5 text-accent-foreground dark:text-accent" aria-hidden />
            Disclaimer
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {t("about.disclaimer")}
          </p>
        </CardContent>
      </Card>

      <a
        href="https://github.com/Sak3th2004/AgriBloom-Agentic-ET2026"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center justify-center gap-2 rounded-lg border bg-card p-4 text-sm font-bold shadow-sm transition-colors hover:border-primary"
      >
        <Code2 className="size-5" aria-hidden />
        Open source on GitHub
      </a>
    </motion.div>
  );
}
