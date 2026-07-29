"use client";

import Link from "next/link";
import { History, RefreshCcw, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n";

export default function OfflinePage() {
  const { t } = useI18n();

  return (
    <div className="flex flex-col items-center gap-4 py-20 text-center">
      <span className="flex size-16 items-center justify-center rounded-2xl bg-muted">
        <WifiOff className="size-8 text-muted-foreground" aria-hidden />
      </span>
      <h1 className="text-xl font-extrabold">{t("offline.title")}</h1>
      <p className="max-w-xs text-sm text-muted-foreground">{t("offline.body")}</p>
      <div className="flex gap-2">
        <Button onClick={() => window.location.reload()}>
          <RefreshCcw /> {t("offline.retry")}
        </Button>
        <Link href="/history">
          <Button variant="outline">
            <History /> {t("nav.history")}
          </Button>
        </Link>
      </div>
    </div>
  );
}
