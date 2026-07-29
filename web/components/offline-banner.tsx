"use client";

import * as React from "react";
import { WifiOff } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useI18n } from "@/lib/i18n";

export function OfflineBanner() {
  const { t } = useI18n();
  const [offline, setOffline] = React.useState(false);

  React.useEffect(() => {
    setOffline(!navigator.onLine);
    const on = () => setOffline(false);
    const off = () => setOffline(true);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", off);
    };
  }, []);

  return (
    <AnimatePresence>
      {offline && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="overflow-hidden bg-accent text-accent-foreground"
          role="status"
        >
          <div className="mx-auto flex max-w-3xl items-center gap-2 px-4 py-2 text-sm font-semibold">
            <WifiOff className="size-4 shrink-0" aria-hidden />
            {t("offlineBanner")}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
