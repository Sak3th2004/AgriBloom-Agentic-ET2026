"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { History, Home, LifeBuoy } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

export function BottomNav() {
  const pathname = usePathname();
  const { t } = useI18n();

  const items = [
    { href: "/", icon: Home, label: t("nav.home") },
    { href: "/history", icon: History, label: t("nav.history") },
    { href: "/about", icon: LifeBuoy, label: t("nav.help") },
  ];

  return (
    <nav
      aria-label="Main"
      className="fixed inset-x-0 bottom-0 z-40 border-t bg-background/90 backdrop-blur-md safe-bottom md:hidden"
    >
      <div className="mx-auto flex max-w-3xl items-stretch justify-around">
        {items.map(({ href, icon: Icon, label }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex min-w-[72px] flex-col items-center gap-0.5 px-3 py-2 text-[11px] font-semibold transition-colors",
                active ? "text-primary" : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Icon className={cn("size-6", active && "fill-primary/15")} aria-hidden />
              {label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
