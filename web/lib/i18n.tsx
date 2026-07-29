"use client";

import * as React from "react";
import { DEFAULT_LANGUAGE, LANGUAGES } from "@/lib/constants/languages";

import en from "@/messages/en.json";
import hi from "@/messages/hi.json";
import kn from "@/messages/kn.json";
import te from "@/messages/te.json";
import ta from "@/messages/ta.json";
import bn from "@/messages/bn.json";
import mr from "@/messages/mr.json";
import gu from "@/messages/gu.json";
import ml from "@/messages/ml.json";
import pa from "@/messages/pa.json";

type Messages = Record<string, string>;

const MESSAGES: Record<string, Messages> = { en, hi, kn, te, ta, bn, mr, gu, ml, pa };

const STORAGE_KEY = "agribloom.lang";

interface I18nContextValue {
  locale: string;
  setLocale: (code: string) => void;
  t: (key: string) => string;
}

const I18nContext = React.createContext<I18nContextValue>({
  locale: DEFAULT_LANGUAGE,
  setLocale: () => {},
  t: (k) => k,
});

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = React.useState(DEFAULT_LANGUAGE);

  React.useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && LANGUAGES.some((l) => l.code === saved)) setLocaleState(saved);
  }, []);

  const setLocale = React.useCallback((code: string) => {
    setLocaleState(code);
    localStorage.setItem(STORAGE_KEY, code);
    document.documentElement.lang = code;
  }, []);

  const t = React.useCallback(
    (key: string): string => MESSAGES[locale]?.[key] ?? MESSAGES.en[key] ?? key,
    [locale]
  );

  const value = React.useMemo(() => ({ locale, setLocale, t }), [locale, setLocale, t]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return React.useContext(I18nContext);
}
