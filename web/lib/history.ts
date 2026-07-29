import type { TDiagnoseResponse } from "@/lib/schemas";

/** localStorage persistence for past diagnoses (history + result page). */

const LIST_KEY = "agribloom.history";
const ITEM_PREFIX = "agribloom.diag.";
const MAX_ITEMS = 50;

export interface HistoryEntry {
  id: string;
  date: string; // ISO
  disease: string;
  crop: string;
  status: string;
  language: string;
}

export function saveDiagnosis(result: TDiagnoseResponse): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(ITEM_PREFIX + result.id, JSON.stringify(result));
    const entry: HistoryEntry = {
      id: result.id,
      date: new Date().toISOString(),
      disease: result.disease?.display_name ?? "—",
      crop: result.disease?.crop ?? "",
      status: result.status,
      language: result.language,
    };
    const list = listHistory().filter((e) => e.id !== result.id);
    list.unshift(entry);
    const trimmed = list.slice(0, MAX_ITEMS);
    for (const removed of list.slice(MAX_ITEMS)) {
      localStorage.removeItem(ITEM_PREFIX + removed.id);
    }
    localStorage.setItem(LIST_KEY, JSON.stringify(trimmed));
  } catch {
    // storage full/unavailable — history is best-effort
  }
}

export function listHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(LIST_KEY) ?? "[]") as HistoryEntry[];
  } catch {
    return [];
  }
}

export function getDiagnosis(id: string): TDiagnoseResponse | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(ITEM_PREFIX + id);
    return raw ? (JSON.parse(raw) as TDiagnoseResponse) : null;
  } catch {
    return null;
  }
}

export function clearHistory(): void {
  if (typeof window === "undefined") return;
  for (const e of listHistory()) localStorage.removeItem(ITEM_PREFIX + e.id);
  localStorage.removeItem(LIST_KEY);
}
