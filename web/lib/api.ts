import {
  ApiError,
  ChatResponse,
  DiagnoseResponse,
  HealthResponse,
  LanguageInfo,
  MetaResponse,
  TranscribeResponse,
  type TChatResponse,
  type TDiagnoseResponse,
  type TTranscribeResponse,
} from "@/lib/schemas";
import { z } from "zod";

/**
 * Typed API client for the FRONTEND_PLAN.md §6 contract.
 *
 * NEXT_PUBLIC_API_BASE_URL:
 *   ""            -> same-origin (the built-in mock API under /app/api/v1)
 *   "https://..." -> the real FastAPI backend (flip when it's live)
 */
const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

export class ApiRequestError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

async function parseOrThrow<T>(res: Response, schema: z.ZodType<T>): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = ApiError.safeParse(body);
    if (err.success) throw new ApiRequestError(err.data.code, err.data.message);
    throw new ApiRequestError("HTTP_" + res.status, res.statusText || "Request failed");
  }
  return schema.parse(body);
}

export interface DiagnoseInput {
  image?: File | Blob | null;
  text?: string;
  language: string;
  lat?: number;
  lon?: number;
  offline?: boolean;
}

export async function diagnose(input: DiagnoseInput): Promise<TDiagnoseResponse> {
  const fd = new FormData();
  if (input.image) fd.append("image", input.image, "leaf.jpg");
  if (input.text) fd.append("text", input.text);
  fd.append("language", input.language);
  if (input.lat !== undefined) fd.append("lat", String(input.lat));
  if (input.lon !== undefined) fd.append("lon", String(input.lon));
  fd.append("offline", String(Boolean(input.offline)));

  const res = await fetch(`${BASE}/api/v1/diagnose`, { method: "POST", body: fd });
  return parseOrThrow(res, DiagnoseResponse);
}

export async function chat(input: {
  session_id: string;
  question: string;
  language: string;
  history: { role: "user" | "assistant"; content: string }[];
}): Promise<TChatResponse> {
  const res = await fetch(`${BASE}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return parseOrThrow(res, ChatResponse);
}

export async function transcribe(
  audio: Blob,
  language = "auto"
): Promise<TTranscribeResponse> {
  const fd = new FormData();
  fd.append("audio", audio, "voice.webm");
  fd.append("language", language);
  const res = await fetch(`${BASE}/api/v1/voice/transcribe`, {
    method: "POST",
    body: fd,
  });
  return parseOrThrow(res, TranscribeResponse);
}

export async function getLanguages() {
  const res = await fetch(`${BASE}/api/v1/languages`);
  return parseOrThrow(res, z.array(LanguageInfo));
}

export async function getMeta() {
  const res = await fetch(`${BASE}/api/v1/meta`);
  return parseOrThrow(res, MetaResponse);
}

export async function getHealth() {
  const res = await fetch(`${BASE}/api/v1/health`);
  return parseOrThrow(res, HealthResponse);
}

/** SSE endpoint for the live agent-progress stepper. */
export function diagnoseStreamUrl(job: string): string {
  return `${BASE}/api/v1/diagnose/stream?job=${encodeURIComponent(job)}`;
}

/** audio_url / pdf_url from the API may be relative — prefix the base. */
export function fileUrl(path: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${BASE}${path}`;
}
