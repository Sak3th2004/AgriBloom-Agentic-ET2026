import { z } from "zod";

/**
 * zod schemas for the FRONTEND_PLAN.md §6 API contract.
 * Every API response is parsed with these at runtime so contract
 * drift from the backend is caught immediately.
 */

export const DiagnoseStatus = z.enum([
  "ok",
  "uncertain",
  "invalid_image",
  "blocked",
  "error",
]);

export const Top3Item = z.object({
  label: z.string(),
  display_name: z.string(),
  confidence: z.number().min(0).max(1),
});

export const Disease = z.object({
  label: z.string(),
  display_name: z.string(),
  crop: z.string(),
  confidence: z.number().min(0).max(1),
  is_uncertain: z.boolean(),
  top3: z.array(Top3Item),
  source: z.enum(["ensemble", "efficientnet", "dinov2_knn", "llava", "fallback"]),
});

export const Compliance = z.object({
  allowed: z.boolean(),
  risk_level: z.enum(["low", "medium", "high"]),
  status: z.enum(["safe", "warning", "unsafe"]),
  blocked_substances: z.array(z.string()),
  safe_alternatives: z.array(z.string()),
  disclaimer: z.string(),
});

export const Weather = z.object({
  temp_c: z.number(),
  humidity: z.number(),
  rain_mm: z.number(),
  desc: z.string(),
});

export const Market = z.object({
  crop: z.string(),
  modal_price: z.number(),
  unit: z.string(),
  mandi: z.string(),
});

export const Chart = z.object({
  days: z.array(z.number()),
  without_treatment: z.array(z.number()),
  with_treatment: z.array(z.number()),
});

export const DiagnoseResponse = z.object({
  id: z.string(),
  session_id: z.string(),
  language: z.string(),
  status: DiagnoseStatus,
  disease: Disease.nullable(),
  treatment: z.string().nullable(),
  recommendations: z.array(z.string()),
  compliance: Compliance.nullable(),
  knowledge: z
    .object({
      weather: Weather.nullable(),
      market: Market.nullable(),
    })
    .nullable(),
  chart: Chart.nullable(),
  audio_url: z.string().nullable(),
  pdf_url: z.string().nullable(),
  elapsed_seconds: z.number(),
});

export const ChatResponse = z.object({
  answer: z.string(),
  language: z.string(),
  audio_url: z.string().nullable(),
});

export const TranscribeResponse = z.object({
  text: z.string(),
  language: z.string(),
});

export const LanguageInfo = z.object({
  code: z.string(),
  name: z.string(),
  native: z.string(),
});

export const HealthResponse = z.object({
  status: z.string(),
  version: z.string(),
});

export const MetaResponse = z.object({
  crops: z.array(z.string()),
  quick_symptoms: z.array(
    z.object({ id: z.string(), label: z.string(), icon: z.string() })
  ),
  districts: z.array(
    z.object({ name: z.string(), lat: z.number(), lon: z.number() })
  ),
});

export const ApiError = z.object({
  status: z.literal("error"),
  code: z.string(),
  message: z.string(),
});

export const StreamStep = z.object({
  stage: z.string(),
  message: z.string(),
  progress: z.number().min(0).max(1),
});

export type TDiagnoseResponse = z.infer<typeof DiagnoseResponse>;
export type TChatResponse = z.infer<typeof ChatResponse>;
export type TTranscribeResponse = z.infer<typeof TranscribeResponse>;
export type TLanguageInfo = z.infer<typeof LanguageInfo>;
export type TMetaResponse = z.infer<typeof MetaResponse>;
export type TApiError = z.infer<typeof ApiError>;
export type TStreamStep = z.infer<typeof StreamStep>;
export type TDisease = z.infer<typeof Disease>;
export type TCompliance = z.infer<typeof Compliance>;
export type TChart = z.infer<typeof Chart>;
