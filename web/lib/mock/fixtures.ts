import type { TDiagnoseResponse } from "@/lib/schemas";

/**
 * Mock fixtures for the built-in mock API (app/api/v1).
 * Scenario is picked from trigger words in the submitted text — see pickScenario().
 * These mirror FRONTEND_PLAN.md §6 example payloads.
 */

function chart14(base: number, declineTo: number, recoverTo: number) {
  const days = Array.from({ length: 14 }, (_, i) => i);
  const without = days.map((d) =>
    Math.round(base - ((base - declineTo) * d) / 13)
  );
  const withT = days.map((d) => {
    if (d < 3) return Math.round(base - ((base - (base - 4)) * d) / 3);
    return Math.round(base - 4 + (((recoverTo - (base - 4)) * (d - 3)) / 10));
  });
  return { days, without_treatment: without, with_treatment: withT };
}

const weather = { temp_c: 29, humidity: 72, rain_mm: 2.4, desc: "Partly cloudy" };

export function buildFixture(
  scenario: string,
  id: string,
  language: string
): TDiagnoseResponse {
  const common = {
    id,
    session_id: `sess_${id.slice(5)}`,
    language,
    audio_url: `/api/v1/files/${id}.wav`,
    pdf_url: `/api/v1/files/${id}.pdf`,
    elapsed_seconds: 4.2,
  };

  switch (scenario) {
    case "blocked":
      return {
        ...common,
        status: "blocked",
        disease: {
          label: "cotton_aphids",
          display_name: "Cotton Aphid Infestation",
          crop: "cotton",
          confidence: 0.88,
          is_uncertain: false,
          top3: [
            { label: "cotton_aphids", display_name: "Cotton Aphid Infestation", confidence: 0.88 },
            { label: "cotton_whitefly", display_name: "Cotton Whitefly", confidence: 0.07 },
            { label: "cotton_healthy", display_name: "Healthy Cotton", confidence: 0.03 },
          ],
          source: "ensemble",
        },
        treatment:
          "Spray neem oil (5 ml per litre of water) on the underside of leaves in the early morning. Repeat after 7 days. Introduce ladybird beetles if available. Avoid broad-spectrum insecticides that kill natural predators.",
        recommendations: [
          "Remove heavily infested shoots and destroy them away from the field",
          "Use yellow sticky traps (10 per acre) to monitor aphid levels",
          "Avoid excess nitrogen fertilizer — soft growth attracts aphids",
        ],
        compliance: {
          allowed: false,
          risk_level: "high",
          status: "unsafe",
          blocked_substances: ["Monocrotophos", "Ridomil MZ"],
          safe_alternatives: [
            "Neem oil 5% (azadirachtin) — organic, safe near harvest",
            "Imidacloprid 17.8% SL at 0.3 ml/litre — follow label PHI",
            "Verticillium lecanii bio-insecticide",
          ],
          disclaimer:
            "Monocrotophos is BANNED for use on vegetables and several crops in India (CIB&RC). Consult your local KVK before applying any pesticide.",
        },
        knowledge: {
          weather,
          market: { crop: "cotton", modal_price: 7100, unit: "quintal", mandi: "Warangal" },
        },
        chart: chart14(78, 40, 88),
      };

    case "healthy":
      return {
        ...common,
        status: "ok",
        disease: {
          label: "tomato_healthy",
          display_name: "Healthy Tomato Plant",
          crop: "tomato",
          confidence: 0.94,
          is_uncertain: false,
          top3: [
            { label: "tomato_healthy", display_name: "Healthy Tomato Plant", confidence: 0.94 },
            { label: "tomato_early_blight", display_name: "Tomato Early Blight", confidence: 0.04 },
            { label: "tomato_leaf_mold", display_name: "Tomato Leaf Mold", confidence: 0.01 },
          ],
          source: "efficientnet",
        },
        treatment:
          "No disease detected — your crop looks healthy! Continue regular care: water at the base (not on leaves), and scout the field twice a week for early signs of pests or spots.",
        recommendations: [
          "Maintain balanced NPK — avoid over-fertilizing with nitrogen",
          "Keep 45–60 cm spacing for airflow to prevent fungal disease",
          "Scout the underside of leaves weekly for eggs and mites",
        ],
        compliance: {
          allowed: true,
          risk_level: "low",
          status: "safe",
          blocked_substances: [],
          safe_alternatives: [],
          disclaimer: "No pesticide needed for a healthy crop. Consult your local KVK for preventive schedules.",
        },
        knowledge: {
          weather,
          market: { crop: "tomato", modal_price: 1850, unit: "quintal", mandi: "Kolar" },
        },
        chart: null,
      };

    case "uncertain":
      return {
        ...common,
        status: "uncertain",
        disease: {
          label: "unknown_crop_disease",
          display_name: "Possible Fungal Infection",
          crop: "unknown",
          confidence: 0.31,
          is_uncertain: true,
          top3: [
            { label: "unknown_crop_disease", display_name: "Possible Fungal Infection", confidence: 0.31 },
            { label: "leaf_spot_generic", display_name: "Generic Leaf Spot", confidence: 0.27 },
            { label: "nutrient_deficiency", display_name: "Nutrient Deficiency", confidence: 0.22 },
          ],
          source: "dinov2_knn",
        },
        treatment:
          "The image is not clear enough for a confident diagnosis. General care: remove visibly damaged leaves, avoid overhead watering, and retake a closer photo of a single affected leaf in daylight.",
        recommendations: [
          "Retake the photo: one leaf, filling the frame, in daylight",
          "Photograph both the top and bottom of the leaf",
          "If spots spread fast, call the Kisan helpline: 1800-180-1551",
        ],
        compliance: {
          allowed: true,
          risk_level: "medium",
          status: "warning",
          blocked_substances: [],
          safe_alternatives: [],
          disclaimer: "Do not spray chemicals until the disease is confirmed. Consult your local KVK.",
        },
        knowledge: { weather, market: null },
        chart: null,
      };

    case "invalid_image":
      return {
        ...common,
        status: "invalid_image",
        disease: null,
        treatment: null,
        recommendations: [
          "Hold the camera 15–20 cm from a single leaf",
          "Use natural daylight, avoid shadows and flash",
          "Keep the leaf flat and in focus before shooting",
        ],
        compliance: null,
        knowledge: null,
        chart: null,
        audio_url: null,
        pdf_url: null,
      };

    default:
      // happy path — grape downy mildew (the §6.1 example)
      return {
        ...common,
        status: "ok",
        disease: {
          label: "grape_downy_mildew",
          display_name: "Grape Downy Mildew",
          crop: "grape",
          confidence: 0.85,
          is_uncertain: false,
          top3: [
            { label: "grape_downy_mildew", display_name: "Grape Downy Mildew", confidence: 0.85 },
            { label: "grape_black_rot", display_name: "Grape Black Rot", confidence: 0.09 },
            { label: "grape_healthy", display_name: "Healthy Grape", confidence: 0.03 },
          ],
          source: "ensemble",
        },
        treatment:
          "Spray Bordeaux mixture (1%) or copper oxychloride 50% WP at 3 g/litre of water. Cover the underside of leaves where the white downy growth appears. Repeat every 10 days during humid weather. Prune dense foliage to improve airflow.",
        recommendations: [
          "Remove and burn infected leaves — do not compost them",
          "Irrigate in the morning so leaves dry before evening",
          "Avoid overhead sprinklers during the monsoon period",
          "Ensure 2.5–3 m row spacing for ventilation in the vineyard",
        ],
        compliance: {
          allowed: true,
          risk_level: "low",
          status: "safe",
          blocked_substances: [],
          safe_alternatives: [],
          disclaimer:
            "Copper-based fungicides are approved by CIB&RC for grapes. Observe a 15-day pre-harvest interval. Consult your local KVK.",
        },
        knowledge: {
          weather,
          market: { crop: "grape", modal_price: 4500, unit: "quintal", mandi: "Bengaluru" },
        },
        chart: chart14(80, 42, 90),
      };
  }
}

/** Deterministic scenario triggers so every UI state is reachable from the mock. */
export function pickScenario(text: string, hasImage: boolean): string {
  const q = text.toLowerCase();
  if (q.includes("blur") || q.includes("invalid")) return "invalid_image";
  if (
    q.includes("ridomil") ||
    q.includes("banned") ||
    q.includes("monocrotophos") ||
    q.includes("insect")
  )
    return "blocked";
  if (q.includes("healthy")) return "healthy";
  if (q.includes("unknown") || q.includes("not sure") || q.includes("strange"))
    return "uncertain";
  if (!hasImage && q.trim().length < 12) return "uncertain";
  return "happy";
}

export const MOCK_CHAT_ANSWERS: Record<string, string> = {
  en: "Spray once every 7–10 days while the weather stays humid. Stop at least 15 days before harvest. If new spots keep appearing after two sprays, contact your local KVK for a field inspection.",
  hi: "जब तक मौसम नम रहे, हर 7–10 दिन में एक बार छिड़काव करें। कटाई से कम से कम 15 दिन पहले रोक दें। दो छिड़काव के बाद भी नए धब्बे दिखें तो अपने KVK से खेत की जांच कराएं।",
  te: "వాతావరణం తేమగా ఉన్నంత వరకు ప్రతి 7–10 రోజులకు ఒకసారి పిచికారీ చేయండి. కోతకు కనీసం 15 రోజుల ముందు ఆపండి. రెండు పిచికారీల తర్వాత కూడా కొత్త మచ్చలు కనిపిస్తే మీ KVK ని సంప్రదించండి.",
  kn: "ಹವಾಮಾನ ತೇವವಿರುವವರೆಗೆ ಪ್ರತಿ 7–10 ದಿನಗಳಿಗೊಮ್ಮೆ ಸಿಂಪಡಿಸಿ. ಕೊಯ್ಲಿಗೆ ಕನಿಷ್ಠ 15 ದಿನ ಮೊದಲು ನಿಲ್ಲಿಸಿ. ಎರಡು ಸಿಂಪಡಣೆಯ ನಂತರವೂ ಹೊಸ ಕಲೆಗಳು ಕಂಡರೆ ನಿಮ್ಮ KVK ಸಂಪರ್ಕಿಸಿ.",
  ta: "வானிலை ஈரமாக இருக்கும் வரை 7–10 நாட்களுக்கு ஒருமுறை தெளிக்கவும். அறுவடைக்கு குறைந்தது 15 நாட்களுக்கு முன் நிறுத்தவும். இரண்டு தெளிப்புக்குப் பிறகும் புதிய புள்ளிகள் தோன்றினால் உங்கள் KVK ஐ தொடர்பு கொள்ளவும்.",
};

export const MOCK_TRANSCRIPTIONS: Record<string, string> = {
  en: "There are black spots spreading on my grape leaves, what should I do?",
  hi: "मेरी अंगूर की पत्तियों पर काले धब्बे फैल रहे हैं, क्या करूं?",
  te: "నా ద్రాక్ష ఆకులపై నల్ల మచ్చలు వ్యాపిస్తున్నాయి, ఏమి చేయాలి?",
  kn: "ನನ್ನ ದ್ರಾಕ್ಷಿ ಎಲೆಗಳ ಮೇಲೆ ಕಪ್ಪು ಕಲೆಗಳು ಹರಡುತ್ತಿವೆ, ಏನು ಮಾಡಬೇಕು?",
  ta: "என் திராட்சை இலைகளில் கருப்பு புள்ளிகள் பரவுகின்றன, என்ன செய்வது?",
};
