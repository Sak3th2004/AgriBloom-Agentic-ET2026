import { NextResponse } from "next/server";
import districts from "@/lib/constants/districts.json";
import { QUICK_SYMPTOMS } from "@/lib/constants/symptoms";

/** Mock of §6.5 GET /api/v1/meta. */
export async function GET() {
  return NextResponse.json({
    crops: [
      "grape", "tomato", "cotton", "rice", "wheat", "maize", "chilli",
      "banana", "mango", "potato", "onion", "sugarcane",
    ],
    quick_symptoms: QUICK_SYMPTOMS.map((s) => ({
      id: s.id,
      label: s.id.replace(/_/g, " "),
      icon: s.id,
    })),
    districts,
  });
}
