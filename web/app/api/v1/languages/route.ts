import { NextResponse } from "next/server";
import { LANGUAGES } from "@/lib/constants/languages";

/** Mock of §6.5 GET /api/v1/languages. */
export async function GET() {
  return NextResponse.json(LANGUAGES);
}
