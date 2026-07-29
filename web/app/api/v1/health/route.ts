import { NextResponse } from "next/server";

/** Mock of §6.5 GET /api/v1/health. */
export async function GET() {
  return NextResponse.json({ status: "ok", version: "2.0.0-mock" });
}
