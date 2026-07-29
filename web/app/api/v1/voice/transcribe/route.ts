import { NextRequest, NextResponse } from "next/server";
import { MOCK_TRANSCRIPTIONS } from "@/lib/mock/fixtures";

export const runtime = "nodejs";

/** Mock of §6.4 POST /api/v1/voice/transcribe — mic audio to text. */
export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const audio = form.get("audio");
    if (!(audio instanceof Blob) || audio.size === 0) {
      return NextResponse.json(
        { status: "error", code: "MISSING_AUDIO", message: "Audio file is required." },
        { status: 400 }
      );
    }
    const language = ((form.get("language") as string) || "auto").toLowerCase();

    await new Promise((r) => setTimeout(r, 1000));

    const lang = language === "auto" ? "en" : language;
    return NextResponse.json({
      text: MOCK_TRANSCRIPTIONS[lang] ?? MOCK_TRANSCRIPTIONS.en,
      language: lang,
    });
  } catch {
    return NextResponse.json(
      { status: "error", code: "BAD_REQUEST", message: "Malformed form data." },
      { status: 400 }
    );
  }
}
