import { NextRequest, NextResponse } from "next/server";
import { MOCK_CHAT_ANSWERS } from "@/lib/mock/fixtures";

export const runtime = "nodejs";

/** Mock of §6.2 POST /api/v1/chat — follow-up Q&A. */
export async function POST(req: NextRequest) {
  let body: { question?: string; language?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { status: "error", code: "BAD_REQUEST", message: "Invalid JSON body." },
      { status: 400 }
    );
  }

  if (!body.question?.trim()) {
    return NextResponse.json(
      { status: "error", code: "MISSING_QUESTION", message: "Question is required." },
      { status: 400 }
    );
  }

  await new Promise((r) => setTimeout(r, 1200));

  const language = body.language ?? "en";
  return NextResponse.json({
    answer: MOCK_CHAT_ANSWERS[language] ?? MOCK_CHAT_ANSWERS.en,
    language,
    audio_url: null,
  });
}
