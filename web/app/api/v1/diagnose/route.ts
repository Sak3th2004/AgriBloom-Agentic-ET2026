import { NextRequest, NextResponse } from "next/server";
import { buildFixture, pickScenario } from "@/lib/mock/fixtures";

export const runtime = "nodejs";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Mock of §6.1 POST /api/v1/diagnose — replaced by the real FastAPI backend later. */
export async function POST(req: NextRequest) {
  let text = "";
  let hasImage = false;
  let language = "en";

  try {
    const form = await req.formData();
    text = (form.get("text") as string) ?? "";
    language = (form.get("language") as string) || "en";
    const image = form.get("image");
    hasImage = image instanceof Blob && image.size > 0;

    if (!hasImage && !text.trim()) {
      return NextResponse.json(
        {
          status: "error",
          code: "MISSING_INPUT",
          message: "Provide an image or a text question.",
        },
        { status: 400 }
      );
    }

    if (hasImage && (image as Blob).size > 10 * 1024 * 1024) {
      return NextResponse.json(
        {
          status: "error",
          code: "IMAGE_TOO_LARGE",
          message: "Image must be under 10 MB.",
        },
        { status: 413 }
      );
    }
  } catch {
    return NextResponse.json(
      { status: "error", code: "BAD_REQUEST", message: "Malformed form data." },
      { status: 400 }
    );
  }

  // Simulate the multi-agent pipeline latency (matches the SSE stepper timing).
  await sleep(4200);

  const id = `diag_${Math.random().toString(36).slice(2, 10)}`;
  const scenario = pickScenario(text, hasImage);
  return NextResponse.json(buildFixture(scenario, id, language));
}
