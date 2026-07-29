export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Mock of §6.3 GET /api/v1/diagnose/stream — SSE agent-progress events. */
export async function GET() {
  const steps = [
    { stage: "orchestrator", message: "Understanding your question", progress: 0.15, delay: 500 },
    { stage: "vision", message: "Analyzing image", progress: 0.35, delay: 900 },
    { stage: "knowledge", message: "Finding treatment", progress: 0.6, delay: 1100 },
    { stage: "compliance", message: "Checking safety rules", progress: 0.8, delay: 900 },
    { stage: "output", message: "Preparing advice", progress: 0.95, delay: 700 },
  ];

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      for (const s of steps) {
        await new Promise((r) => setTimeout(r, s.delay));
        controller.enqueue(
          encoder.encode(
            `event: step\ndata: ${JSON.stringify({
              stage: s.stage,
              message: s.message,
              progress: s.progress,
            })}\n\n`
          )
        );
      }
      controller.enqueue(encoder.encode(`event: done\ndata: {}\n\n`));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
