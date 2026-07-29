import { NextRequest } from "next/server";

export const runtime = "nodejs";

/**
 * Mock of §6.5 GET /api/v1/files/{name}.
 * Generates a short spoken-answer placeholder tone (.wav) and a one-page
 * advisory PDF (.pdf) so the audio player and download button are fully testable.
 */

function generateWav(): Buffer {
  const sampleRate = 16000;
  const seconds = 1.6;
  const n = Math.floor(sampleRate * seconds);
  const data = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    // Gentle two-tone chime with exponential decay — placeholder for TTS audio.
    const env = Math.exp(-2.2 * t);
    const s = 0.35 * env * (Math.sin(2 * Math.PI * 523.25 * t) + 0.6 * Math.sin(2 * Math.PI * 659.25 * t));
    data.writeInt16LE(Math.round(s * 32767), i * 2);
  }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

function generatePdf(id: string): Buffer {
  const lines = [
    "AgriBloom Advisory (mock)",
    `Diagnosis ID: ${id}`,
    "Disease: Grape Downy Mildew (85% confidence)",
    "Treatment: Bordeaux mixture 1% / copper oxychloride 3 g per litre",
    "Compliance: SAFE - approved by CIB&RC for grapes",
    "Disclaimer: Confirm with your local KVK before spraying.",
  ];
  const textOps = lines
    .map((l, i) => `BT /F1 ${i === 0 ? 16 : 11} Tf 50 ${770 - i * 26} Td (${l.replace(/[()\\]/g, "")}) Tj ET`)
    .join("\n");

  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    `<< /Length ${textOps.length} >>\nstream\n${textOps}\nendstream`,
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
  ];

  let body = "%PDF-1.4\n";
  const offsets: number[] = [];
  objects.forEach((obj, i) => {
    offsets.push(body.length);
    body += `${i + 1} 0 obj\n${obj}\nendobj\n`;
  });
  const xrefStart = body.length;
  body += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const off of offsets) {
    body += `${String(off).padStart(10, "0")} 00000 n \n`;
  }
  body += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`;
  return Buffer.from(body, "latin1");
}

export async function GET(
  _req: NextRequest,
  { params }: { params: { name: string } }
) {
  const name = params.name;

  if (name.endsWith(".wav") || name.endsWith(".mp3")) {
    const wav = generateWav();
    return new Response(new Uint8Array(wav), {
      headers: {
        "Content-Type": "audio/wav",
        "Cache-Control": "public, max-age=3600",
      },
    });
  }

  if (name.endsWith(".pdf")) {
    const pdf = generatePdf(name.replace(".pdf", ""));
    return new Response(new Uint8Array(pdf), {
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="${name}"`,
        "Cache-Control": "public, max-age=3600",
      },
    });
  }

  return Response.json(
    { status: "error", code: "NOT_FOUND", message: "File not found." },
    { status: 404 }
  );
}
