// One-off asset generator: rasterizes the SVG brand assets into the PNG
// sizes required for PWA install + social link previews.
// Run: node scripts/generate-icons.mjs
import sharp from "sharp";
import { mkdir } from "node:fs/promises";

const jobs = [
  { src: "public/icons/leaf.svg", out: "public/icons/icon-192.png", w: 192, h: 192 },
  { src: "public/icons/leaf.svg", out: "public/icons/icon-512.png", w: 512, h: 512 },
  { src: "public/icons/leaf-maskable.svg", out: "public/icons/icon-maskable-512.png", w: 512, h: 512 },
  { src: "public/icons/leaf.svg", out: "public/apple-touch-icon.png", w: 180, h: 180 },
  { src: "public/og.svg", out: "public/og.png", w: 1200, h: 630 },
];

await mkdir("public/icons", { recursive: true });
for (const { src, out, w, h } of jobs) {
  await sharp(src, { density: 300 }).resize(w, h).png().toFile(out);
  console.log(`wrote ${out} (${w}x${h})`);
}
