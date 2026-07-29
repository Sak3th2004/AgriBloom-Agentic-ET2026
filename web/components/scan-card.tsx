"use client";

import * as React from "react";
import Image from "next/image";
import { useDropzone } from "react-dropzone";
import { Camera, ImagePlus, X } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

interface ScanCardProps {
  file: File | null;
  onFile: (file: File | null) => void;
}

/** Downscale to max 1280px JPEG so uploads work on 2G connections. */
async function compressImage(file: File): Promise<File> {
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, 1280 / Math.max(bitmap.width, bitmap.height));
    if (scale === 1 && file.size < 1_500_000) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((r) =>
      canvas.toBlob(r, "image/jpeg", 0.82)
    );
    if (!blob) return file;
    return new File([blob], "leaf.jpg", { type: "image/jpeg" });
  } catch {
    return file;
  }
}

export function ScanCard({ file, onFile }: ScanCardProps) {
  const { t } = useI18n();
  const [preview, setPreview] = React.useState<string | null>(null);
  const cameraInputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFiles = React.useCallback(
    async (files: File[]) => {
      const first = files[0];
      if (!first) return;
      onFile(await compressImage(first));
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleFiles,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    multiple: false,
    noClick: true,
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "relative overflow-hidden rounded-lg border-2 border-dashed bg-card p-5 text-center transition-colors",
        isDragActive ? "border-primary bg-primary/5" : "border-input"
      )}
      {...(getRootProps() as object)}
    >
      <input {...getInputProps()} aria-label={t("home.upload")} />
      {/* Camera capture on mobile */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void handleFiles([f]);
          e.target.value = "";
        }}
      />

      {preview ? (
        <div className="relative">
          <Image
            src={preview}
            alt="Leaf preview"
            width={640}
            height={360}
            unoptimized
            className="mx-auto max-h-64 w-auto rounded-md object-contain"
          />
          <div className="mt-3 flex justify-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => cameraInputRef.current?.click()}
            >
              <Camera /> {t("home.changePhoto")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              aria-label="Remove photo"
              onClick={() => onFile(null)}
            >
              <X />
            </Button>
          </div>
        </div>
      ) : (
        <>
          <div className="mx-auto mb-3 flex size-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <Camera className="size-8" aria-hidden />
          </div>
          <h2 className="text-lg font-bold">{t("home.scanTitle")}</h2>
          <p className="mx-auto mt-1 max-w-xs text-sm text-muted-foreground">
            {t("home.scanHint")}
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-3">
            <Button size="lg" onClick={() => cameraInputRef.current?.click()}>
              <Camera /> {t("home.takePhoto")}
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => {
                // open the dropzone file dialog
                const input = document.querySelector<HTMLInputElement>(
                  'input[type="file"]:not([capture])'
                );
                input?.click();
              }}
            >
              <ImagePlus /> {t("home.upload")}
            </Button>
          </div>
        </>
      )}
    </motion.div>
  );
}
