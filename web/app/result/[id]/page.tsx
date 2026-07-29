"use client";

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import {
  CloudSun,
  Download,
  Droplets,
  IndianRupee,
  Leaf,
  ListChecks,
  MessageCircleQuestion,
  Pill,
  RefreshCcw,
  Share2,
  Sprout,
  ThermometerSun,
  TriangleAlert,
} from "lucide-react";
import type { TDiagnoseResponse } from "@/lib/schemas";
import { getDiagnosis } from "@/lib/history";
import { fileUrl } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfidenceMeter } from "@/components/confidence-meter";
import { ComplianceCard } from "@/components/compliance-card";
import { BloomChart } from "@/components/bloom-chart";
import { AudioPlayer } from "@/components/audio-player";

export default function ResultPage() {
  const { t } = useI18n();
  const params = useParams<{ id: string }>();
  const [result, setResult] = React.useState<TDiagnoseResponse | null | undefined>(undefined);

  React.useEffect(() => {
    setResult(getDiagnosis(params.id));
  }, [params.id]);

  if (result === undefined) return null;

  if (result === null) {
    return (
      <div className="flex flex-col items-center gap-4 py-16 text-center">
        <TriangleAlert className="size-10 text-muted-foreground" aria-hidden />
        <p className="font-semibold text-muted-foreground">{t("result.notFound")}</p>
        <Button onClick={() => (window.location.href = "/")}>
          <RefreshCcw /> {t("result.newScan")}
        </Button>
      </div>
    );
  }

  const audio = fileUrl(result.audio_url);
  const pdf = fileUrl(result.pdf_url);
  const shareText = result.disease
    ? `AgriBloom: ${result.disease.display_name} (${Math.round(result.disease.confidence * 100)}%). ${result.treatment ?? ""}`.slice(0, 300)
    : "AgriBloom crop advisory";

  const invalidImage = result.status === "invalid_image";
  const uncertain = result.disease?.is_uncertain || result.status === "uncertain";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-extrabold tracking-tight">{t("result.title")}</h1>
        <Link href="/" className="text-sm font-semibold text-primary hover:underline">
          + {t("result.newScan")}
        </Link>
      </div>

      {/* Retake tip for unreadable photos */}
      {invalidImage && (
        <Card className="border-2 border-accent/60 bg-accent/5" role="alert">
          <CardHeader>
            <CardTitle className="text-accent-foreground dark:text-accent">
              <TriangleAlert className="size-5" aria-hidden /> {t("result.invalidImage")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-inside list-disc space-y-1 text-sm text-muted-foreground">
              {result.recommendations.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Diagnosis */}
      {result.disease && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <Card>
            <CardHeader>
              <div className="flex items-start justify-between gap-2">
                <CardTitle className="text-lg">
                  <Leaf className="size-5 text-primary" aria-hidden />
                  {result.disease.display_name}
                </CardTitle>
                <Badge variant="default" className="capitalize">
                  <Sprout aria-hidden /> {result.disease.crop}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <ConfidenceMeter value={result.disease.confidence} />
              {uncertain && (
                <p className="flex items-start gap-2 rounded-md bg-accent/10 px-3 py-2 text-sm font-semibold text-accent-foreground dark:text-accent">
                  <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
                  {t("result.uncertain")}
                </p>
              )}
              <div className="flex flex-wrap gap-1.5">
                {result.disease.top3.slice(1).map((alt) => (
                  <Badge key={alt.label} variant="secondary">
                    {alt.display_name} · {Math.round(alt.confidence * 100)}%
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Safety — loud, right below the verdict */}
      {result.compliance && <ComplianceCard compliance={result.compliance} />}

      {/* Treatment */}
      {result.treatment && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <Card>
            <CardHeader>
              <CardTitle>
                <Pill className="size-5 text-primary" aria-hidden />
                {t("result.treatment")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed">{result.treatment}</p>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Recommendations */}
      {!invalidImage && result.recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>
              <ListChecks className="size-5 text-primary" aria-hidden />
              {t("result.recommendations")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {result.recommendations.map((r) => (
                <li key={r} className="flex items-start gap-2 text-sm">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
                  {r}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Weather + market */}
      {result.knowledge && (result.knowledge.weather || result.knowledge.market) && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {result.knowledge.weather && (
            <Card>
              <CardHeader>
                <CardTitle>
                  <CloudSun className="size-5 text-primary" aria-hidden />
                  {t("result.weather")}
                </CardTitle>
              </CardHeader>
              <CardContent className="flex items-center gap-4 text-sm">
                <span className="flex items-center gap-1.5 font-bold">
                  <ThermometerSun className="size-4 text-accent" aria-hidden />
                  {result.knowledge.weather.temp_c}°C
                </span>
                <span className="flex items-center gap-1.5">
                  <Droplets className="size-4 text-primary" aria-hidden />
                  {result.knowledge.weather.humidity}%
                </span>
                <span className="text-muted-foreground">{result.knowledge.weather.desc}</span>
              </CardContent>
            </Card>
          )}
          {result.knowledge.market && (
            <Card>
              <CardHeader>
                <CardTitle>
                  <IndianRupee className="size-5 text-primary" aria-hidden />
                  {t("result.market")}
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm">
                <p className="text-xl font-extrabold">
                  ₹{result.knowledge.market.modal_price.toLocaleString("en-IN")}
                  <span className="text-sm font-medium text-muted-foreground">
                    {" "}/ {result.knowledge.market.unit}
                  </span>
                </p>
                <p className="capitalize text-muted-foreground">
                  {result.knowledge.market.crop} · {result.knowledge.market.mandi}
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Bloom recovery chart */}
      {result.chart && <BloomChart chart={result.chart} />}

      {/* Actions */}
      <div className="flex flex-wrap gap-2">
        {audio && <AudioPlayer src={audio} />}
        {pdf && (
          <Button variant="outline" onClick={() => window.open(pdf, "_blank")}>
            <Download /> {t("result.pdf")}
          </Button>
        )}
        <Button
          variant="outline"
          onClick={() =>
            window.open(`https://wa.me/?text=${encodeURIComponent(shareText)}`, "_blank")
          }
        >
          <Share2 /> {t("result.share")}
        </Button>
        <Link href={`/chat/${result.session_id}?diag=${result.id}`} className="flex-1 sm:flex-none">
          <Button variant="secondary" className="w-full">
            <MessageCircleQuestion /> {t("result.followup")}
          </Button>
        </Link>
      </div>
    </div>
  );
}
