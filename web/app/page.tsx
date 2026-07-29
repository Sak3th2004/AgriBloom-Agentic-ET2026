"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AlertCircle, Check, Loader2, LocateFixed, MapPin } from "lucide-react";
import { diagnose } from "@/lib/api";
import { saveDiagnosis } from "@/lib/history";
import districts from "@/lib/constants/districts.json";
import { useI18n } from "@/lib/i18n";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { ScanCard } from "@/components/scan-card";
import { MicRecorder } from "@/components/mic-recorder";
import { QuickSymptomChips } from "@/components/quick-symptom-chips";
import { AgentStepper } from "@/components/agent-stepper";

const COORDS_KEY = "agribloom.coords";

interface Coords {
  lat: number;
  lon: number;
}

export default function HomePage() {
  const { t, locale } = useI18n();
  const router = useRouter();

  const [file, setFile] = React.useState<File | null>(null);
  const [text, setText] = React.useState("");
  const [district, setDistrict] = React.useState(districts[0].name);
  const [offline, setOffline] = React.useState(false);
  const [inputError, setInputError] = React.useState(false);
  const [coords, setCoords] = React.useState<Coords | null>(null);
  const [geoStatus, setGeoStatus] = React.useState<
    "idle" | "loading" | "set" | "error"
  >("idle");

  React.useEffect(() => {
    try {
      const saved = localStorage.getItem(COORDS_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as Coords;
        if (typeof parsed.lat === "number" && typeof parsed.lon === "number") {
          setCoords(parsed);
          setGeoStatus("set");
        }
      }
    } catch {
      // ignore corrupt storage
    }
  }, []);

  const requestLocation = () => {
    if (geoStatus === "set") {
      // Tapping the active chip switches back to the district dropdown.
      setCoords(null);
      setGeoStatus("idle");
      localStorage.removeItem(COORDS_KEY);
      return;
    }
    if (!("geolocation" in navigator)) {
      setGeoStatus("error");
      return;
    }
    setGeoStatus("loading");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const next = { lat: pos.coords.latitude, lon: pos.coords.longitude };
        setCoords(next);
        setGeoStatus("set");
        localStorage.setItem(COORDS_KEY, JSON.stringify(next));
      },
      () => setGeoStatus("error"),
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 300000 }
    );
  };

  const mutation = useMutation({
    mutationFn: diagnose,
    onSuccess: (result) => {
      saveDiagnosis(result);
      router.push(`/result/${result.id}`);
    },
  });

  const submit = () => {
    if (!file && !text.trim()) {
      setInputError(true);
      return;
    }
    setInputError(false);
    const d = districts.find((x) => x.name === district) ?? districts[0];
    mutation.mutate({
      image: file,
      text: text.trim(),
      language: locale,
      lat: coords?.lat ?? d.lat,
      lon: coords?.lon ?? d.lon,
      offline,
    });
  };

  return (
    <div className="space-y-5">
      {/* Header */}
      <motion.section initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-extrabold tracking-tight">{t("app.name")}</h1>
        <p className="mt-1 text-sm font-medium text-muted-foreground">
          {t("app.tagline")}
        </p>
      </motion.section>

      <ScanCard file={file} onFile={setFile} />

      {/* Text + voice */}
      <section aria-label={t("home.orAsk")} className="space-y-2">
        <p className="text-sm font-semibold text-muted-foreground">
          {t("home.orAsk")}
        </p>
        <div className="flex items-end gap-2">
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={t("home.placeholder")}
            aria-label={t("home.orAsk")}
          />
          <MicRecorder onTranscribed={(txt) => setText(txt)} />
        </div>
      </section>

      <QuickSymptomChips onPick={(q) => setText(q)} />

      {/* Location + offline */}
      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            type="button"
            variant={geoStatus === "set" ? "default" : "outline"}
            size="sm"
            onClick={requestLocation}
            aria-pressed={geoStatus === "set"}
          >
            {geoStatus === "loading" ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : geoStatus === "set" ? (
              <Check className="size-4" aria-hidden />
            ) : (
              <LocateFixed className="size-4" aria-hidden />
            )}
            {geoStatus === "set" ? t("home.locationSet") : t("home.useLocation")}
          </Button>
          {geoStatus !== "set" && (
            <label className="flex items-center gap-2 text-sm font-semibold">
              <MapPin className="size-4 text-primary" aria-hidden />
              {t("home.district")}
              <select
                value={district}
                onChange={(e) => setDistrict(e.target.value)}
                className="h-11 cursor-pointer rounded-md border-2 border-input bg-card px-3 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {districts.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="flex items-center gap-2 text-sm font-semibold">
            {t("home.offline")}
            <Switch checked={offline} onCheckedChange={setOffline} aria-label={t("home.offline")} />
          </label>
        </div>
        {geoStatus === "error" && (
          <p className="text-xs font-semibold text-muted-foreground" role="status">
            {t("home.locationError")}
          </p>
        )}
      </section>

      {/* Errors */}
      {(inputError || mutation.isError) && (
        <div
          role="alert"
          className="flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm font-semibold text-destructive"
        >
          <AlertCircle className="size-4 shrink-0" aria-hidden />
          {inputError ? t("error.needInput") : t("common.error")}
        </div>
      )}

      {/* Submit */}
      <Button
        size="lg"
        className="w-full text-base"
        onClick={submit}
        disabled={mutation.isPending}
      >
        {mutation.isPending && <Loader2 className="animate-spin" />}
        {mutation.isPending ? t("home.analyzing") : t("home.submit")}
      </Button>

      <AgentStepper active={mutation.isPending} />
    </div>
  );
}
