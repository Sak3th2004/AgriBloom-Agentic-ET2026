"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AlertCircle, Loader2, MapPin } from "lucide-react";
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

export default function HomePage() {
  const { t, locale } = useI18n();
  const router = useRouter();

  const [file, setFile] = React.useState<File | null>(null);
  const [text, setText] = React.useState("");
  const [district, setDistrict] = React.useState(districts[0].name);
  const [offline, setOffline] = React.useState(false);
  const [inputError, setInputError] = React.useState(false);

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
      lat: d.lat,
      lon: d.lon,
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

      {/* District + offline */}
      <section className="flex flex-wrap items-center gap-x-6 gap-y-3">
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
        <label className="flex items-center gap-2 text-sm font-semibold">
          {t("home.offline")}
          <Switch checked={offline} onCheckedChange={setOffline} aria-label={t("home.offline")} />
        </label>
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
