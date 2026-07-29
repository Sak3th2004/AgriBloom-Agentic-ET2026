"use client";

import * as React from "react";
import { Loader2, Mic, Square } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { transcribe } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

interface MicRecorderProps {
  onTranscribed: (text: string) => void;
}

export function MicRecorder({ onTranscribed }: MicRecorderProps) {
  const { t, locale } = useI18n();
  const [recording, setRecording] = React.useState(false);
  const recorderRef = React.useRef<MediaRecorder | null>(null);
  const chunksRef = React.useRef<Blob[]>([]);

  const mutation = useMutation({
    mutationFn: (blob: Blob) => transcribe(blob, locale),
    onSuccess: (res) => onTranscribed(res.text),
  });

  const stop = React.useCallback(() => {
    recorderRef.current?.stop();
    recorderRef.current?.stream.getTracks().forEach((tr) => tr.stop());
    setRecording(false);
  }, []);

  const start = React.useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        if (blob.size > 0) mutation.mutate(blob);
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch {
      // mic permission denied — leave text input as the path
    }
  }, [mutation]);

  React.useEffect(() => () => stop(), [stop]);

  return (
    <Button
      type="button"
      variant={recording ? "destructive" : "secondary"}
      size="icon"
      aria-label={recording ? t("home.listening") : t("home.speak")}
      title={recording ? t("home.listening") : t("home.speak")}
      onClick={recording ? stop : start}
      disabled={mutation.isPending}
      className={cn("shrink-0 rounded-full", recording && "animate-pulse")}
    >
      {mutation.isPending ? (
        <Loader2 className="animate-spin" />
      ) : recording ? (
        <Square />
      ) : (
        <Mic />
      )}
    </Button>
  );
}
