"use client";

import * as React from "react";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, Bot, Loader2, SendHorizonal, User } from "lucide-react";
import { chat } from "@/lib/api";
import { getDiagnosis } from "@/lib/history";
import { useI18n } from "@/lib/i18n";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatPage() {
  const { t, locale } = useI18n();
  const params = useParams<{ session: string }>();
  const search = useSearchParams();
  const diagId = search.get("diag");

  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState("");
  const bottomRef = React.useRef<HTMLDivElement>(null);
  const [diseaseName, setDiseaseName] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (diagId) {
      const diag = getDiagnosis(diagId);
      if (diag?.disease) setDiseaseName(diag.disease.display_name);
    }
  }, [diagId]);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const mutation = useMutation({
    mutationFn: (question: string) =>
      chat({
        session_id: params.session,
        question,
        language: locale,
        history: messages,
      }),
    onSuccess: (res) => {
      setMessages((m) => [...m, { role: "assistant", content: res.answer }]);
    },
  });

  const send = () => {
    const q = input.trim();
    if (!q || mutation.isPending) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");
    mutation.mutate(q);
  };

  return (
    <div className="flex min-h-[70vh] flex-col">
      <div className="mb-3 flex items-center gap-2">
        {diagId && (
          <Link href={`/result/${diagId}`} aria-label="Back">
            <Button variant="ghost" size="icon">
              <ArrowLeft />
            </Button>
          </Link>
        )}
        <div>
          <h1 className="text-lg font-extrabold">{t("chat.title")}</h1>
          {diseaseName && (
            <p className="text-xs font-semibold text-muted-foreground">{diseaseName}</p>
          )}
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto rounded-lg border bg-card/50 p-4">
        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className={cn("flex gap-2", m.role === "user" && "flex-row-reverse")}
            >
              <span
                className={cn(
                  "flex size-8 shrink-0 items-center justify-center rounded-full",
                  m.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-secondary-foreground"
                )}
              >
                {m.role === "user" ? <User className="size-4" /> : <Bot className="size-4" />}
              </span>
              <p
                className={cn(
                  "max-w-[80%] rounded-lg px-4 py-2.5 text-sm leading-relaxed shadow-sm",
                  m.role === "user"
                    ? "rounded-tr-sm bg-primary text-primary-foreground"
                    : "rounded-tl-sm border bg-card"
                )}
              >
                {m.content}
              </p>
            </motion.div>
          ))}
        </AnimatePresence>

        {mutation.isPending && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" aria-hidden />
            {t("common.loading")}
          </div>
        )}
        {mutation.isError && (
          <p role="alert" className="text-sm font-semibold text-destructive">
            {t("common.error")}
          </p>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="mt-3 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder={t("chat.placeholder")}
          aria-label={t("chat.placeholder")}
          className="h-12 flex-1 rounded-lg border-2 border-input bg-card px-4 text-base shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <Button size="icon" className="size-12" onClick={send} aria-label={t("chat.send")}>
          <SendHorizonal />
        </Button>
      </div>
    </div>
  );
}
