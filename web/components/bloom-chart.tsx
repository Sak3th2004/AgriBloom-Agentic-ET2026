"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TChart } from "@/lib/schemas";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";
import { useI18n } from "@/lib/i18n";

export function BloomChart({ chart }: { chart: TChart }) {
  const { t } = useI18n();

  const data = chart.days.map((d, i) => ({
    day: d,
    with: chart.with_treatment[i],
    without: chart.without_treatment[i],
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <TrendingUp className="size-5 text-primary" aria-hidden />
          {t("result.chart")}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -22 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="day"
                tick={{ fontSize: 11 }}
                stroke="hsl(var(--muted-foreground))"
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 11 }}
                stroke="hsl(var(--muted-foreground))"
              />
              <Tooltip
                contentStyle={{
                  background: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: 12,
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line
                name={t("result.chartWith")}
                dataKey="with"
                stroke="hsl(var(--success))"
                strokeWidth={2.5}
                dot={false}
              />
              <Line
                name={t("result.chartWithout")}
                dataKey="without"
                stroke="hsl(var(--destructive))"
                strokeWidth={2.5}
                strokeDasharray="6 4"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
