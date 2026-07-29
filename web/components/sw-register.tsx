"use client";

import * as React from "react";

export function SwRegister() {
  React.useEffect(() => {
    if ("serviceWorker" in navigator && process.env.NODE_ENV === "production") {
      navigator.serviceWorker.register("/sw.js").catch(() => {
        // PWA is progressive enhancement — ignore registration failures
      });
    }
  }, []);
  return null;
}
