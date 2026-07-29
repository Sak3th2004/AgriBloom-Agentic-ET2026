import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { Providers } from "@/components/providers";
import { TopBar } from "@/components/top-bar";
import { BottomNav } from "@/components/bottom-nav";
import { OfflineBanner } from "@/components/offline-banner";
import { SwRegister } from "@/components/sw-register";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "AgriBloom — Free AI Crop Doctor",
  description:
    "Scan a leaf, ask in your language, and get safe, regulation-checked crop treatment advice. Free and open source, built for Indian farmers.",
  manifest: "/manifest.json",
  icons: { icon: "/icons/leaf.svg" },
  appleWebApp: { capable: true, statusBarStyle: "default", title: "AgriBloom" },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#14532d" },
    { media: "(prefers-color-scheme: dark)", color: "#0b1512" },
  ],
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} font-sans antialiased`}>
        <Providers>
          <OfflineBanner />
          <TopBar />
          <main className="mx-auto w-full max-w-3xl px-4 pb-24 pt-4 md:pb-10">
            {children}
          </main>
          <BottomNav />
          <SwRegister />
        </Providers>
      </body>
    </html>
  );
}
