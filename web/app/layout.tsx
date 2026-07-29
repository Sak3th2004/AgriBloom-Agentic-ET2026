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

const APP_URL =
  process.env.NEXT_PUBLIC_APP_URL ?? "https://agribloom.vercel.app";
const TITLE = "AgriBloom — Free AI Crop Doctor";
const DESCRIPTION =
  "Scan a leaf, ask in your language, and get safe, regulation-checked crop treatment advice. Free and open source, built for Indian farmers.";

export const metadata: Metadata = {
  metadataBase: new URL(APP_URL),
  title: TITLE,
  description: DESCRIPTION,
  manifest: "/manifest.json",
  icons: {
    icon: [
      { url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
      { url: "/icons/leaf.svg", type: "image/svg+xml" },
    ],
    apple: "/apple-touch-icon.png",
  },
  appleWebApp: { capable: true, statusBarStyle: "default", title: "AgriBloom" },
  openGraph: {
    type: "website",
    siteName: "AgriBloom",
    title: TITLE,
    description: DESCRIPTION,
    url: APP_URL,
    images: [{ url: "/og.png", width: 1200, height: 630, alt: TITLE }],
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    images: ["/og.png"],
  },
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
