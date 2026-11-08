import type { Metadata, Viewport } from "next";
import { Be_Vietnam_Pro } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const beVietnam = Be_Vietnam_Pro({
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-be-vietnam",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Trợ lý pháp luật thuế",
  description: "Hỏi đáp pháp luật thuế Việt Nam có trích dẫn nguồn (RAG).",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

const NAV = [
  { href: "/", label: "Trò chuyện" },
  { href: "/search", label: "Tìm kiếm" },
  { href: "/admin", label: "Quản trị" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" className={beVietnam.variable}>
      <body className="font-sans">
        <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b border-slate-200 bg-white/90 px-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
          <Link href="/" className="text-base font-semibold">
            Trợ lý pháp luật thuế
          </Link>
          <nav className="flex gap-1 text-sm" aria-label="Điều hướng chính">
            {NAV.map((n) => (
              <Link
                key={n.href}
                href={n.href}
                className="rounded-md px-2 py-1 text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
              >
                {n.label}
              </Link>
            ))}
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
