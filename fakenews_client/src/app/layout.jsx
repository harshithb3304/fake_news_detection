import "@/app/globals.css";
import { Inter } from "next/font/google";
//import { ThemeProvider } from "@/components/theme-provider";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Fake News Detector",
  description: "Detect fake news articles using AI",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <main className="min-h-screen bg-background">{children}</main>
      </body>
    </html>
  );
}

