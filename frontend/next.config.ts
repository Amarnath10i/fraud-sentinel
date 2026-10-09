import type { NextConfig } from "next";

// `NEXT_OUTPUT=export next build` produces a static site in ./out that the
// FastAPI service serves from the same origin (the public demo). Plain
// `next build` / `next dev` stay a normal Next.js app.
const isExport = process.env.NEXT_OUTPUT === "export";

const nextConfig: NextConfig = isExport ? { output: "export", trailingSlash: true } : {};

export default nextConfig;
