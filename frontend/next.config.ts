import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: { 
    // Allow larger proxied request bodies in dev/proxy mode.
    // Large file uploads should still prefer direct backend upload.
    proxyClientMaxBodySize: "100mb",
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
