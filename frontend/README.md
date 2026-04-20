This frontend is designed to talk directly to the Ubuntu-hosted FastAPI backend.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## API Architecture

- Local development talks directly to `http://127.0.0.1:8000/api` unless you override it.
- Vercel production should point directly at the public HTTPS Ubuntu backend URL.
- Uploads and standard API calls use the same backend origin.

Environment variables:

```bash
# Browser-visible API base.
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api

# Optional display/base URL for the OpenAI-compatible endpoint.
# If omitted, it is derived from NEXT_PUBLIC_API_BASE_URL.
NEXT_PUBLIC_OPENAI_COMPAT_BASE_URL=http://127.0.0.1:8000/v1
```

### Local development

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api
```

### Vercel production

```bash
NEXT_PUBLIC_API_BASE_URL=https://api.your-domain.com/api
NEXT_PUBLIC_OPENAI_COMPAT_BASE_URL=https://api.your-domain.com/v1
```

When the backend runs with `SOLOLLM_PRIVATE_ACCESS_ENABLED=true`, the frontend will show the owner login screen automatically. No extra Vercel auth environment variable is required for that flow.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

Set the production environment variables above in Vercel, deploy the frontend, and make sure the Ubuntu backend is reachable over HTTPS with CORS configured for your Vercel domain.

Check out the [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for platform-specific details.
