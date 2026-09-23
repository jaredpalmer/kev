import { NextResponse, type NextRequest } from "next/server";
import type { SystemOneRequest } from "@/lib/kev";

export async function POST(req: NextRequest) {
  const apiKey = process.env.TYPESAFE_API_KEY || process.env.AI_GATEWAY_API_KEY;
  if (!apiKey) return NextResponse.json({ error: "TYPESAFE_API_KEY is not configured in .env" }, { status: 500 });

  let body: SystemOneRequest;
  try {
    body = (await req.json()) as SystemOneRequest;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const baseUrl = (process.env.TYPESAFE_API_BASE || "https://api.typesafe.ai").replace(/\/+$/, "");
  const model = body.model === "typesafe-ai/jev" || body.model === "jev" ? "jev-latest" : (body.model || "jev-latest");

  try {
    const started = performance.now();
    const r = await fetch(`${baseUrl}/v1/systemone`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
        "x-api-key": apiKey,
      },
      body: JSON.stringify({ ...body, model }),
    });

    const latency_ms = performance.now() - started;
    if (!r.ok) {
      const errText = await r.text();
      return NextResponse.json({ error: `${r.status}: ${errText}` }, { status: r.status });
    }

    const data = await r.json();
    return NextResponse.json({ ...data, latency_ms: data.latency_ms ?? latency_ms });
  } catch (err) {
    return NextResponse.json({ error: (err as Error).message }, { status: 502 });
  }
}
