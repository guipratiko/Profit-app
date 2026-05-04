import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

function getBackendBaseUrl() {
  return process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "";
}

function buildBackendUrl(request: NextRequest, path: string[]) {
  const backendBaseUrl = getBackendBaseUrl();
  if (!backendBaseUrl) {
    throw new Error("Defina NEXT_PUBLIC_API_BASE_URL com a URL publica do backend FastAPI antes de publicar o frontend.");
  }

  const targetUrl = new URL(backendBaseUrl);
  const joinedPath = path.map(encodeURIComponent).join("/");
  targetUrl.pathname = `${targetUrl.pathname.replace(/\/$/, "")}/${joinedPath}`;
  targetUrl.search = request.nextUrl.search;
  return targetUrl;
}

async function proxyRequest(request: NextRequest, context: RouteContext) {
  try {
    const { path } = await context.params;
    const targetUrl = buildBackendUrl(request, path);
    const upstreamHeaders = new Headers(request.headers);

    upstreamHeaders.set("bypass-tunnel-reminder", "1");
    upstreamHeaders.delete("host");

    const upstreamResponse = await fetch(targetUrl, {
      method: request.method,
      headers: upstreamHeaders,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer(),
      cache: "no-store",
      redirect: "follow",
    });

    const responseHeaders = new Headers(upstreamResponse.headers);
    responseHeaders.delete("content-encoding");
    responseHeaders.delete("content-length");
    responseHeaders.delete("transfer-encoding");
    return new Response(await upstreamResponse.arrayBuffer(), {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Falha ao conectar com o backend FastAPI.";
    return Response.json({ detail: message }, { status: 502 });
  }
}

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PUT = proxyRequest;
export const DELETE = proxyRequest;
export const PATCH = proxyRequest;