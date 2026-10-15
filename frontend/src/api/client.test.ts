import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, downloadSessionFile, fetchModelPlan, fetchPreview, uploadCsv } from "./client";
import { getSessionToken, storeSessionToken } from "@/lib/sessionToken";

/** The guarantee under test is narrow but easy to break: every call scoped to a session has
 * to carry the access token, and the token must never end up in a URL -- query strings land
 * in server logs, browser history and referrers, which is exactly where it can't be. */

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function lastCall(): [string, RequestInit | undefined] {
  const mock = vi.mocked(globalThis.fetch);
  const [url, init] = mock.mock.calls[mock.mock.calls.length - 1];
  return [String(url), init as RequestInit | undefined];
}

function headerFrom(init: RequestInit | undefined): string | undefined {
  return (init?.headers as Record<string, string> | undefined)?.["X-Session-Token"];
}

describe("session-scoped requests", () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps the token handed back by upload", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({
        session_id: "s1",
        session_token: "tok-1",
        file_name: "data.csv",
        row_count: 2,
        column_count: 2,
        columns: ["a", "b"],
      })
    );

    await uploadCsv(new File(["a,b\n1,2\n"], "data.csv", { type: "text/csv" }));
    expect(getSessionToken("s1")).toBe("tok-1");
  });

  it("attaches the token to a session-scoped GET", async () => {
    storeSessionToken("s1", "tok-1");
    vi.mocked(globalThis.fetch).mockResolvedValue(jsonResponse({ rows: [] }));

    await fetchPreview("s1");
    const [url, init] = lastCall();
    expect(headerFrom(init)).toBe("tok-1");
    expect(url).not.toContain("tok-1");
  });

  it("attaches the token to a download, and not to the URL", async () => {
    storeSessionToken("s1", "tok-1");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response("col\n1\n", { status: 200, headers: { "Content-Type": "text/csv" } })
    );
    // jsdom implements neither, and the click would otherwise try to navigate.
    vi.stubGlobal("URL", { ...URL, createObjectURL: () => "blob:x", revokeObjectURL: () => {} });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    await downloadSessionFile("s1", "/api/export/s1/csv", "cleaned.csv");

    const [url, init] = lastCall();
    expect(headerFrom(init)).toBe("tok-1");
    expect(url).not.toContain("tok-1");
    expect(click).toHaveBeenCalled();
  });

  it("turns a refused download into a readable error, not a downloaded error page", async () => {
    storeSessionToken("s1", "wrong");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(JSON.stringify({ error: "Not your session", detail: "..." }), {
        status: 403,
        headers: { "Content-Type": "application/json" },
      })
    );

    await expect(
      downloadSessionFile("s1", "/api/export/s1/csv", "cleaned.csv")
    ).rejects.toMatchObject({ status: 403, error: "Not your session" });
  });

  it("sends no token header at all when there isn't one", async () => {
    // A development server isn't enforcing tokens; those requests still have to work.
    vi.mocked(globalThis.fetch).mockResolvedValue(jsonResponse({ rows: [] }));

    await fetchPreview("s1");
    expect(headerFrom(lastCall()[1])).toBeUndefined();
  });
});


describe("an error response", () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => vi.unstubAllGlobals());

  const refuse = (body: unknown) =>
    new Response(JSON.stringify(body), { status: 422, headers: { "Content-Type": "application/json" } });

  it("keeps the reason code the server gave", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      refuse({ error: "Model could not be trained", detail: "Blocked.", code: "data_blocked" })
    );
    const thrown = await fetchModelPlan("s1").catch((err: unknown) => err);
    expect(thrown).toBeInstanceOf(ApiError);
    expect((thrown as ApiError).code).toBe("data_blocked");
  });

  it("has no code when the server gave none", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(refuse({ error: "Nope", detail: "Because." }));
    const thrown = await fetchModelPlan("s1").catch((err: unknown) => err);
    expect((thrown as ApiError).code).toBeUndefined();
  });
});
