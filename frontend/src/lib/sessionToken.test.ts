import { beforeEach, describe, expect, it, vi } from "vitest";
import { getSessionToken, sessionHeaders, storeSessionToken } from "./sessionToken";

describe("session tokens", () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it("gives back the token it was given", () => {
    storeSessionToken("abc", "tok-abc");
    expect(getSessionToken("abc")).toBe("tok-abc");
  });

  it("keeps sessions apart", () => {
    storeSessionToken("abc", "tok-abc");
    storeSessionToken("def", "tok-def");
    expect(sessionHeaders("def")).toEqual({ "X-Session-Token": "tok-def" });
  });

  it("has no token for a session it never saw", () => {
    expect(getSessionToken("unknown")).toBeNull();
    expect(sessionHeaders("unknown")).toEqual({});
  });

  it("has no token for a missing session id", () => {
    expect(sessionHeaders(null)).toEqual({});
    expect(sessionHeaders(undefined)).toEqual({});
  });

  it("sends no header rather than an empty one", () => {
    // A blank X-Session-Token would read as "I have a token and it's wrong". Sending nothing
    // is what a development server that isn't enforcing tokens expects.
    storeSessionToken("abc", "");
    expect(sessionHeaders("abc")).toEqual({});
  });

  it("survives storage being unavailable", () => {
    // Private browsing throws on setItem. The request then goes out unauthenticated and the
    // server answers with a readable 403 -- what must not happen is the page crashing here.
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("denied");
    });
    expect(() => storeSessionToken("abc", "tok-abc")).not.toThrow();

    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("denied");
    });
    expect(getSessionToken("abc")).toBeNull();
  });
});
