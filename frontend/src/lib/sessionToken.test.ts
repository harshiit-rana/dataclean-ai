import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  forgetSession,
  getSessionToken,
  listKeptProjects,
  rememberKeptProject,
  replaceSessionToken,
  sessionHeaders,
  storeSessionToken,
} from "./sessionToken";

describe("session tokens", () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
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

  it("finds a kept project's token after the tab's own storage is gone (#120)", () => {
    rememberKeptProject({ id: "p1", name: "Churn", keptAt: null }, "tok-p1");
    sessionStorage.clear();
    expect(getSessionToken("p1")).toBe("tok-p1");
    expect(listKeptProjects()).toEqual([{ id: "p1", name: "Churn", keptAt: null }]);
  });

  it("replaces a rotated token wherever it is held, and nowhere it is not", () => {
    storeSessionToken("s1", "old");
    rememberKeptProject({ id: "s1", name: "One", keptAt: null }, "old");
    replaceSessionToken("s1", "new");
    expect(sessionStorage.getItem("dataclean.session.s1")).toBe("new");
    expect(localStorage.getItem("dataclean.project.s1")).toBe("new");
    replaceSessionToken("s2", "new");
    expect(getSessionToken("s2")).toBeNull();
  });

  it("forgets a deleted project completely", () => {
    storeSessionToken("s1", "tok");
    rememberKeptProject({ id: "s1", name: "One", keptAt: null }, "tok");
    forgetSession("s1");
    expect(getSessionToken("s1")).toBeNull();
    expect(listKeptProjects()).toEqual([]);
  });

  it("reads a damaged list as empty rather than failing", () => {
    localStorage.setItem("dataclean.projects", "{ not json");
    expect(listKeptProjects()).toEqual([]);
  });
});
