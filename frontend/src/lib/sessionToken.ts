/** Per-session access tokens.
 *
 * The server issues one at upload and requires it on every request scoped to that session,
 * so knowing a session id isn't enough to read the data behind it.
 *
 * Held in sessionStorage rather than localStorage: a session is a single piece of work in a
 * single tab, and leaving the token behind after the tab closes would keep access alive
 * longer than the thing it protects. The consequence is deliberate — reopening the URL in a
 * new browser doesn't get you back in, which is the point.
 *
 * A project the owner chose to keep (#120) is the exception: its token moves to localStorage,
 * with the project's name, so the upload page can list it and it opens again after the tab
 * closes. The token is also shown to the owner once, to save, because it is the only way back
 * in from another browser.
 */

const PREFIX = "dataclean.session.";

export function storeSessionToken(sessionId: string, token: string): void {
  if (!token) return;
  try {
    sessionStorage.setItem(PREFIX + sessionId, token);
  } catch {
    // Private browsing or a storage quota. The request will simply be unauthenticated,
    // which the server answers with a readable 403 rather than anything worse.
  }
}

export function getSessionToken(sessionId: string | null | undefined): string | null {
  if (!sessionId) return null;
  try {
    return sessionStorage.getItem(PREFIX + sessionId) ?? localStorage.getItem(KEPT_PREFIX + sessionId);
  } catch {
    return null;
  }
}

const KEPT_PREFIX = "dataclean.project.";
const PROJECTS_KEY = "dataclean.projects";

export interface KeptProject {
  id: string;
  name: string;
  keptAt: string | null;
}

export function listKeptProjects(): KeptProject[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(PROJECTS_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((p) => p && typeof p.id === "string") : [];
  } catch {
    return [];
  }
}

/** Remembers a kept project in this browser: its token, so it opens after the tab closes, and its name for the list. */
export function rememberKeptProject(project: KeptProject, token: string): void {
  try {
    localStorage.setItem(KEPT_PREFIX + project.id, token);
    const others = listKeptProjects().filter((p) => p.id !== project.id);
    localStorage.setItem(PROJECTS_KEY, JSON.stringify([project, ...others]));
  } catch {
    // Storage blocked: the project is still kept on the server; the owner has the key they were shown.
  }
}

/** Replaces the token wherever this browser holds it (after the key is rotated). */
export function replaceSessionToken(sessionId: string, token: string): void {
  try {
    if (sessionStorage.getItem(PREFIX + sessionId) !== null) sessionStorage.setItem(PREFIX + sessionId, token);
    if (localStorage.getItem(KEPT_PREFIX + sessionId) !== null) localStorage.setItem(KEPT_PREFIX + sessionId, token);
  } catch {
    // As above.
  }
}

/** Forgets a project or session in this browser: its token and its place in the list. */
export function forgetSession(sessionId: string): void {
  try {
    sessionStorage.removeItem(PREFIX + sessionId);
    localStorage.removeItem(KEPT_PREFIX + sessionId);
    localStorage.setItem(PROJECTS_KEY, JSON.stringify(listKeptProjects().filter((p) => p.id !== sessionId)));
  } catch {
    // As above.
  }
}

/** Header to attach to any session-scoped request. Empty when there's no token, so calls
 * still work against a development server that isn't enforcing them. */
export function sessionHeaders(sessionId: string | null | undefined): Record<string, string> {
  const token = getSessionToken(sessionId);
  return token ? { "X-Session-Token": token } : {};
}
