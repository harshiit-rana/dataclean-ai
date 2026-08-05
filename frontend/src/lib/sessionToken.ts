/** Per-session access tokens.
 *
 * The server issues one at upload and requires it on every request scoped to that session,
 * so knowing a session id isn't enough to read the data behind it.
 *
 * Held in sessionStorage rather than localStorage: a session is a single piece of work in a
 * single tab, and leaving the token behind after the tab closes would keep access alive
 * longer than the thing it protects. The consequence is deliberate — reopening the URL in a
 * new browser doesn't get you back in, which is the point.
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
    return sessionStorage.getItem(PREFIX + sessionId);
  } catch {
    return null;
  }
}

/** Header to attach to any session-scoped request. Empty when there's no token, so calls
 * still work against a development server that isn't enforcing them. */
export function sessionHeaders(sessionId: string | null | undefined): Record<string, string> {
  const token = getSessionToken(sessionId);
  return token ? { "X-Session-Token": token } : {};
}
