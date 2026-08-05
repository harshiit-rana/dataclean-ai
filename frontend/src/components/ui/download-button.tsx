import { useState } from "react";
import { Button, type ButtonProps } from "@/components/ui/button";
import { ApiError, downloadSessionFile } from "@/api/client";

// `onError` is also a DOM handler name, so it has to be excluded from what's inherited --
// here it means "the download failed", not the <button>'s error event.
interface DownloadButtonProps extends Omit<ButtonProps, "onClick" | "onError"> {
  sessionId: string;
  /** API path, e.g. `/api/export/abc/zip`. */
  path: string;
  filename: string;
  children: React.ReactNode;
  onError?: (message: string) => void;
}

/** A download that carries the session's access token.
 *
 * Replaces `<a href download>` throughout: a link can't set a header, and the token must not
 * go in the URL. Renders as a button so the whole app's downloads behave and look the same. */
export function DownloadButton({
  sessionId,
  path,
  filename,
  children,
  onError,
  ...buttonProps
}: DownloadButtonProps) {
  const [busy, setBusy] = useState(false);

  const handleClick = async () => {
    setBusy(true);
    try {
      await downloadSessionFile(sessionId, path, filename);
    } catch (err) {
      onError?.(
        err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err)
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Button {...buttonProps} onClick={handleClick} disabled={busy || buttonProps.disabled}>
      {busy ? "Preparing..." : children}
    </Button>
  );
}
