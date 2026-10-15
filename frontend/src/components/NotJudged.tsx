import type { ReactNode } from "react";
import type { NotJudgedBlock } from "@/api/types";

/** The one treatment for a state the app declines to judge: the verdict on a small test set, drift on
 * a small file, a check against a few real answers, a cut-off that could not be chosen.
 *
 * Sky rather than red or amber on purpose: "cannot tell" is not "it is bad", and colouring it as a
 * failure would push a reader towards the wrong conclusion as hard as calling it fine. The sentence
 * (written on the server, always "Not enough … to judge: N rows, at least M needed." or "Cannot judge:
 * …") leads; whatever else the surface has to say follows it. */
export function NotJudged({ block, children }: { block: NotJudgedBlock; children?: ReactNode }) {
  return (
    <div role="status" aria-label="Cannot judge" className="rounded-md border border-sky-500/40 bg-sky-500/5 p-3 text-sm">
      <p className="font-medium">{block.sentence}</p>
      {children && <div className="mt-1 text-muted-foreground">{children}</div>}
    </div>
  );
}
