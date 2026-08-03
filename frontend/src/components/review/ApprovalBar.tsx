import { Button } from "@/components/ui/button";

interface ApprovalBarProps {
  total: number;
  approvedCount: number;
  rejectedCount: number;
  undecidedCount: number;
  submitting: boolean;
  disabled: boolean;
  onSubmit: () => void;
}

/** Sticky footer summarizing decisions and submitting them to the cleaning run.
 * Undecided cards are treated as rejected by the backend (only explicitly approved ids are
 * applied), so the count is surfaced here rather than silently ignored. */
export function ApprovalBar({
  total,
  approvedCount,
  rejectedCount,
  undecidedCount,
  submitting,
  disabled,
  onSubmit,
}: ApprovalBarProps) {
  if (total === 0) return null;

  return (
    <div className="sticky bottom-0 -mx-8 border-t border-border bg-background/95 px-8 py-4 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm">
          <span className="font-medium">{approvedCount}</span> approved &middot;{" "}
          <span className="font-medium">{rejectedCount}</span> rejected
          {undecidedCount > 0 && (
            <span className="text-muted-foreground">
              {" "}
              &middot; {undecidedCount} undecided (will be skipped)
            </span>
          )}
        </div>
        <Button onClick={onSubmit} disabled={disabled || submitting}>
          {submitting
            ? "Applying fixes..."
            : approvedCount === 0
              ? "Skip all and continue"
              : `Apply ${approvedCount} approved fix${approvedCount === 1 ? "" : "es"}`}
        </Button>
      </div>
    </div>
  );
}
