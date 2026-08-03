import { Badge } from "@/components/ui/badge";

interface IssueCountSummaryProps {
  counts: { high: number; medium: number; low: number } | null;
}

export function IssueCountSummary({ counts }: IssueCountSummaryProps) {
  if (!counts) return null;

  return (
    <div className="flex items-center gap-3">
      <Badge variant="high">{counts.high} high</Badge>
      <Badge variant="medium">{counts.medium} medium</Badge>
      <Badge variant="low">{counts.low} low</Badge>
    </div>
  );
}
