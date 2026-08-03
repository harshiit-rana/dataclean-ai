interface AgentStatusBarProps {
  status: "idle" | "streaming" | "interrupted" | "error";
  currentAgent: string | null;
  statusMessage: string;
}

const AGENT_LABELS: Record<string, string> = {
  goal_intake: "Reviewing your goal",
  profiling_agent: "Profiling your data",
  explanation_agent: "Writing suggestions",
};

/** PRD NFR-16: the user should always know which agent is currently working. */
export function AgentStatusBar({ status, currentAgent, statusMessage }: AgentStatusBarProps) {
  if (status === "idle") return null;

  const label =
    status === "interrupted"
      ? "Ready for your review"
      : status === "error"
        ? "Something went wrong"
        : (currentAgent && AGENT_LABELS[currentAgent]) || statusMessage || "Working...";

  return (
    <div className="flex items-center gap-2 rounded-md border border-border bg-muted px-4 py-2 text-sm">
      {status === "streaming" && (
        <span className="h-2 w-2 animate-pulse rounded-full bg-primary" aria-hidden />
      )}
      <span>{label}</span>
    </div>
  );
}
