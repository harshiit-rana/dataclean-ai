import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CleanPage } from "./CleanPage";
import type { ActionLogEntry } from "@/api/client";

/** Recovering a session after a reload.
 *
 * The approved decisions arrive in navigation state, so they are gone the moment the page is
 * refreshed — and this page used to answer that with "start from the review step", stranding
 * someone who had merely pressed F5. It became a functional gap rather than a cosmetic one
 * when the business rules panel moved here: a reload left approved rules with no route back
 * to them at all. Found by reloading the page in a browser; nothing in the suite noticed.
 */

const fetchActionLog = vi.fn();

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>("@/api/client");
  return { ...actual, fetchActionLog: (id: string) => fetchActionLog(id) };
});

// The panels below the log each fetch on mount; none of that is what these tests are about.
vi.mock("@/components/rules/RulesSection", () => ({
  RulesSection: () => <div>rules panel</div>,
}));
vi.mock("@/components/readiness/ReadinessPanel", () => ({
  ReadinessPanel: () => <div>readiness panel</div>,
}));
vi.mock("@/components/clean/ChatPanel", () => ({ ChatPanel: () => <div>chat panel</div> }));
vi.mock("@/hooks/useCleaningRun", () => ({
  useCleaningRun: () => ({
    status: "idle",
    actions: [],
    appliedCount: 0,
    skippedCount: 0,
    supersededCount: 0,
    error: null,
    start: vi.fn(),
  }),
}));

function entry(overrides: Partial<ActionLogEntry> = {}): ActionLogEntry {
  return {
    action_id: "a1",
    card_id: "c1",
    column_name: "customer_age",
    operation: "fill_missing_median",
    status: "applied",
    timestamp: "2026-08-07T10:00:00Z",
    rejection_reason: null,
    failure_reason: null,
    before_stats: null,
    after_stats: null,
    ...overrides,
  } as ActionLogEntry;
}

/** Rendered with no navigation state, which is exactly what a reload produces. */
function renderReloaded() {
  render(
    <MemoryRouter initialEntries={["/clean/session-1"]}>
      <Routes>
        <Route path="/clean/:sessionId" element={<CleanPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("CleanPage after a reload", () => {
  beforeEach(() => fetchActionLog.mockReset());

  it("recovers a session that was already cleaned", async () => {
    fetchActionLog.mockResolvedValue({ actions: [entry()] });
    renderReloaded();

    expect(await screen.findByText("Your cleaned data")).toBeInTheDocument();
    expect(screen.getByText(/a reload doesn't lose it/)).toBeInTheDocument();
  });

  it("brings back the rules panel, which lives only on this page", async () => {
    fetchActionLog.mockResolvedValue({ actions: [entry()] });
    renderReloaded();
    expect(await screen.findByText("rules panel")).toBeInTheDocument();
  });

  it("brings back the applied fixes so they can still be undone", async () => {
    fetchActionLog.mockResolvedValue({ actions: [entry()] });
    renderReloaded();
    expect(await screen.findByText(/customer_age/)).toBeInTheDocument();
  });

  it("still sends a genuinely fresh session back to the review step", async () => {
    fetchActionLog.mockResolvedValue({ actions: [] });
    renderReloaded();

    expect(await screen.findByText(/Start from there to clean a dataset/)).toBeInTheDocument();
    expect(screen.queryByText("rules panel")).not.toBeInTheDocument();
  });

  it("does not flash the dead end while it is still looking", async () => {
    let release: (value: { actions: ActionLogEntry[] }) => void = () => {};
    fetchActionLog.mockReturnValue(
      new Promise<{ actions: ActionLogEntry[] }>((resolve) => {
        release = resolve;
      })
    );
    renderReloaded();

    expect(screen.getByText(/Looking for this session/)).toBeInTheDocument();
    expect(screen.queryByText(/Start from there to clean a dataset/)).not.toBeInTheDocument();

    // Settled before the test ends, so the pending promise can't outlive it.
    release({ actions: [] });
    await waitFor(() =>
      expect(screen.getByText(/Start from there to clean a dataset/)).toBeInTheDocument()
    );
  });
});
