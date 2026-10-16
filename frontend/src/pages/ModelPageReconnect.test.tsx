import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelPlan, ModelResult } from "@/api/types";
import type { StreamHandlers } from "@/api/sse";
import type { ImproveStreamEvent } from "@/api/types";

/** A search runs as a job (#103): a dropped stream, or a reload, picks the progress up again. */

const streamImprove = vi.fn();
const streamJob = vi.fn();

vi.mock("@/api/client", async (original) => ({
  ...(await original<typeof import("@/api/client")>()),
  fetchModelPlan: () =>
    Promise.resolve({ session_id: "s1", target_column: "outcome", blockers: [], features: { numeric: [], categorical: [], dropped: [] } } as unknown as ModelPlan),
  fetchModelResult: () =>
    Promise.resolve({
      task: "classification",
      target_column: "outcome",
      features: { numeric: ["x"], categorical: [], dropped: [] },
      split: { strategy: "random", time_column: null, group_column: null, train_rows: 160, test_rows: 40 },
      results: [],
      best_model: "m",
    } as unknown as ModelResult),
  fetchModelHistory: () => Promise.resolve(null),
}));
vi.mock("@/api/sse", async (original) => ({
  ...(await original<typeof import("@/api/sse")>()),
  streamImprove: (...args: unknown[]) => streamImprove(...args),
  streamJob: (...args: unknown[]) => streamJob(...args),
}));
vi.mock("@/components/model/ModelResults", () => ({ ModelResults: () => <div>results</div> }));
vi.mock("@/components/model/ModelPlanForm", () => ({ ModelPlanForm: () => <div>form</div> }));

const { ModelPage } = await import("./ModelPage");
const { StreamRequestError } = await import("@/api/sse");

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/model/s1"]}>
      <Routes>
        <Route path="/model/:sessionId" element={<ModelPage />} />
      </Routes>
    </MemoryRouter>
  );
}

async function startSearch(): Promise<StreamHandlers<ImproveStreamEvent>> {
  renderPage();
  fireEvent.click(await screen.findByRole("button", { name: "Improve it" }));
  return streamImprove.mock.calls[0][2] as StreamHandlers<ImproveStreamEvent>;
}

const ROUND = { round: 1, name: "baseline", kept: true } as never;

beforeEach(() => {
  sessionStorage.clear();
  streamImprove.mockReset().mockReturnValue(new AbortController());
  streamJob.mockReset().mockReturnValue(new AbortController());
});

describe("ModelPage — reconnecting to a search (#103)", () => {
  it("picks a dropped stream up after the events it already showed", async () => {
    const handlers = await startSearch();
    act(() => {
      handlers.onEvent({ type: "job", job_id: "j1" });
      handlers.onEvent({ type: "round", round: ROUND });
      handlers.onClose?.();
    });
    await waitFor(() => expect(streamJob).toHaveBeenCalled(), { timeout: 3000 });
    expect(streamJob.mock.calls[0].slice(0, 3)).toEqual(["s1", "j1", 1]);
    expect(screen.queryByText(/Lost the connection/)).not.toBeInTheDocument();
  });

  it("remembers the job for this tab and forgets it when the search ends", async () => {
    const handlers = await startSearch();
    act(() => handlers.onEvent({ type: "job", job_id: "j1" }));
    expect(sessionStorage.getItem("dataclean.job.s1")).toBe("j1");
    act(() => handlers.onEvent({ type: "error", detail: "No usable columns." }));
    expect(sessionStorage.getItem("dataclean.job.s1")).toBeNull();
  });

  it("does not reconnect to a stream the server refused", async () => {
    const handlers = await startSearch();
    act(() => {
      handlers.onEvent({ type: "job", job_id: "j1" });
      handlers.onError(new StreamRequestError("This data can't be trained on.", 422, "data_blocked"));
    });
    expect(await screen.findByText("This data can't be trained on.")).toBeInTheDocument();
    await new Promise((resolve) => setTimeout(resolve, 1200));
    expect(streamJob).not.toHaveBeenCalled();
  });

  it("follows a search that was interrupted and resumed as a new job (#104)", async () => {
    const handlers = await startSearch();
    act(() => {
      handlers.onEvent({ type: "job", job_id: "j1" });
      handlers.onEvent({ type: "round", round: ROUND });
      handlers.onEvent({ type: "error", detail: "The process stopped.", code: "job_interrupted", resumed_as: "j2" });
    });
    expect(streamJob.mock.calls[0].slice(0, 3)).toEqual(["s1", "j2", 0]);
    expect(sessionStorage.getItem("dataclean.job.s1")).toBe("j2");
    expect(screen.queryByText("The process stopped.")).not.toBeInTheDocument();
  });

  it("after a reload, follows the search this tab started from its first event", async () => {
    sessionStorage.setItem("dataclean.job.s1", "j9");
    renderPage();
    await waitFor(() => expect(streamJob).toHaveBeenCalled());
    expect(streamJob.mock.calls[0].slice(0, 3)).toEqual(["s1", "j9", 0]);
  });

  it("gives up after a few tries and says the search is still running", async () => {
    const handlers = await startSearch();
    act(() => handlers.onEvent({ type: "job", job_id: "j1" }));
    // Each reconnection fails at once.
    streamJob.mockImplementation((_s: string, _j: string, _after: number, h: StreamHandlers<ImproveStreamEvent>) => {
      setTimeout(() => h.onClose?.(), 0);
      return new AbortController();
    });
    act(() => handlers.onClose?.());
    expect(await screen.findByText(/Lost the connection/, {}, { timeout: 9000 })).toBeInTheDocument();
    expect(streamJob).toHaveBeenCalledTimes(3);
  }, 15000);
});
