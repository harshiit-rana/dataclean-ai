import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ImproveRequest, ModelPlan, ModelResult } from "@/api/types";

const streamImprove = vi.fn();

vi.mock("@/api/client", async (original) => ({
  ...(await original<typeof import("@/api/client")>()),
  fetchModelPlan: () => Promise.resolve({ session_id: "s1", target_column: "outcome", blockers: [], features: { numeric: [], categorical: [], dropped: [] } } as unknown as ModelPlan),
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
}));
vi.mock("@/components/model/ModelResults", () => ({ ModelResults: () => <div>results</div> }));
vi.mock("@/components/model/ModelPlanForm", () => ({ ModelPlanForm: () => <div>form</div> }));

const { ModelPage } = await import("./ModelPage");

async function improveWith(minutes: string): Promise<ImproveRequest> {
  streamImprove.mockReturnValue(new AbortController());
  render(
    <MemoryRouter initialEntries={["/model/s1"]}>
      <Routes>
        <Route path="/model/:sessionId" element={<ModelPage />} />
      </Routes>
    </MemoryRouter>
  );
  const box = await screen.findByLabelText("Spend at most");
  if (minutes) fireEvent.change(box, { target: { value: minutes } });
  fireEvent.click(screen.getByRole("button", { name: "Improve it" }));
  return streamImprove.mock.calls[0][1] as ImproveRequest;
}

describe("ModelPage — a time limit on the search (#47)", () => {
  beforeEach(() => streamImprove.mockReset());

  it("sends no limit when the box is empty", async () => {
    expect((await improveWith("")).time_limit_seconds).toBeNull();
  });

  it("sends the minutes as seconds", async () => {
    expect((await improveWith("2")).time_limit_seconds).toBe(120);
  });

  it("accepts a fraction of a minute", async () => {
    expect((await improveWith("0.5")).time_limit_seconds).toBe(30);
  });

  it("keeps a tiny limit at the one second the server accepts", async () => {
    expect((await improveWith("0.001")).time_limit_seconds).toBe(1);
  });

  it("keeps a huge limit at the hour the server accepts", async () => {
    expect((await improveWith("500")).time_limit_seconds).toBe(3600);
  });

  it("ignores text that is not a number, rather than sending nonsense", async () => {
    expect((await improveWith("soon")).time_limit_seconds).toBeNull();
  });
});
