import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelPlan } from "@/api/types";

const trainModel = vi.fn();
const fetchModelPlan = vi.fn();

vi.mock("@/api/client", async (original) => ({
  ...(await original<typeof import("@/api/client")>()),
  fetchModelPlan: (...args: unknown[]) => fetchModelPlan(...args),
  fetchModelResult: () => Promise.reject(new Error("none yet")),
  fetchModelHistory: () => Promise.resolve(null),
  trainModel: (...args: unknown[]) => trainModel(...args),
}));
vi.mock("@/components/model/ModelPlanForm", () => ({
  ModelPlanForm: ({ onTrain, preselect }: { onTrain: (request: unknown) => void; preselect?: unknown }) => (
    <div>
      <span data-testid="preselect">{JSON.stringify(preselect)}</span>
      <button onClick={() => onTrain({ excluded_columns: [], model_names: [], split_strategy: "random", time_column: null })}>
        train now
      </button>
    </div>
  ),
}));

const { ModelPage } = await import("./ModelPage");
const { ApiError } = await import("@/api/client");

const PLAN = {
  session_id: "s1",
  target_column: "outcome",
  blockers: [],
  features: {
    numeric: [],
    categorical: [],
    dropped: [{ column: "customer_id", reason: "Every value is different, so it is an identifier." }],
  },
} as unknown as ModelPlan;

function renderPage(address = "/model/s1") {
  render(
    <MemoryRouter initialEntries={[address]}>
      <Routes>
        <Route path="/model/:sessionId" element={<ModelPage />} />
      </Routes>
    </MemoryRouter>
  );
}

async function train() {
  renderPage();
  fireEvent.click(await screen.findByText("train now"));
}

describe("ModelPage — a refused or failed training (#99)", () => {
  beforeEach(() => {
    trainModel.mockReset();
    fetchModelPlan.mockReset();
    fetchModelPlan.mockResolvedValue(PLAN);
  });

  it("turns a coded refusal into its action", async () => {
    trainModel.mockRejectedValueOnce(
      new ApiError("Model could not be trained", "There are no usable columns left to predict from.", 422, "no_usable_columns")
    );
    await train();
    expect(await screen.findByText("No columns are left to learn from")).toBeInTheDocument();
    expect(screen.getByText("customer_id")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open the column chooser" })).toBeInTheDocument();
  });

  it("shows today's message for an error without a code", async () => {
    trainModel.mockRejectedValueOnce(new ApiError("Model could not be trained", "Something odd.", 422));
    await train();
    expect(await screen.findByText("Training didn't run")).toBeInTheDocument();
    expect(screen.getByText("Something odd.")).toBeInTheDocument();
  });

  it("does not let a later uncoded error inherit an earlier error's action", async () => {
    trainModel
      .mockRejectedValueOnce(new ApiError("Model could not be trained", "Blocked.", 422, "data_blocked"))
      .mockRejectedValueOnce(new ApiError("Model could not be trained", "Something odd.", 422));
    await train();
    expect(await screen.findByRole("link", { name: "See the failing readiness checks" })).toBeInTheDocument();

    fireEvent.click(screen.getByText("train now"));
    await waitFor(() => expect(screen.getByText("Training didn't run")).toBeInTheDocument());
    expect(screen.queryByRole("link", { name: "See the failing readiness checks" })).not.toBeInTheDocument();
  });
});


describe("ModelPage — set up from a readiness check (#100)", () => {
  beforeEach(() => {
    trainModel.mockReset();
    fetchModelPlan.mockReset();
    fetchModelPlan.mockResolvedValue(PLAN);
  });

  it("hands the form the columns and split the address names", async () => {
    renderPage("/model/s1?exclude=leaky&exclude=other&split=group&group=customer");
    const shown = await screen.findByTestId("preselect");
    expect(JSON.parse(shown.textContent!)).toEqual({ exclude: ["leaky", "other"], split: "group", group: "customer" });
  });

  it("hands it nothing when the address names nothing", async () => {
    renderPage();
    const shown = await screen.findByTestId("preselect");
    expect(JSON.parse(shown.textContent!)).toEqual({ exclude: [] });
  });

  it("does not trust a split that is not grouping", async () => {
    renderPage("/model/s1?split=time&exclude=x");
    const shown = await screen.findByTestId("preselect");
    expect(JSON.parse(shown.textContent!)).toEqual({ exclude: ["x"] });
  });
});
