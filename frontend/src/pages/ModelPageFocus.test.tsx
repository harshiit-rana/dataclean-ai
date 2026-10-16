import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import type { ModelPlan } from "@/api/types";

/** Found in the keyboard walkthrough (#101): a finished training replaced the form that held focus and left it on
 * <body>, with nothing read out. Focus now moves to the verdict. */

vi.mock("@/api/client", async (original) => ({
  ...(await original<typeof import("@/api/client")>()),
  fetchModelPlan: () =>
    Promise.resolve({ session_id: "s1", target_column: "outcome", blockers: [], features: { numeric: [], categorical: [], dropped: [] } } as unknown as ModelPlan),
  fetchModelResult: () => Promise.reject(new Error("none yet")),
  fetchModelHistory: () => Promise.resolve(null),
  trainModel: () => Promise.resolve({ task: "classification", results: [], best_model: "m" }),
}));
vi.mock("@/components/model/ModelPlanForm", () => ({
  ModelPlanForm: ({ onTrain }: { onTrain: (request: unknown) => void }) => (
    <button onClick={() => onTrain({ excluded_columns: [], model_names: [], split_strategy: "random", time_column: null })}>
      train now
    </button>
  ),
}));
vi.mock("@/components/model/ModelResults", () => ({
  ModelResults: () => (
    <h2 id="model-verdict" tabIndex={-1}>
      This model works
    </h2>
  ),
}));

const { ModelPage } = await import("./ModelPage");

describe("ModelPage — where focus goes when a result arrives (#101)", () => {
  it("moves focus to the verdict", async () => {
    render(
      <MemoryRouter initialEntries={["/model/s1"]}>
        <Routes>
          <Route path="/model/:sessionId" element={<ModelPage />} />
        </Routes>
      </MemoryRouter>
    );
    const button = await screen.findByText("train now");
    button.focus();
    fireEvent.click(button);
    await waitFor(() => expect(screen.getByRole("heading", { name: "This model works" })).toHaveFocus());
  });
});
