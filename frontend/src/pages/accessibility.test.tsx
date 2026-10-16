import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { axeViolations } from "@/test/axe";

/** Automated accessibility checks for the pages without a test file of their own (#101). The Apply, Clean and
 * Model pages are checked in theirs. Each page is rendered inside `<main>`, as the app shell renders it. */

vi.mock("@/api/client", async (original) => ({
  ...(await original<typeof import("@/api/client")>()),
  fetchPreview: () =>
    Promise.resolve({
      session_id: "s1",
      columns: [
        { name: "spend", dtype: "float64", null_count: 2, unique_count: 40, is_critical: false, is_target: false },
        { name: "churned", dtype: "object", null_count: 0, unique_count: 2, is_critical: false, is_target: true },
      ],
      rows: [{ spend: 12.5, churned: "yes" }, { spend: null, churned: "no" }],
      total_rows: 2,
      total_columns: 2,
      preview_row_count: 2,
    }),
  fetchAuditReport: () => Promise.resolve({ markdown: "# Data Cleaning Audit Report\n\nAll good.", has_pdf: true }),
  fetchReadiness: () => Promise.reject(new Error("not needed here")),
  fetchExportVerification: () => Promise.resolve({ status: "not_checked" }),
}));
vi.mock("@/hooks/useProfileStream", () => ({
  useProfileStream: () => ({
    status: "complete", currentAgent: null, statusMessage: "", issueCounts: { high: 0, medium: 1, low: 0 },
    issues: [], cards: [], streamingCardText: {}, error: null, errorStatus: null, partialWarning: null,
  }),
}));

const { UploadPage } = await import("./UploadPage");
const { ReviewPage } = await import("./ReviewPage");
const { ExportPage } = await import("./ExportPage");

function renderAt(path: string, route: string, element: React.ReactElement) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <main>
        <Routes>
          <Route path={route} element={element} />
        </Routes>
      </main>
    </MemoryRouter>
  );
}

describe("Accessibility of the pages (#101)", () => {
  it("Upload", async () => {
    renderAt("/", "/", <UploadPage />);
    expect(await axeViolations(document.body)).toEqual([]);
  });

  it("Review", async () => {
    renderAt("/review/s1", "/review/:sessionId", <ReviewPage />);
    await screen.findAllByText("spend");
    expect(await axeViolations(document.body)).toEqual([]);
  });

  it("Export", async () => {
    renderAt("/export/s1", "/export/:sessionId", <ExportPage />);
    await screen.findByText("All good.");
    expect(await axeViolations(document.body)).toEqual([]);
  });
});
