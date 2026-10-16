import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { axeViolations } from "@/test/axe";
import { getSessionToken, listKeptProjects, rememberKeptProject } from "@/lib/sessionToken";

/** The kept projects this browser holds keys for, and opening one with a saved key (#120). */

const fetchProject = vi.fn();
vi.mock("@/api/client", async (original) => ({
  ...(await original<typeof import("@/api/client")>()),
  fetchProject: (id: string) => fetchProject(id),
}));

const { KeptProjects } = await import("./KeptProjects");

function renderList() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route path="/" element={<main><KeptProjects /></main>} />
        <Route path="/model/:sessionId" element={<p>model page</p>} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  vi.clearAllMocks();
});

describe("KeptProjects", () => {
  it("lists what this browser remembers, newest first, with a way in", () => {
    rememberKeptProject({ id: "a", name: "Older", keptAt: "2026-10-01T00:00:00Z" }, "tok-a");
    rememberKeptProject({ id: "b", name: "Newer", keptAt: "2026-10-16T00:00:00Z" }, "tok-b");
    renderList();
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("Newer");
    expect(items[1]).toHaveTextContent("Older");
    expect(screen.getByRole("link", { name: "Open Newer" })).toHaveAttribute("href", "/model/b");
  });

  it("says so when there are none", () => {
    renderList();
    expect(screen.getByText(/None in this browser/)).toBeInTheDocument();
  });

  it("opens a project with its id and key, and remembers it", async () => {
    fetchProject.mockResolvedValue({ kept: true, name: "Churn study", kept_at: "2026-10-16T12:00:00Z", file_name: "c.csv" });
    renderList();
    fireEvent.click(screen.getByRole("button", { name: "Open a project with its key" }));
    fireEvent.change(screen.getByLabelText("Project id"), { target: { value: " s9 " } });
    fireEvent.change(screen.getByLabelText("Project key"), { target: { value: "tok-9" } });
    fireEvent.click(screen.getByRole("button", { name: "Open project" }));

    expect(await screen.findByText("model page")).toBeInTheDocument();
    expect(getSessionToken("s9")).toBe("tok-9");
    expect(listKeptProjects()[0]).toMatchObject({ id: "s9", name: "Churn study" });
  });

  it("forgets a key the server refuses and says no project matches", async () => {
    const { ApiError } = await import("@/api/client");
    fetchProject.mockRejectedValue(new ApiError("Forbidden", "denied", 403));
    renderList();
    fireEvent.click(screen.getByRole("button", { name: "Open a project with its key" }));
    fireEvent.change(screen.getByLabelText("Project id"), { target: { value: "s9" } });
    fireEvent.change(screen.getByLabelText("Project key"), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Open project" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("No project matches that id and key.");
    expect(getSessionToken("s9")).toBeNull();
    expect(listKeptProjects()).toEqual([]);
  });

  it("has no accessibility violations with the form open", async () => {
    rememberKeptProject({ id: "a", name: "Churn study", keptAt: null }, "tok-a");
    const { container } = renderList();
    fireEvent.click(screen.getByRole("button", { name: "Open a project with its key" }));
    expect(await axeViolations(container)).toEqual([]);
  });
});
