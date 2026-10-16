import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { axeViolations } from "@/test/axe";
import { getSessionToken, listKeptProjects, storeSessionToken } from "@/lib/sessionToken";

/** Keeping, renaming, re-keying and deleting a project from any session page (#120). */

const fetchProject = vi.fn();
const keepProject = vi.fn();
const rotateProjectToken = vi.fn();
const deleteSession = vi.fn();

vi.mock("@/api/client", async (original) => ({
  ...(await original<typeof import("@/api/client")>()),
  fetchProject: (id: string) => fetchProject(id),
  keepProject: (id: string, name: string) => keepProject(id, name),
  rotateProjectToken: (id: string) => rotateProjectToken(id),
  deleteSession: (id: string) => deleteSession(id),
}));

const { ProjectBar } = await import("./ProjectBar");

const EPHEMERAL = { kept: false, name: null, kept_at: null, file_name: "churn.csv" };
const KEPT = { kept: true, name: "Churn study", kept_at: "2026-10-16T12:00:00+00:00", file_name: "churn.csv" };

function renderBar() {
  return render(
    <MemoryRouter initialEntries={["/model/s1"]}>
      <Routes>
        <Route path="/:step/:sessionId" element={<main><ProjectBar /></main>} />
        <Route path="/" element={<p>upload page</p>} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  sessionStorage.clear();
  localStorage.clear();
  vi.clearAllMocks();
  storeSessionToken("s1", "tok-1");
});

describe("ProjectBar", () => {
  it("says an ordinary session is deleted after a day, and offers to keep it", async () => {
    fetchProject.mockResolvedValue(EPHEMERAL);
    renderBar();
    expect(await screen.findByText(/deleted a day after you last use it/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Keep this project" })).toBeInTheDocument();
  });

  it("keeps it under a name, remembers it in this browser and shows the key once", async () => {
    fetchProject.mockResolvedValue(EPHEMERAL);
    keepProject.mockResolvedValue({ ...KEPT, token: null });
    renderBar();
    fireEvent.click(await screen.findByRole("button", { name: "Keep this project" }));
    fireEvent.change(screen.getByLabelText("Project name"), { target: { value: "Churn study" } });
    fireEvent.click(screen.getByRole("button", { name: "Keep it" }));

    expect(await screen.findByText("Save this project key now.")).toBeInTheDocument();
    expect(screen.getByLabelText("Project key")).toHaveTextContent("tok-1");
    expect(keepProject).toHaveBeenCalledWith("s1", "Churn study");
    expect(listKeptProjects()).toEqual([{ id: "s1", name: "Churn study", keptAt: KEPT.kept_at }]);
    expect(screen.getByText("Churn study")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "I've saved it" }));
    expect(screen.queryByText("Save this project key now.")).not.toBeInTheDocument();
  });

  it("uses the key the server issued when it had to issue one", async () => {
    fetchProject.mockResolvedValue(EPHEMERAL);
    keepProject.mockResolvedValue({ ...KEPT, token: "tok-new" });
    renderBar();
    fireEvent.click(await screen.findByRole("button", { name: "Keep this project" }));
    fireEvent.change(screen.getByLabelText("Project name"), { target: { value: "Churn study" } });
    fireEvent.click(screen.getByRole("button", { name: "Keep it" }));
    expect(await screen.findByLabelText("Project key")).toHaveTextContent("tok-new");
  });

  it("replaces the key everywhere this browser holds it when a new one is made", async () => {
    fetchProject.mockResolvedValue(KEPT);
    rotateProjectToken.mockResolvedValue({ token: "tok-2" });
    renderBar();
    fireEvent.click(await screen.findByRole("button", { name: "New key" }));
    expect(await screen.findByLabelText("Project key")).toHaveTextContent("tok-2");
    expect(getSessionToken("s1")).toBe("tok-2");
  });

  it("deletes only once the project's name is typed, then forgets it and goes home", async () => {
    fetchProject.mockResolvedValue(KEPT);
    deleteSession.mockResolvedValue({ deleted: "s1", files_removed: 9, bytes_removed: 1000, checkpoint_deleted: true });
    renderBar();
    fireEvent.click(await screen.findByRole("button", { name: "Delete project" }));
    const confirm = screen.getByRole("button", { name: "Delete for good" });
    expect(confirm).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/to confirm/), { target: { value: "Churn" } });
    expect(confirm).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/to confirm/), { target: { value: "Churn study" } });
    fireEvent.click(confirm);

    expect(await screen.findByText("upload page")).toBeInTheDocument();
    expect(deleteSession).toHaveBeenCalledWith("s1");
    expect(getSessionToken("s1")).toBeNull();
  });

  it("says why when deleting fails, and stays put", async () => {
    const { ApiError } = await import("@/api/client");
    fetchProject.mockResolvedValue(EPHEMERAL);
    deleteSession.mockRejectedValue(new ApiError("Forbidden", "This session needs its key.", 403));
    renderBar();
    fireEvent.click(await screen.findByRole("button", { name: "Delete now" }));
    fireEvent.change(screen.getByLabelText(/to confirm/), { target: { value: "delete" } });
    fireEvent.click(screen.getByRole("button", { name: "Delete for good" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This session needs its key.");
  });

  it("has no accessibility violations, kept or not, with the forms open", async () => {
    fetchProject.mockResolvedValue(KEPT);
    const { container } = renderBar();
    fireEvent.click(await screen.findByRole("button", { name: "Delete project" }));
    await waitFor(() => expect(screen.getByLabelText(/to confirm/)).toBeInTheDocument());
    expect(await axeViolations(container)).toEqual([]);
  });
});
