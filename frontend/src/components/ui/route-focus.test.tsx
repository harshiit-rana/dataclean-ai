import { act, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useNavigate } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { RouteFocus } from "./route-focus";

/** Found in the keyboard walkthrough (#101): after a navigation focus was left on <body>. */

function First() {
  const navigate = useNavigate();
  return (
    <>
      <h1>First page</h1>
      <button onClick={() => navigate("/second")}>Go on</button>
    </>
  );
}

function renderApp() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <RouteFocus />
      <main>
        <Routes>
          <Route path="/" element={<First />} />
          <Route path="/second" element={<h1>Second page</h1>} />
        </Routes>
      </main>
    </MemoryRouter>
  );
}

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve(null)));

describe("RouteFocus", () => {
  it("leaves focus alone on the first load", async () => {
    renderApp();
    await act(nextFrame);
    expect(document.body).toHaveFocus();
  });

  it("moves focus to the new page's heading after a navigation", async () => {
    renderApp();
    fireEvent.click(screen.getByRole("button", { name: "Go on" }));
    await act(nextFrame);
    expect(screen.getByRole("heading", { name: "Second page" })).toHaveFocus();
  });
});
