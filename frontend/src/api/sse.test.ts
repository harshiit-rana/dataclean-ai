import { beforeEach, describe, expect, it, vi } from "vitest";

const fetchEventSource = vi.fn(() => Promise.resolve());
vi.mock("@microsoft/fetch-event-source", () => ({ fetchEventSource }));

const { streamImprove, streamProfile, streamResume } = await import("./sse");

type Options = { method: string; body?: string; openWhenHidden?: boolean };
const lastCall = () => {
  const calls = fetchEventSource.mock.calls as unknown as [string, Options][];
  return calls[calls.length - 1];
};
const lastOptions = () => lastCall()[1];
const lastUrl = () => lastCall()[0];
const handlers = { onEvent: vi.fn(), onError: vi.fn() };

describe("POST streams", () => {
  beforeEach(() => fetchEventSource.mockClear());

  /* fetch-event-source's default aborts the request when the tab is hidden and sends it
     again when the tab comes back. Every stream here is a POST that does something --
     profile, apply fixes, run a search -- so switching tabs and back used to re-send it. */
  it.each([
    ["profile", () => streamProfile("s1", handlers)],
    [
      "resume",
      () =>
        streamResume(
          "s1",
          { approved_action_ids: [], skipped_action_ids: [], skip_reasons: {} },
          handlers
        ),
    ],
    [
      "improve",
      () =>
        streamImprove(
          "s1",
          {
            excluded_columns: [],
            model_names: [],
            split_strategy: "random",
            time_column: null,
            include_deep_diagnostics: true,
          },
          handlers
        ),
    ],
  ])("the %s stream is not re-sent when the tab is hidden and shown again", (_name, start) => {
    start();
    expect(lastOptions().openWhenHidden).toBe(true);
    expect(lastOptions().method).toBe("POST");
  });

  it("sends the improve request to the streaming endpoint with its body", () => {
    streamImprove(
      "s1",
      {
        excluded_columns: ["customer_id"],
        model_names: [],
        split_strategy: "group",
        time_column: null,
        group_column: "customer_id",
        include_deep_diagnostics: true,
        goal: 0.8,
      },
      handlers
    );
    expect(lastUrl()).toMatch(/\/api\/model\/s1\/improve\/stream$/);
    expect(JSON.parse(lastOptions().body!)).toMatchObject({
      split_strategy: "group",
      group_column: "customer_id",
      goal: 0.8,
    });
  });
});
