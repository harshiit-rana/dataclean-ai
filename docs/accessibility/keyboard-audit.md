# Keyboard-only walkthrough, 16 October 2026 (#101)

The automated part of #101 (axe on every page, landmarks, heading levels, a contrast test for every token pairing)
was done before this. This is the manual part the owner asked for: the running app, used with the keyboard only,
across upload, cleaning, training and results.

## How it was done

- The app ran locally: the backend with an in-memory checkpointer and stand-in AI replies (no Redis or API key on the
  machine used), the frontend on the Vite dev server, in Chrome, with the bundled `samples/subscription_churn.csv`.
- At each step, moving focus with Tab and acting with Enter, and recording the focused element and whether it had a
  visible focus style.
- Also checked from the page itself: the accessible name of every control in the tab order, controls with repeated
  names, form fields without labels, groups of choices without a group name, and live regions.

**Not done: a screen-reader test.** Names, roles, live regions and focus order were checked from the accessibility
tree and the DOM, which shows what a screen reader is given, not how NVDA, JAWS or VoiceOver read it in practice.
That still needs a person with a real screen reader: #121.

## Findings and fixes

| Page | Finding | Fix |
| --- | --- | --- |
| Review | The data table's column headers opened their details on a mouse click only; Tab skipped them. | Each header is now a button inside the `<th>`, with `aria-pressed` for the open column. The scrollable table is focusable, so its rows scroll from the keyboard. A critical column says "(critical)" in words, not only with a star. |
| Review | 13 cards each had buttons named only "Approve" and "Reject": a screen reader moving between them heard no difference. | The buttons are named "Approve the fix for *column*" / "Reject the fix for *column*" (the visible word stays first), sit in a group named after the column, and carry `aria-pressed`. |
| Review | A declined card faded to 60% opacity, taking its text below readable contrast. | Marked by its badge and a dashed border instead; the text keeps its contrast. |
| Every page | After moving to another page (approving fixes, "Build a model"), focus was left on `<body>`, so a keyboard user started again from the top with nothing announced. | Focus moves to the new page's main heading after a navigation (`RouteFocus`), or to `<main>` while a lazily loaded page is still arriving. |
| Model | When training finished, the form holding the focused button was replaced by the results: focus fell to `<body>` and the arrival of a result was not announced. | Focus moves to the verdict heading when a new result arrives (training, the improvement search, restoring a run). |
| Model | Choice groups (columns, how to split, which models, the cut-off objective, derived inputs, extreme values) had no group name, so a radio button was read as "1 of 3" without its question. The date-column dropdown had no label. | Each card of choices is a group named by its title; the split options are a radio group; the dropdown is labelled. |
| Model results | 12 chart buttons were all named "Show table". | Named "Show table for *chart title*" (and "Show chart for ..."). |
| Review, Clean, Upload, Model, Apply, Export | Profiling stages, "Reading your file...", "Training...", and error messages appeared without being announced. | The profiling status bar and progress lines are live status regions; error blocks are alerts. |
| Upload | The drop zone was focusable but had the role "presentation", so it announced nothing. | It is a button named "Choose a CSV file, or drop one here" (or names the chosen file). The hidden file input sits beside it rather than inside it, so no control is nested in another. |
| Every page | Buttons and the drop zone showed only the browser's default focus ring, whose contrast on the dark theme depends on the browser. | An explicit ring, the same as the text fields' (`focus-visible:ring-2 ring-primary` with an offset). |
| Export | A download button was named only "clean.py". | "Cleaning script (clean.py)". |

Checked and fine: the tab order follows the visual order on every page; every text field and dropdown on the upload,
apply and model pages has a label; the "Keep this project" bar and its forms; links that look like buttons are
reachable and work with Enter; controls inside collapsed sections are skipped until the section is opened.

## Found on the way, not an accessibility issue

Opening the review page in a development build said a brand-new session had "already been reviewed and cleaned".
React runs effects twice in development, so profiling was started, aborted at once, and started again, and the second
request read the half-started checkpoint as a finished one. Fixed in the profile route: a session counts as finished
only once cleaning has written its cleaned file. A production build does not run effects twice, but an aborted first
request (a quick reload) could produce the same refusal there.

## Tests

Each fix has a test: `DataTable.test.tsx`, `SuggestionCardView.test.tsx`, `route-focus.test.tsx`,
`ModelPageFocus.test.tsx`, `ModelPlanForm.test.tsx` (named groups), `ChartCard.test.tsx` (toggle names), plus the
existing axe checks on every page, which still pass.
