# 0001. The evaluation protocol: what counts as a look at the test set, and where a fresh holdout comes from

- **Status:** Accepted, 14 October 2026. Written under delegated authority for Milestone 1
  (issue #2) and open to the owner's revision — see "What would change this".
- **Implements:** #5 (ledger), #6 (recording), #7 (rules), #8 (sealed holdout), #9 (new data),
  #90 – #92 (what the user sees).
- **Reproduce the numbers:** `backend/.venv/Scripts/python.exe docs/decisions/0001-worked-example.py`

## The problem

Since `ec21948` no *automated* decision reads the held-out rows: candidates are chosen on
training folds, the improvement loop searches on folds, the cut-off is chosen on out-of-fold
predictions. The *user* can still select on them by hand. Every train reveals the test score, the
run history lines the scores up, and "change a column, retrain, look" ten times makes the number
on the page the best of ten draws on the same fifth of the rows. The score is then reported as
though it were one honest measurement.

The principle we hold (roadmap, #1): **the held-out test set never informs a decision.** For an
automatic decision that is a property we can test. For a human decision it can only be made
*visible*: count the looks, say what they cost, and offer a way to get a measurement nothing was
chosen against.

## Decision

### 1. What is a look

A **look** is one *successful* response that computes and returns metrics measured on the
held-out rows for a model the app just fitted.

| Route | Counts as a look? | Why |
| --- | --- | --- |
| `POST /api/model/{id}/train` | **Yes**, one | Fits, then scores the chosen model on the held-out rows. |
| `POST /api/model/{id}/improve` | **Yes**, one | The search itself never opens the held-out rows; the final rebuild does, once. The rounds are not looks. |
| `POST /api/model/{id}/improve/stream` | **Yes**, one | Same run. Counted when the result is produced and saved, even if the client has gone: the score was revealed to the app's user-facing record, and a refresh will show it. |
| `GET /api/model/{id}/result` | No | Re-reads a stored result; nothing new is revealed. |
| `GET /api/model/{id}/card`, `/history`, `/script`, `/download`, `/correlations`, `/plan` | No | None computes held-out performance. (`/card` renders a stored result.) |
| `POST /api/apply/{id}` with labelled rows | **No — a different event**, `new_data_evaluation` | Newly uploaded rows are not the session's held-out rows. Recorded in the ledger so it can be shown, never counted against the held-out set. |
| `POST /api/apply/{id}` without answers | No | Predictions only; no performance is computed. |
| `GET /api/apply/{id}/{apply_id}/{kind}` | No | A download. |
| `GET /api/export/*`, `GET /api/readiness/*` | No | No model scoring. |
| Any request that fails, or is refused (blocked readiness → 422) | No | Nothing was revealed. |
| `POST /api/model/{id}/finalise` (#8) | **Its own event**, `final_evaluation` | Opens the sealed partition, once. See §4. |
| `POST /api/model/{id}/runs/{run_id}/reproduce` (#72) | **No — its own event**, `reproduction` | Re-runs a configuration that was fixed before this score, on the same data, to check the recorded numbers come back. Nothing can be chosen by it and no run is saved from it; when it reproduces it reveals nothing new. Recorded in the ledger so it can be shown, never counted. |
| `POST /api/model/{id}/runs/{run_id}/restore` (#67) | No | Makes a stored run current; reveals its stored result again, fits nothing. |

A new route that returns held-out performance must be added to this table in the same change,
and to the property suite (#23).

### 2. What a look is counted against

The **split definition**: `(strategy, time_column, group_column, test_fraction, seed)` — plus
`final_fraction` when a sealed holdout is in use, and the duplicate-grouping flag when #16 lands.
Two runs share a count only if they hold out the same rows by the same rule. The key is a hash
of those fields in a fixed order (`evaluation_ledger.split_key`).

**Dataset versions do not reset the count.** Each look records the dataset version and its full
fingerprint (#3, #4), so the history is auditable, but the *status* a user sees is computed
across every version of the session for that split. Reasons:

- Cleaning after a look is the commonest way to select on the test set by hand: see the score,
  decide the data needs more work, fix it, look again. A reset on every version would make
  the counter easy to launder — one undo and it reads "independent" again.
- With the same seed and a changed row count the held-out rows move, but the old held-out rows
  have by then been trained on and *decided over*; the information has not gone anywhere.
- The cost of being wrong is asymmetric. Over-counting after a cleaning change shows a warning
  the user can read and dismiss; under-counting presents a used holdout as fresh.

Changing the split definition *does* start a new count: it is a different held-out set. But the
response also carries `looks_on_other_splits`, and when it is non-zero the message says that those
rows overlap and have been seen — the new count is not a clean slate, and says so.

### 3. What happens as looks accumulate

For the *k*-th look at a split (counting the current one):

| Look | Status | What the user sees |
| --- | --- | --- |
| 1 | `independent` | The score, unqualified. |
| 2 – 3 | `reused` | The score with a structured note: how many times, and that if anything was changed because of an earlier score, this is no longer an independent estimate. |
| 4 and beyond | `exhausted` | The verdict **headline itself** is qualified, and the score is labelled optimistic. |

**The threshold has a reason.** If a person keeps the best of *k* looks, the expected inflation
is the expected maximum of *k* independent draws of the score's own noise. In units of the
score's standard error that is 0.56 at *k* = 2, 0.85 at 3, **1.03 at 4**, 1.16 at 5. We call a
holdout exhausted when the expected optimism from best-of-*k* selection reaches one standard
error — the size of the effect the verdict machinery already treats as noise. That is *k* = 4.
It is a worst case (it assumes every look was used to pick the best), which is the right
direction to err for a statement about independence.

**Nothing is blocked.** A strict limit would make small files unusable and would only teach
people to work around it. A user with 300 rows keeps every capability; they are told the truth
sooner. The qualification text, verbatim:

- *reused:* "These held-out rows have now been used to evaluate a model **N times** in this
  session. If you changed anything because of an earlier score, this one is no longer an
  independent estimate."
- *exhausted:* "These held-out rows have been used **N times**, and the choices made between runs
  were informed by them, so this score is optimistic. Treat it as a development figure, not an
  independent one. For an independent estimate, use the model on new labelled rows ('Use this on
  a new file')."

Thresholds are named constants in `evaluation_ledger.py` (`INDEPENDENT_LOOKS = 1`,
`EXHAUSTED_AT_LOOK = 4`) with this reasoning beside them.

### 4. How a fresh holdout is obtained

Two ways, in order of preference. **A re-split is not one of them.**

1. **New labelled rows (#9).** The apply flow already scores the model on rows uploaded later.
   When that file carries at least `MIN_TEST_ROWS_TO_JUDGE` labelled rows (and the rarer class
   clears its own minimum), none overlap the original upload (checked by row hash), and the
   upload has not already been counted (its fingerprint is in the ledger), the result is recorded
   as an independent evaluation on new data and shown beside — never in place of — the original
   score. This is the cleanest holdout there is: it did not exist when anything was decided.
2. **A sealed final holdout (#8), opt-in at plan time.** A third partition, never evaluated while
   iterating, opened once by an explicit `finalise`. It is **only sealed if it existed before the
   first look**: the plan offers it, and the ledger records whether any look preceded it. A
   sealed partition carved after looks have been taken is shown as `sealed after N looks`, and
   the qualification stays.

**Why not simply re-split with a new seed?** Because every row already informed a decision. The
old test rows are now training rows; the user chose columns, cleaning and models after seeing the
old test score. A new random test set drawn from the same rows is *less* contaminated than the old
one only in the sense that the model has not been scored on it yet — the choices that produced the
model were still shaped by the rows it now trains on. Offering "get a fresh test set" as a button
would turn an exhausted holdout back into a green "independent" with nothing having changed. The
app does not offer it. If a user changes the split rule on purpose, §2 applies, with the overlap
disclosed.

### 5. How it is shown

- **Verdict card:** "evaluated on these held-out rows *N* times", with the split rule (#92).
  One look is rendered quietly; two or more as a warning.
- **Every score** carries one of three labels: *exploratory (training folds)*, *held-out
  evaluation* (with the reuse status), *new data* (#91).
- **Model card and run history:** the status at the time of the run is recorded with the run and
  printed in the card (#7, #94). Runs from before the ledger existed read "not recorded", never
  "independent".
- **Exports:** `train_model.py` reproduces the app's split and is unaffected by the ledger. The
  model card carries the evaluation section.

## The three designs weighed

Counts below are for `samples/subscription_churn.csv` (1,218 labelled rows, 853 / 365). The
app's own bar for a held-out score to mean anything is 30 rows and 10 of the rarer class.

| Design | Training | Development | Final | Verdict |
| --- | --- | --- | --- | --- |
| **(a) Count and warn** (two-way 80 / 20) | 974 | 244 (73 rarer) | — | **Adopted for every session.** Cheap, never blocks, honest about reuse. |
| **(b) Sealed final holdout** (70 / 15 / 15) | 852 | 183 (55 rarer) | 183 (55 rarer) | **Adopted as an opt-in**, offered only when *both* held-out partitions clear the bar. |
| **(c) Nested cross-validation** (5 × 5) | ≈ 915 per inner fit | 243 per outer fold | — | **Rejected for now.** |

And the same arithmetic on a 300-row file with a 20% rarer class:

| Design | Training | Development | Final | Result |
| --- | --- | --- | --- | --- |
| (a) | 240 | 60 (12 rarer) | — | Can be judged. |
| (b) | 210 | 45 (**9 rarer**) | 45 (**9 rarer**) | Both too small to judge: not offered. |

- **Why (b) is not the default.** It takes 30% of the data out of training for every session, and
  on the 300-row file leaves neither partition able to support a verdict. A protocol that makes
  small files untrainable is worse than one that tells them the truth. So it is offered when the
  arithmetic works and withheld, with the reason, when it does not.
- **Why (c) is rejected.** It answers "how good is this *procedure*" and costs 30 fits per model
  against 6, and it leaves no sealed set: afterwards every row has informed some decision, so
  there is nothing to hold up for a user who has been iterating. On small data the app's answer
  is already "not enough to judge". It can be revisited if budgets grow (#47) and the strategy
  interface (#48) makes a nested evaluation cheap to add.

## What a user with 300 rows can still do

Everything. Train, improve, retrain and compare — nothing is blocked. They see the score
unqualified once, then with the reuse note. The sealed holdout is not offered, with the reason
("each held-out part would have 9 of the rarer class; 10 are needed"). Their route to an
independent estimate is new labelled rows through "Use this on a new file", and the app says so
at the point the headline becomes qualified.

## Consequences

- A **ledger** (`test_set_ledger.json`, per session, never truncated, kept across
  `model_store.clear`) stores one entry per look and per new-data evaluation (#5, #6).
- Training results carry `evaluation: {look_count, limit, status, message, looks_on_other_splits}`,
  computed from the ledger *before* the current look is recorded, then recorded (#7).
- Existing runs have no status. They render as "not recorded".
- The **property suite (#23)** is the guard: every new decision path adds a case that flips the
  held-out answers and asserts the decision does not move. A new route that reveals held-out
  performance adds a row to the table in §1.
- Users who iterate legitimately will see warnings quickly. The messages say what to do next,
  not only what is wrong.

## As built

Decisions the record left open, settled when the sealed holdout (#8) and the new-data check (#9)
were built:

- **Sizes.** 70 / 15 / 15: `SEALED_FINAL_SIZE` and `SEALED_DEV_SIZE` in `modeling.py`. The
  development partition is what `/train` and `/improve` report on, the role the test set has
  always had; `/finalise` is the only route that reads the sealed one.
- **What fixes the sealed rows.** Only the data, the seed, the split strategy and its column, and
  (when copies are kept together) the file's own columns. Never the inputs chosen, the models
  tried, the derived inputs ticked or anything the improvement loop finds, so nothing done while
  iterating can move them. A hash of the sealed rows is stored and checked at `finalise`; if the
  data no longer produces those rows the evaluation is refused rather than done on others.
- **Once per (data version, split rule).** A second call returns the stored result with a note. A
  *different model* asking for the same rows gets that stored result too, with the reason: scoring
  a second model on rows the first result may already have informed is not independent. If the
  data changes, a new version may open its own sealed rows, and the response says how many times
  the rule has been opened on earlier versions — those rows overlap, so it is never presented as a
  clean slate.
- **Sealed late.** `sealed_after_looks` records how many looks had been taken when the seal was
  first made, and the first record for a (data, split rule) wins. A seal taken after looks is still
  unopened but is shown as "sealed after N looks", since the rows informed earlier decisions as
  training or development rows.
- **New data (#9).** A check on a file's real answers is recorded only when a score was computed,
  once per distinct file (by content hash), against the model that was current, excluding rows
  whose inputs match the original upload. It is shown beside the original score, never in place
  of it, and does not count as a look.

## What would change this

- **The owner prefers blocking** after a limit. The ledger and statuses are the same; only the
  routes' behaviour in the `exhausted` state would change. We advise against it for the reason
  above.
- **The threshold.** `EXHAUSTED_AT_LOOK` is derived from a worst-case argument; simulation
  on real iteration logs could justify a looser value. It is one constant.
- **Resetting on a dataset version.** If users find over-counting after pure cleaning changes
  unfair, the alternative is to reset only when the held-out *rows* are provably the same (a
  fingerprint of the held-out partition) — which keeps the laundering protection for edits to
  the rows themselves. More machinery for a narrower case; not worth building before it is asked
  for.
