"""The model card: what was built, how well it works, and what not to trust about it.

Assembled in code with only the interpretation slotted in from the LLM -- same split as the
audit report, and for the same reason. Numbers that pass through a language model can come
back subtly changed, and this is a document people quote.

The cautions section is the part that earns the document's keep. A model card that only
reports scores invites the reader to act on them; the useful one says which scores are thin,
which columns were withheld, and how far the numbers moved across folds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.services import model_export, why_this_model
from app.services.ml_readiness import MIN_ROWS_PER_CLASS

logger = logging.getLogger(__name__)

# Re-exported from model_store, which owns every artifact filename so `clear` has one list
# to work from.
from app.services.model_store import CARD_FILENAME, SCRIPT_FILENAME  # noqa: E402

_VERDICT_HEADINGS = {
    "useful": "This model works",
    "marginal": "This model is weak",
    "no_better_than_guessing": "This model does not work",
    "not_enough_test_data": "There isn't enough data to judge this model",
}

_METRIC_LABELS = {
    "accuracy": ("Accuracy", "How often the answer is right overall."),
    "precision": ("Precision", "Of the cases it flags, how many really are."),
    "recall": ("Recall", "Of the real cases, how many it catches."),
    "f1": ("F1", "A single score balancing precision and recall."),
    "roc_auc": ("ROC AUC", "How well it ranks cases. 0.50 is a coin flip, 1.00 is perfect."),
    "roc_auc_ovr_macro": (
        "ROC AUC (macro)",
        "How well it ranks each answer against all the others, averaged over the answers. 0.50 is "
        "chance, 1.00 is perfect.",
    ),
    "brier_score": (
        "Brier score",
        "How far its probabilities are from what happened, squared and averaged. 0 is perfect; "
        "lower is better, and the 'no model at all' row is what always predicting the overall "
        "rate scores.",
    ),
    "expected_calibration_error": (
        "Calibration error",
        "On average, how far 'X% likely' is from how often it really happens. 0 means honest "
        "probabilities; lower is better. Noisy on a small test set, so read it with its range.",
    ),
    "r2": ("R²", "How much of the variation it explains. 0 is no better than the average."),
    "mae": ("Mean absolute error", "The typical size of a miss, in the target's own units."),
    "rmse": ("RMSE", "Like the above, but punishes large misses more."),
    "mae_vs_baseline": (
        "Error vs. the median guess",
        "Its typical miss as a fraction of the typical miss from always guessing the middle "
        "value. 1.00 means no better than that guess; 0.60 means misses 60% as large; lower is "
        "better.",
    ),
}


def _split_description(split: dict) -> str:
    strategy = split.get("strategy")
    if strategy == "time":
        gap = split.get("gap_days")
        text = f"by date, on {split.get('time_column')}"
        if gap:
            dropped = split.get("gap_rows_dropped")
            text += (
                f", leaving a gap of {gap:,} days between training and test"
                + (f" ({dropped:,} rows inside it were dropped)" if dropped is not None else "")
            )
        return text
    if strategy == "group":
        return f"keeping every {split.get('group_column')} on one side"
    return "at random"


def _fmt_with_interval(value, interval: dict | None) -> str:
    """A number, and where it could plausibly land: `0.78 (0.71–0.84)`."""
    if not interval:
        return _fmt(value)
    return f"{_fmt(value)} ({_fmt(interval.get('low'))}–{_fmt(interval.get('high'))})"


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


_SELECTION_NOTE = {
    "cross_validation": "chosen on training folds",
    "only_candidate": "the only model trained",
    "default_order": "kept by default order",
}


def _comparison_table(result: dict) -> list[str]:
    """Every candidate side by side. Which models were tried and rejected is part of the
    record -- without it, a reader can't tell whether the chosen one was the best of several
    or the only one attempted.

    The winner is chosen on cross-validated folds of the training rows, so those lead: a table
    of every candidate's *test* score with one marked "chosen" reads as though the highest test
    score won, and invites choosing by it, which is selecting on the test set by hand. The
    held-out scores follow, for the chosen model only.

    A result saved before models were chosen on folds has no fold scores. Its choice was made
    on the test set, so its test-score table is the accurate description of it."""
    if result.get("selection"):
        return _selection_tables(result)
    return _held_out_comparison(result)


def _selection_tables(result: dict) -> list[str]:
    selection = result["selection"]
    method = selection.get("method")
    metric = selection.get("metric")
    label = _METRIC_LABELS.get(metric or "", (metric or "", ""))[0]

    lines = ["", "### How the model was chosen", ""]
    lines.append(
        "Compared on cross-validated folds of the training rows, never on the held-out rows, "
        f"so the choice cannot have been made with the test answers ({selection.get('chosen_on')})."
    )
    if selection.get("rule_note"):
        lines += ["", selection["rule_note"]]
    lines += [
        "",
        f"| Model | Fold score{f' ({label})' if label else ''} | Fold spread |",
        "| --- | --- | --- |",
    ]
    for entry in result.get("results", []):
        name = entry.get("label", entry.get("model"))
        if entry.get("failed"):
            lines.append(f"| {name} | did not train | — |")
            continue
        if entry.get("model") == result.get("best_model"):
            note = (
                "simplest within one standard error of the best"
                if selection.get("rule") == "one_standard_error"
                else _SELECTION_NOTE.get(method, "chosen")
            )
            name = f"**{name}** (chosen; {note})"
        score = entry.get("cv_score")
        spread = entry.get("cv_std")
        lines.append(
            f"| {name} | {_fmt(score)} | {'± ' + _fmt(spread) if spread is not None else '—'} |"
        )

    if selection.get("distinguishable_note"):
        lines += ["", selection["distinguishable_note"]]
    stability = selection.get("seed_stability") or {}
    if stability.get("seed_dependent"):
        lines += ["", stability["note"]]
    elif stability.get("applicable"):
        lines += [
            "",
            f"Refitted with {len(stability.get('seeds') or [])} random seeds, the chosen model's fold "
            f"score moved by {stability['spread']:.3f}"
            + (f", less than the {stability['gap']:.3f} lead over the runner-up, so the choice is not the "
               f"seed's doing." if stability.get("gap") is not None else "."),
        ]

    single = (result.get("baseline") or {}).get("best_single_column") or {}
    if single.get("applicable"):
        lines += ["", f"**Against its best single input.** {single['note']}"]
    elif single.get("reason"):
        lines += ["", f"_Best single input not compared: {single['reason']}_"]

    lines += ["", "### Held-out evaluation", ""]
    lines.append(
        "Scored once, on rows the model never saw and nothing was chosen against, for the "
        "chosen model only."
    )
    lines += _held_out_comparison(result, chosen_only=True)
    chosen = next((r for r in result.get("results", []) if r.get("model") == result.get("best_model")), {})
    intervals = chosen.get("metrics_intervals") or {}
    if intervals.get("metrics"):
        unit = "whole groups" if intervals.get("by") == "groups" else "rows"
        lines += [
            "",
            f"Figures in brackets are the middle {round(intervals['level'] * 100)}% of "
            f"{intervals['resamples']:,} resamples of the held-out {unit}: the range the score could "
            f"plausibly land in on a different set of {unit}, not a guarantee.",
        ]
    return lines


def _held_out_comparison(result: dict, chosen_only: bool = False) -> list[str]:
    task = result.get("task")
    keys = (
        ["r2", "mae", "rmse", "mae_vs_baseline"]
        if task == "regression"
        else [
            "accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_ovr_macro",
            "brier_score", "expected_calibration_error",
        ]
    )
    present = [
        k
        for k in keys
        if any(k in (r.get("metrics") or {}) for r in result.get("results", []))
    ]
    if not present:
        return []

    header = "| Model | " + " | ".join(_METRIC_LABELS[k][0] for k in present) + " |"
    divider = "| --- " * (len(present) + 1) + "|"
    rows = [header, divider]

    for entry in result.get("results", []):
        if chosen_only and entry.get("model") != result.get("best_model"):
            continue
        name = entry.get("label", entry.get("model"))
        if entry.get("failed"):
            rows.append(f"| {name} | " + " | ".join(["did not train"] * len(present)) + " |")
            continue
        if entry.get("model") == result.get("best_model"):
            name = f"**{name}** (chosen)"
        metrics = entry.get("metrics", {})
        intervals = ((entry.get("metrics_intervals") or {}).get("metrics")) or {}
        rows.append(
            f"| {name} | "
            + " | ".join(_fmt_with_interval(metrics.get(k), intervals.get(k)) for k in present)
            + " |"
        )

    baseline = result.get("baseline") or {}
    for row in [baseline, *(baseline.get("others") or [])]:
        rows.append(
            f"| _No model at all ({row.get('strategy', 'guessing')})_ | "
            + " | ".join(_fmt(row.get(k)) for k in present)
            + " |"
        )
    return ["", *rows, ""]


def _metric_glossary(result: dict) -> list[str]:
    best = next(
        (r for r in result.get("results", []) if r.get("model") == result.get("best_model")),
        {},
    )
    metrics = best.get("metrics", {})
    present = [k for k in _METRIC_LABELS if k in metrics]
    if not present:
        return []
    lines = ["", "**What these mean:**", ""]
    lines += [f"- **{_METRIC_LABELS[k][0]}** — {_METRIC_LABELS[k][1]}" for k in present]
    return lines


def _confusion_table(diagnostics: dict) -> list[str]:
    cells = diagnostics.get("confusion_cells") or []
    if not cells:
        return []
    labels = sorted({c["actual"] for c in cells})
    lookup = {(c["actual"], c["predicted"]): c for c in cells}

    lines = [
        "",
        "### Where it gets things wrong",
        "",
        "Rows are what actually happened; columns are what the model said.",
        "",
        "| Actual \\ Predicted | " + " | ".join(labels) + " |",
        "| --- " * (len(labels) + 1) + "|",
    ]
    for actual in labels:
        values = []
        for predicted in labels:
            cell = lookup.get((actual, predicted), {})
            count = cell.get("count", 0)
            share = cell.get("share_of_actual", 0)
            values.append(f"{count:,} ({share:.0%})")
        lines.append(f"| **{actual}** | " + " | ".join(values) + " |")
    return lines


def _per_class_table(diagnostics: dict) -> list[str]:
    rows = diagnostics.get("per_class") or []
    if not rows:
        return []
    lines = [
        "",
        "### Performance on each answer",
        "",
        "| Answer | Precision | Recall | F1 | Examples in the test set |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} "
            f"| {_fmt(row['f1'])} | {row['support']:,} |"
        )
    return lines


def _error_by_range_table(diagnostics: dict) -> list[str]:
    rows = diagnostics.get("error_by_range") or []
    if not rows:
        return []
    lines = [
        "",
        "### Where it is most wrong",
        "",
        "Models are often accurate in the middle of the range and poor at the extremes, "
        "which a single overall score hides.",
        "",
        "| Actual value | Typical error | Average bias | Rows |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        direction = "over" if row["mean_error"] > 0 else "under"
        lines.append(
            f"| {row['range']} | {_fmt(row['mean_absolute_error'])} | "
            f"{_fmt(abs(row['mean_error']))} {direction} | {row['count']:,} |"
        )
    return lines


def _calibration_section(result: dict) -> list[str]:
    """Whether the probabilities were corrected, and what that did.

    Written when calibration was applied, and also when it was asked for and refused -- a card
    that says nothing would let a reader assume it was done. When it was neither asked for nor
    needed there is nothing to say."""
    calibration = result.get("calibration")
    if not calibration:
        return []

    if calibration.get("method") in (None, "none"):
        if not calibration.get("refused_because"):
            return []
        return [
            "",
            "## Probability calibration",
            "",
            f"Calibration ({calibration.get('requested')}) was asked for and not applied: "
            f"{calibration['refused_because']} The probabilities are the model's own.",
        ]

    method = calibration["method"]
    before, after = calibration.get("before") or {}, calibration.get("after") or {}
    rows = [
        ("brier_score", "Brier score (lower is better)"),
        ("expected_calibration_error", "Calibration error (lower is better)"),
        ("roc_auc", "ROC AUC"),
        ("accuracy", "Accuracy at the default 0.50 line"),
        ("recall", "Recall at the default 0.50 line"),
        ("precision", "Precision at the default 0.50 line"),
    ]
    lines = [
        "",
        "## Probability calibration",
        "",
        f"The model's probabilities were corrected with **{method}** calibration, fitted on "
        f"{calibration.get('fitted_on')}. The held-out rows were scored once more, with it, to "
        f"show the change; they did not take part in fitting it.",
        "",
        "| | Before | After |",
        "| --- | --- | --- |",
    ]
    for key, label in rows:
        if key in before and key in after:
            lines.append(f"| {label} | {_fmt(before[key])} | {_fmt(after[key])} |")
    lines += [
        "",
        "Calibration changes what a probability means, not which cases rank higher"
        + (
            ", so the AUC is unchanged."
            if method == "sigmoid" and not calibration.get("ensemble")
            else "; the AUC can move slightly."
        )
        + " It can change the answers at the default 0.50 line, because 0.50 now sits on honest "
        "probabilities: on a rare answer that flags fewer rows. Choose the cut-off for the "
        "trade-off that matters rather than reading the default line.",
    ]
    if calibration.get("ensemble"):
        lines += [
            "",
            "This was a forward-in-time split, whose folds cover only the later rows, so the "
            "calibrated model averages one copy per fold rather than a single model with a "
            "single mapping; the AUC difference above includes that averaging.",
        ]
    return lines


def _operating_point_section(result: dict) -> list[str]:
    """Where the cut-off sits, what it was chosen for, and how the test set answered.

    Only written when the user chose a trade-off. At the default there is nothing to
    explain, and a section saying "0.50, because that is the default" would be padding."""
    point = result.get("operating_point")
    if not point or point.get("objective") in (None, "default"):
        return []

    observed = point.get("observed") or {}
    expected = point.get("expected") or {}
    default = point.get("at_default") or {}
    ranges = ((point.get("observed_intervals") or {}).get("metrics")) or {}
    label = point.get("positive_label")

    lines = [
        "",
        "## Where the cut-off sits",
        "",
        f"The model flags a row as **{label}** when its probability is at or above "
        f"**{point.get('threshold')}**, not the usual 0.50. That line was chosen so that "
        f"{point.get('description')}.",
        "",
        f"It was chosen on {point.get('chosen_on')}. The held-out figures below are "
        f"therefore measured at a line the test set had no say in, and the gap between the "
        f"two columns is how much the training folds over- or under-promised.",
        *(
            [
                "",
                f"The probabilities were calibrated ({point['calibrated_with']}), so this line is on "
                f"calibrated probabilities: {point.get('threshold')} means the model says it is "
                f"{float(point.get('threshold') or 0):.0%} likely, and means it.",
            ]
            if point.get("calibrated_with")
            else []
        ),
        "",
        "| At this cut-off | Expected (training folds) | Held-out test set | Test set at 0.50 |",
        "| --- | --- | --- | --- |",
        f"| Share of real cases caught (recall) | {_fmt(expected.get('recall'))} | "
        f"{_fmt_with_interval(observed.get('recall'), ranges.get('recall'))} | {_fmt(default.get('recall'))} |",
        f"| Correct when flagged (precision) | {_fmt(expected.get('precision'))} | "
        f"{_fmt_with_interval(observed.get('precision'), ranges.get('precision'))} | {_fmt(default.get('precision'))} |",
        f"| Rows flagged | {_fmt(expected.get('flagged'))} of {_fmt(expected.get('rows'))} | "
        f"{_fmt(observed.get('flagged'))} of {_fmt(observed.get('rows'))} | "
        f"{_fmt(default.get('flagged'))} of {_fmt(default.get('rows'))} |",
        f"| Real cases missed | {_fmt(expected.get('missed'))} | "
        f"{_fmt(observed.get('missed'))} | {_fmt(default.get('missed'))} |",
        f"| False alarms | {_fmt(expected.get('false_alarms'))} | "
        f"{_fmt(observed.get('false_alarms'))} | {_fmt(default.get('false_alarms'))} |",
    ]
    if point.get("note"):
        lines += ["", point["note"]]
    lines += [
        "",
        "The precision and recall in the comparison table above are at 0.50, so every model "
        "is measured the same way. The line only applies to the chosen model, and "
        "`model.joblib` does not know about it -- `predict()` will answer at 0.50. The "
        "snippet at the end of this card shows how to apply it.",
    ]
    return lines


def _importance_table(best: dict) -> list[str]:
    rows = best.get("column_importance") or []
    if not rows:
        return []
    lines = [
        "",
        "## What the model is using",
        "",
        "Share of the model's decision that each column accounts for.",
        "",
        "| Column | Share |",
        "| --- | --- |",
    ]
    lines += [f"| {row['column']} | {row['importance']:.1%} |" for row in rows]
    return lines


def _derived_section(result: dict) -> list[str]:
    """The inputs that weren't in the file.

    A reader comparing the model card against their own spreadsheet will not find
    `signup_date__weekday` in it, and a column they cannot locate is a column they cannot
    check. Each one says where it came from and why it exists."""
    recipes = (result.get("features") or {}).get("derived") or []
    if not recipes:
        return []

    lines = [
        "",
        "## Inputs derived from the data",
        "",
        "These columns are not in the file. They were computed from columns that a model "
        "could not otherwise use -- a date is a different value on nearly every row, and a "
        "paragraph of text is not a category. `train_model.py` rebuilds every one of them.",
        "",
        "| From | Became | Why |",
        "| --- | --- | --- |",
    ]
    for recipe in recipes:
        outputs = ", ".join(f"`{o}`" for o in recipe.get("outputs") or [])
        if not outputs:
            # frequent_categories adds no column to the frame; it changes how one is encoded.
            outputs = "its most common values, encoded; the rest pooled as *other*"
        lines.append(
            f"| `{recipe.get('column')}` | {outputs} | "
            f"{str(recipe.get('reason', '')).replace('|', '/')} |"
        )
    return lines


def _clipping_section(result: dict) -> list[str]:
    """Which inputs had their extreme values pulled back, and to what.

    The limits are real numbers learned from the training rows, and a reader checking the
    model against their own spreadsheet needs them: a value of 4,200 in the file entered the
    model as something much smaller.
    """
    features = result.get("features") or {}
    clipped = features.get("clipped") or []
    if not clipped:
        return []
    limits = features.get("clip_limits") or {}

    def bound(value) -> str:
        return "no limit" if value is None else f"{value:g}"

    lines = [
        "",
        "## Extreme values clipped",
        "",
        "Before fitting, values in these inputs beyond a limit were pulled back to it. The "
        "limits were learned from the training rows only, so the held-out rows had no say in "
        "them, and no row was removed. `train_model.py` does the same.",
        "",
        "| Column | Lowest value kept | Highest value kept |",
        "| --- | --- | --- |",
    ]
    for column in clipped:
        entry = limits.get(column) or {}
        lines.append(
            f"| `{column}` | {bound(entry.get('lower'))} | {bound(entry.get('upper'))} |"
        )
    return lines


def _excluded_section(result: dict) -> list[str]:
    features = result.get("features", {})
    dropped = features.get("dropped", [])
    if not dropped:
        return []
    lines = ["", "## Columns not used, and why", "", "| Column | Reason |", "| --- | --- |"]
    for entry in dropped:
        reason = str(entry.get("reason", "")).replace("|", "\\|")
        lines.append(f"| {entry.get('column')} | {reason} |")
    return lines


def _trust_section(result: dict) -> list[str]:
    """Cross-validation spread and the learning curve, read rather than just printed."""
    deep = result.get("deep_diagnostics") or {}
    if deep.get("skipped"):
        return ["", "## How much to trust these numbers", "", deep["skipped"]]

    cv = deep.get("cross_validation")
    curve = deep.get("learning_curve")
    if not cv and not curve:
        return []

    lines = ["", "## How much to trust these numbers", ""]

    if cv:
        lines += [
            f"Re-scored across {len(cv['folds'])} different splits of the training rows, "
            f"**{cv['metric']}** came out at {cv['mean']:.3f} on average, ranging "
            f"{min(cv['folds']):.3f} to {max(cv['folds']):.3f}.",
            "",
        ]
        if cv["spread"] > 0.15:
            lines += [
                "That is a wide range. The single headline score above could have landed "
                "anywhere in it depending on which rows happened to be held back, so treat "
                "it as approximate.",
                "",
            ]
        else:
            lines += ["That is a narrow range, so the headline score is fairly stable.", ""]

    if curve:
        first, last = curve[0], curve[-1]
        measure = _METRIC_LABELS.get(first.get("metric", ""), ("score",))[0]
        lines.append(
            f"Trained on increasing amounts of data, the {measure} on unseen rows went from "
            f"{first['validation_score']:.3f} at {first['training_rows']:,} rows to "
            f"{last['validation_score']:.3f} at {last['training_rows']:,}."
        )
        lines.append("")
        # The same reading the page shows: a statement only when the difference beats the folds'
        # own disagreement, and "inconclusive" when it does not.
        from app.services.diagnostics import read_learning_curve

        reading = deep.get("learning_curve_reading") or read_learning_curve(curve)
        lines += [reading["text"], ""]

    return lines


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _protocol_section(result: dict) -> list[str]:
    """How the numbers on this card were obtained, and what they can and cannot support.

    Assembled entirely from fields of the result and its evaluation record -- no number in it
    passes through a language model. It is the facts that change how a score should be read:
    how the rows were divided, how the model was chosen, how often the held-out rows have been
    used, whether anything was checked on rows that did not exist when the model was built.
    Parts that do not apply (a sealed holdout, new data, a cut-off) are left out, so the section
    stays short; the section itself is in every card.
    """
    split = result.get("split") or {}
    selection = result.get("selection") or {}
    evaluation = result.get("evaluation") or {}
    task = result.get("task")
    best = next((r for r in result.get("results", []) if r.get("model") == result.get("best_model")), {})
    lines = ["", "## How these numbers were obtained", ""]

    # --- the split
    division = (
        f"{_fmt(split.get('train_rows'))} rows to train on and {_fmt(split.get('test_rows'))} held back "
        f"for testing ({float(split.get('test_fraction') or 0):.0%}), {_split_description(split)}."
    )
    recommended = split.get("recommended") or {}
    if recommended.get("strategy"):
        phrase = why_this_model.split_phrase(recommended["strategy"], recommended.get("column"))
        division += (
            f" The app recommended splitting {phrase}; this run split differently, so its scores are "
            f"not from the recommended protocol."
            if split.get("overridden")
            else " This is the split the app recommended."
        )
    lines.append(f"- **How the rows were divided.** {division}")

    # --- choosing the model
    method = selection.get("method")
    if not selection:
        chosen = (
            "The model was picked by its held-out score (this result predates choosing on training "
            "folds), so that score is the best of several draws on those rows."
        )
    elif method == "only_candidate":
        chosen = "Only one model was trained, so none was chosen between."
    elif method == "default_order":
        chosen = f"{_upper_first(str(selection.get('chosen_on') or ''))}."
    else:
        label = _METRIC_LABELS.get(selection.get("metric") or "", (selection.get("metric") or "the score",))[0]
        repeats = int(selection.get("repeats") or 1)
        chosen = (
            f"By cross-validated {label} over {selection.get('folds')} folds of the training rows"
            f"{f', repeated {repeats} times' if repeats > 1 else ''}; the held-out rows took no part."
        )
        if selection.get("distinguishable") is False:
            chosen += " The folds could not separate the chosen model from its closest rival."
        stability = selection.get("seed_stability") or {}
        if stability.get("seed_dependent"):
            chosen += " Which of the two comes out ahead depends on the random seed."
    trail = result.get("improvement")
    if trail:
        chosen += (
            f" It was then improved by a search of {len(trail.get('rounds') or [])} rounds, each judged "
            f"on the training folds."
        )
    lines.append(f"- **How the model was chosen.** {chosen}")

    # --- how often the held-out rows have been used
    if not evaluation:
        used = (
            "Not recorded: this result predates the record of how often the held-out rows are used, so "
            "the score may not be an independent estimate."
        )
    else:
        if evaluation.get("status") == "independent":
            used = (
                "This was the first time these rows were used to evaluate a model, so the score is an "
                "independent estimate."
            )
            if evaluation.get("message"):
                used += f" {evaluation['message']}"
        else:
            used = evaluation.get("message") or (
                f"These rows had been used {evaluation.get('look_count')} times when this model was scored."
            )
    lines.append(f"- **Use of the held-out rows.** {used}")

    if split.get("sealed"):
        sealed = (
            f"A further {_fmt(split.get('final_rows'))} rows ({float(split.get('final_fraction') or 0):.0%}) "
            f"were sealed away"
        )
        if split.get("sealed_after_looks"):
            sealed += f" after {split['sealed_after_looks']} earlier look(s) at the data"
        final = evaluation.get("final")
        if final:
            value = (final.get("metrics") or {}).get(final.get("primary_metric"))
            sealed += (
                f"; they have been opened once, scoring {_fmt(value)} on "
                f"{_METRIC_LABELS.get(final.get('primary_metric') or '', ('the main score',))[0]}."
            )
        else:
            sealed += "; they have not been opened, so the figures here are still a development estimate."
        lines.append(f"- **Sealed final holdout.** {sealed}")
    elif split.get("sealed_unavailable_reason"):
        lines.append(f"- **Sealed final holdout.** Asked for and not made: {split['sealed_unavailable_reason']}")

    new_data = evaluation.get("new_data")
    if new_data:
        latest = new_data.get("latest") or {}
        lines.append(
            f"- **Checked on new data.** {new_data.get('count')} time(s). Latest: {_fmt(latest.get('rows'))} rows "
            f"that did not exist when the model was built, "
            f"{_METRIC_LABELS.get(latest.get('metric') or '', ('the main score',))[0]} {_fmt(latest.get('value'))} "
            f"({_fmt(latest.get('reported'))} when it was built)."
        )

    # --- probabilities and cut-off
    calibration = result.get("calibration") or {}
    if calibration.get("method") not in (None, "none"):
        lines.append(
            f"- **Probabilities.** Calibrated with {calibration['method']}, {calibration.get('fitted_on')}."
        )
    elif calibration.get("refused_because"):
        lines.append(f"- **Probabilities.** Calibration was asked for and not applied: {calibration['refused_because']}")
    elif task == "classification":
        lines.append("- **Probabilities.** As the model produced them, not calibrated.")
    point = result.get("operating_point") or {}
    if point.get("objective") not in (None, "default") and point.get("threshold") is not None:
        lines.append(
            f"- **Cut-off.** {point['threshold']}, chosen on {point.get('chosen_on') or 'the training folds'}; "
            f"the held-out rows had no say in it."
        )

    # --- leaks and overlaps, only where there are any
    overlaps = []
    if split.get("keep_duplicates_together"):
        overlaps.append("exact copies of a row were kept on one side of the split")
    if split.get("test_rows_seen_in_training"):
        overlaps.append(
            f"{split['test_rows_seen_in_training']} test rows are exact copies of training rows, answer included"
        )
    whole_file = evaluation.get("whole_file_statistics") or []
    if whole_file:
        overlaps.append(
            f"{len(whole_file)} approved fix(es) worked a value out over every row, the held-out rows included"
        )
    if overlaps:
        lines.append(f"- **Overlaps.** {_upper_first('; '.join(overlaps))}.")

    # --- what the numbers can and cannot support
    interval = ((best.get("metrics_intervals") or {}).get("metrics") or {}).get(result.get("primary_metric"))
    supports = (
        "The held-out score estimates how this model does on rows drawn the same way as the test rows"
        + (
            f", and on rows like them it could plausibly land between {_fmt(interval.get('low'))} and "
            f"{_fmt(interval.get('high'))}."
            if interval else "."
        )
    )
    lines += ["", f"**What the numbers support.** {supports}", ""]
    limits = ["data collected differently from this file"]
    if split.get("strategy") in (None, "random", "group"):
        limits.append("how the model will do in a later period")
    if evaluation and evaluation.get("status") not in (None, "independent"):
        limits.append(
            f"an independent estimate: the held-out rows had been used {evaluation.get('look_count')} times"
        )
    lines.append(f"**What they do not support.** {_upper_first('; '.join(limits))}.")
    return lines


def _looks_cell(result: dict) -> str:
    """A run made before the ledger existed has no count, and says so. Reading that as one
    look would present a score of unknown history as an independent one."""
    evaluation = result.get("evaluation")
    if not evaluation:
        return "not recorded"
    count = evaluation.get("look_count")
    status = evaluation.get("status")
    return f"{count} ({status})"


# How much a caution should weigh on whether to rely on the model, most severe first. The order is a
# judgement and is written down here so it can be argued with: a *high* caution says the scores above are
# flattered or not independent (so the headline number is wrong in a known direction); a *medium* one says
# the scores are thin or the model is not what was asked for; a *low* one is context that every reader of
# this kind of result should have but that rarely changes the decision.
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def caution_items(result: dict, best: dict) -> list[dict]:
    """The cautions as `{"id", "severity", "text"}`, most severe first (ties keep the order they were found).

    Derived from the evaluation rather than written by the model, so they appear whether or not the LLM
    call succeeded. The model card and the results page both read this list, which is how they come to say
    the same words.
    """
    items: list[dict] = []

    def add(id_: str, severity: str, text: str) -> None:
        items.append({"id": id_, "severity": severity, "text": text})

    split = result.get("split", {})
    test_rows = split.get("test_rows", 0)
    diagnostics = best.get("diagnostics", {})

    evaluation = result.get("evaluation") or {}
    if evaluation.get("message"):
        add(
            "holdout_reuse",
            "high" if evaluation.get("status") == "exhausted" else "medium",
            evaluation["message"],
        )

    whole_file = evaluation.get("whole_file_statistics") or []
    if whole_file:
        listed = ", ".join(f"`{w['operation']}` on `{w['column']}`" for w in whole_file)
        add(
            "whole_file_statistics", "medium",
            f"These fixes were approved while cleaning and their values were worked out over "
            f"every row -- the held-out rows included -- before the data was split: {listed}. "
            f"The effect of a median is usually small, but it is a leak: the scores above are "
            f"slightly flattered, and the model's own training-only imputer had nothing left "
            f"to do. To avoid it, undo those fixes and let the model fill the gaps from the "
            f"training rows only."
        )

    if test_rows and test_rows < 100:
        add(
            "small_test_set", "medium",
            f"Every score here rests on {test_rows} held-out rows. That is a small sample, "
            f"and the figures would move noticeably on a different split."
        )

    seen = split.get("test_rows_seen_in_training") or 0
    if seen and test_rows:
        add(
            "test_rows_are_copies", "high",
            f"{seen} of the {test_rows} test rows ({seen / test_rows:.0%}) are exact copies of "
            f"training rows, answer included -- the model had already been shown those answers, "
            f"so the scores above are flattered by them. If they are duplicated records rather "
            f"than genuine repeats, remove the duplicates during cleaning and retrain."
        )

    for row in diagnostics.get("per_class") or []:
        if row["support"] < MIN_ROWS_PER_CLASS:
            add(
                f"thin_class_{row['label']}", "medium",
                f"Only {row['support']} test rows have the answer {row['label']!r}, so its "
                f"precision and recall above are barely more than anecdote."
            )

    if split.get("strategy") == "random" and result.get("task") == "classification":
        add(
            "random_split", "low",
            "The data was split at random. If these rows have a time order and predictions "
            "will be made about the future, retrain with a date-based split -- otherwise the "
            "model is being tested on rows from the same period it learned from."
        )

    excluded = result.get("features", {}).get("excluded") or []
    if excluded:
        add(
            "withheld_columns", "low",
            f"{', '.join(excluded)} were deliberately withheld from the model. If any of "
            f"them will genuinely be available when a prediction is needed, the scores here "
            f"understate what is achievable."
        )

    derived = (result.get("features") or {}).get("derived") or []
    if derived:
        sources = ", ".join(str(r.get("column")) for r in derived)
        add(
            "derived_inputs", "low",
            f"Some inputs were derived rather than read from the file ({sources}). Whatever "
            f"scores this model, it can only be used on rows where those source columns are "
            f"present and mean the same thing -- a date parsed in a different format, or a "
            f"text field that stops being filled in, silently changes the inputs."
        )

    point = result.get("operating_point") or {}
    if point.get("objective") not in (None, "default") and not point.get("reachable", True):
        add(
            "cutoff_unreachable", "medium",
            f"A cut-off was asked for so that {point.get('description')}, and no cut-off "
            f"achieves that on the training folds. The nearest one was set instead; the "
            f"table under 'Where the cut-off sits' shows what it actually delivers."
        )

    # Anyone reading a model card is entitled to know how many configurations were tried
    # before this one was chosen -- it is the difference between "this is what a model does
    # on your data" and "this is the best of forty attempts".
    trail = result.get("improvement")
    if trail:
        rounds = trail.get("rounds") or []
        accepted = [r for r in rounds if r.get("accepted")]
        add(
            "searched_model", "low",
            f"This model was arrived at by searching, not by a single fit: {len(rounds)} "
            f"rounds were tried and {len(accepted)} were kept. Every round was judged on "
            f"cross-validated folds of the training rows and the held-out set was scored "
            f"once at the end, so the figures above are not the best of many draws on those "
            f"rows -- but they were chosen with more attempts than a single train, and a "
            f"gap between the cross-validated score "
            f"({trail.get('cv_score')}) and the headline score is normal and expected."
        )
        if trail.get("stopped_because") == "goal_out_of_reach":
            add(
                "goal_out_of_reach", "medium",
                f"A target of {trail.get('goal')} was asked for and not reached. The search "
                f"ran out of things worth trying first, which points at the columns rather "
                f"than the model: better inputs would help, more searching would not."
            )

    return sorted(items, key=lambda item: SEVERITY_ORDER[item["severity"]])


def _deterministic_cautions(result: dict, best: dict) -> list[str]:
    """The same cautions as plain text, most severe first."""
    return [item["text"] for item in caution_items(result, best)]


def build_model_card(
    *,
    session_id: str,
    file_name: str,
    result: dict,
    summary: str = "",
    interpretation: str = "",
    cautions: list[str] | None = None,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    best = next(
        (r for r in result.get("results", []) if r.get("model") == result.get("best_model")),
        {},
    )
    assessment = result.get("assessment") or {}
    split = result.get("split", {})
    features = result.get("features", {})
    diagnostics = best.get("diagnostics", {})

    lines: list[str] = [
        "# Model Card",
        "",
        f"**Predicting:** {result.get('target_column')}  ",
        f"**Built from:** {file_name}  ",
        f"**Session ID:** {session_id}  ",
        f"**Generated:** {generated_at}",
        "",
        "## Summary",
        "",
        summary or "_No summary was generated for this model._",
        "",
        f"## {_VERDICT_HEADINGS.get(assessment.get('verdict', ''), 'Result')}",
        "",
        assessment.get("headline", "_No verdict was recorded._"),
        "",
    ]

    if interpretation:
        lines += [interpretation, ""]

    lines += [
        "## What was built",
        "",
        "| | |",
        "| --- | --- |",
        f"| Task | {result.get('task')} |",
        f"| Model chosen | {best.get('label', '—')} |",
        f"| Columns used | {len(features.get('numeric', [])) + len(features.get('categorical', [])) + len(features.get('frequent', []))} |",
        f"| Rows trained on | {_fmt(split.get('train_rows'))} |",
        f"| Rows held back for testing | {_fmt(split.get('test_rows'))} |",
        f"| How the data was split | {_split_description(split)} |",
        f"| Times these held-out rows have been used | {_looks_cell(result)} |",
    ]

    lines += _protocol_section(result)

    lines += ["", "## Results", ""]
    lines += _comparison_table(result)
    lines += _metric_glossary(result)
    lines += _per_class_table(diagnostics)
    lines += _confusion_table(diagnostics)
    lines += _calibration_section(result)
    lines += _operating_point_section(result)
    lines += _error_by_range_table(diagnostics)
    lines += _importance_table(best)
    lines += _derived_section(result)
    lines += _clipping_section(result)
    lines += _excluded_section(result)
    lines += _trust_section(result)
    try:
        lines += why_this_model.as_markdown(why_this_model.explain(result))
    except Exception:  # noqa: BLE001 -- the card is still worth producing
        logger.warning("Could not explain the choice of model", exc_info=True)

    all_cautions = _deterministic_cautions(result, best) + list(cautions or [])
    if all_cautions:
        lines += ["", "## Before you rely on this", ""]
        lines += [f"- {caution}" for caution in all_cautions]

    try:
        script_note = model_export.build_prediction_snippet(result)
    except Exception:  # noqa: BLE001 -- the card is still worth producing
        logger.warning("Prediction snippet could not be rendered", exc_info=True)
        script_note = ""

    lines += [
        "",
        "## Reproducing this",
        "",
        "The export includes `train_model.py`, which rebuilds this model from scratch with "
        "the same preprocessing, the same split and the same settings. Running it should "
        "reproduce the numbers above.",
        "",
        "The fitted model itself is included as `model.joblib`:",
        "",
        "```python",
        script_note,
        "```",
        "",
        "---",
        "",
        "_Generated by DataClean AI. The data this was trained on was cleaned with changes "
        "the user approved individually; see the accompanying audit report._",
    ]
    return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


async def _write_prose(result: dict, file_name: str, goal_context) -> tuple[str, str, list[str]]:
    """Returns (summary, interpretation, cautions). Empty on failure -- the card is still
    worth producing without prose, since every number and warning in it is assembled in
    code."""
    import json

    from app.config import get_settings
    from app.services.llm_client import get_llm_client
    from app.services.prompts.model_prompts import (
        build_model_card_messages,
        build_retry_messages,
    )

    client = get_llm_client(get_settings(), streaming=False)
    messages = build_model_card_messages(result, file_name, goal_context)

    for attempt in range(2):  # initial try + one retry, as elsewhere
        try:
            response = await client.ainvoke(messages)
            raw = response.content if isinstance(response.content, str) else str(response.content)
            parsed = json.loads(_strip_code_fences(raw))
            return (
                str(parsed.get("summary", "")),
                str(parsed.get("interpretation", "")),
                [str(c) for c in (parsed.get("cautions") or [])],
            )
        except Exception as exc:  # noqa: BLE001 -- must never sink the card
            logger.warning("Model card prose failed (attempt %d): %s", attempt + 1, exc)
            messages = build_retry_messages(result, file_name, goal_context, str(exc))

    return "", "", []


async def generate_model_card(
    *, session_id: str, file_name: str, result: dict, goal_context, directory
) -> str:
    """Writes model_card.md and train_model.py into the session directory, returning the
    card's markdown.

    Both artifacts are written together because they are two halves of the same claim: the
    card says what the numbers were, the script lets someone check.
    """
    summary, interpretation, cautions = await _write_prose(result, file_name, goal_context)
    markdown = build_model_card(
        session_id=session_id,
        file_name=file_name,
        result=result,
        summary=summary,
        interpretation=interpretation,
        cautions=cautions,
    )

    directory.mkdir(parents=True, exist_ok=True)
    (directory / CARD_FILENAME).write_text(markdown, encoding="utf-8")

    try:
        (directory / SCRIPT_FILENAME).write_text(
            model_export.build_training_script(result, file_name="cleaned.csv"), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 -- the card is independently useful
        logger.exception("Training script could not be written")

    return markdown
