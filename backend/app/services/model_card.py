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

from app.services import model_export
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
    "r2": ("R²", "How much of the variation it explains. 0 is no better than the average."),
    "mae": ("Mean absolute error", "The typical size of a miss, in the target's own units."),
    "rmse": ("RMSE", "Like the above, but punishes large misses more."),
}


def _split_description(split: dict) -> str:
    strategy = split.get("strategy")
    if strategy == "time":
        return f"by date, on {split.get('time_column')}"
    if strategy == "group":
        return f"keeping every {split.get('group_column')} on one side"
    return "at random"


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _comparison_table(result: dict) -> list[str]:
    """Every candidate side by side. Which models were tried and rejected is part of the
    record -- without it, a reader can't tell whether the chosen one was the best of several
    or the only one attempted."""
    task = result.get("task")
    keys = (
        ["r2", "mae", "rmse"]
        if task == "regression"
        else ["accuracy", "precision", "recall", "f1", "roc_auc"]
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
        name = entry.get("label", entry.get("model"))
        if entry.get("failed"):
            rows.append(f"| {name} | " + " | ".join(["did not train"] * len(present)) + " |")
            continue
        if entry.get("model") == result.get("best_model"):
            name = f"**{name}** (chosen)"
        metrics = entry.get("metrics", {})
        rows.append(f"| {name} | " + " | ".join(_fmt(metrics.get(k)) for k in present) + " |")

    baseline = result.get("baseline") or {}
    baseline_cells = [_fmt(baseline.get(k)) for k in present]
    rows.append(
        f"| _No model at all ({baseline.get('strategy', 'guessing')})_ | "
        + " | ".join(baseline_cells)
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
        "",
        "| At this cut-off | Expected (training folds) | Held-out test set | Test set at 0.50 |",
        "| --- | --- | --- | --- |",
        f"| Share of real cases caught (recall) | {_fmt(expected.get('recall'))} | "
        f"{_fmt(observed.get('recall'))} | {_fmt(default.get('recall'))} |",
        f"| Correct when flagged (precision) | {_fmt(expected.get('precision'))} | "
        f"{_fmt(observed.get('precision'))} | {_fmt(default.get('precision'))} |",
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
            f"Re-scored across {len(cv['folds'])} different splits of the same data, "
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
        gap = last["training_score"] - last["validation_score"]
        still_climbing = last["validation_score"] - first["validation_score"] > 0.05
        lines.append(
            f"Trained on increasing amounts of data, the score on unseen rows went from "
            f"{first['validation_score']:.3f} at {first['training_rows']:,} rows to "
            f"{last['validation_score']:.3f} at {last['training_rows']:,}."
        )
        lines.append("")
        if gap > 0.15:
            lines += [
                "The model scores much better on the rows it trained on than on unseen ones, "
                "which means it is partly memorizing. More data would likely help more than "
                "a different model.",
                "",
            ]
        elif still_climbing:
            lines += [
                "The score was still improving as data was added, so more rows would "
                "probably still be worth having.",
                "",
            ]
        else:
            lines += [
                "The score flattened out, so more rows of the same kind are unlikely to help "
                "much. Better or different columns would matter more.",
                "",
            ]

    return lines


def _deterministic_cautions(result: dict, best: dict) -> list[str]:
    """Warnings derived from the evaluation rather than written by the model, so they appear
    whether or not the LLM call succeeded."""
    cautions: list[str] = []
    split = result.get("split", {})
    test_rows = split.get("test_rows", 0)
    diagnostics = best.get("diagnostics", {})

    if test_rows and test_rows < 100:
        cautions.append(
            f"Every score here rests on {test_rows} held-out rows. That is a small sample, "
            f"and the figures would move noticeably on a different split."
        )

    seen = split.get("test_rows_seen_in_training") or 0
    if seen and test_rows:
        cautions.append(
            f"{seen} of the {test_rows} test rows ({seen / test_rows:.0%}) are exact copies of "
            f"training rows, answer included -- the model had already been shown those answers, "
            f"so the scores above are flattered by them. If they are duplicated records rather "
            f"than genuine repeats, remove the duplicates during cleaning and retrain."
        )

    for row in diagnostics.get("per_class") or []:
        if row["support"] < MIN_ROWS_PER_CLASS:
            cautions.append(
                f"Only {row['support']} test rows have the answer {row['label']!r}, so its "
                f"precision and recall above are barely more than anecdote."
            )

    if split.get("strategy") == "random" and result.get("task") == "classification":
        cautions.append(
            "The data was split at random. If these rows have a time order and predictions "
            "will be made about the future, retrain with a date-based split -- otherwise the "
            "model is being tested on rows from the same period it learned from."
        )

    excluded = result.get("features", {}).get("excluded") or []
    if excluded:
        cautions.append(
            f"{', '.join(excluded)} were deliberately withheld from the model. If any of "
            f"them will genuinely be available when a prediction is needed, the scores here "
            f"understate what is achievable."
        )

    derived = (result.get("features") or {}).get("derived") or []
    if derived:
        sources = ", ".join(str(r.get("column")) for r in derived)
        cautions.append(
            f"Some inputs were derived rather than read from the file ({sources}). Whatever "
            f"scores this model, it can only be used on rows where those source columns are "
            f"present and mean the same thing -- a date parsed in a different format, or a "
            f"text field that stops being filled in, silently changes the inputs."
        )

    point = result.get("operating_point") or {}
    if point.get("objective") not in (None, "default") and not point.get("reachable", True):
        cautions.append(
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
        cautions.append(
            f"This model was arrived at by searching, not by a single fit: {len(rounds)} "
            f"rounds were tried and {len(accepted)} were kept. Every round was judged on "
            f"cross-validated folds of the training rows and the held-out set was scored "
            f"once at the end, so the figures above are not the best of many draws on those "
            f"rows -- but they were chosen with more attempts than a single train, and a "
            f"gap between the cross-validated score "
            f"({trail.get('cv_score')}) and the headline score is normal and expected."
        )
        if trail.get("stopped_because") == "goal_out_of_reach":
            cautions.append(
                f"A target of {trail.get('goal')} was asked for and not reached. The search "
                f"ran out of things worth trying first, which points at the columns rather "
                f"than the model: better inputs would help, more searching would not."
            )

    return cautions


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
    ]

    lines += ["", "## Results", ""]
    lines += _comparison_table(result)
    lines += _metric_glossary(result)
    lines += _per_class_table(diagnostics)
    lines += _confusion_table(diagnostics)
    lines += _operating_point_section(result)
    lines += _error_by_range_table(diagnostics)
    lines += _importance_table(best)
    lines += _derived_section(result)
    lines += _excluded_section(result)
    lines += _trust_section(result)

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
