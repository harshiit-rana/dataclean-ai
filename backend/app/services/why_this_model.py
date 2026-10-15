"""Why this model: how it was chosen, why the others were not, and what constrained the choice.

Everything the page and the model card say here is assembled from fields already in the result --
no language model writes a word of it. Each statement carries its `evidence`: the result fields it
rests on, with their values, so a reader can see the number behind a sentence and a test can check
that every one of them resolves to the value shown.

`explain(result)` is a pure function of a stored result, so it is attached when a result is served
(see `routes_model._with_later_evidence`) and the card calls it again: the two cannot disagree.

A statement never claims more than the fields support. When the chosen model is not clearly ahead
of its closest rival, that statement is a warning and leads the panel.
"""

from __future__ import annotations

from typing import Any

SECTIONS = {
    "how_chosen": "How the model was chosen",
    "alternatives": "Why the others were not chosen",
    "constraints": "What constrained the choice",
    "improvement": "What the improvement search tried",
}

_METRIC_NAMES = {"roc_auc": "area under the curve", "accuracy": "accuracy", "r2": "R-squared"}


# --- evidence -----------------------------------------------------------------------------------


def resolve(result: dict, field: str) -> Any:
    """The value a dotted field path names, or None when it is not there.

    `results[random_forest_classifier].cv_score` picks the list entry whose `model` (or `name`) is
    that, `improvement.rounds[2].score` picks by position, and `.length` is the length of a list.
    """
    node: Any = result
    for token in field.split("."):
        name, _, rest = token.partition("[")
        index = rest.rstrip("]") if rest else None
        if name == "length":
            return len(node) if isinstance(node, (list, tuple)) else None
        if name:
            node = node.get(name) if isinstance(node, dict) else None
        if index is not None:
            if not isinstance(node, list):
                return None
            if index.isdigit():
                node = node[int(index)] if int(index) < len(node) else None
            else:
                node = next(
                    (e for e in node if isinstance(e, dict) and index in (e.get("model"), e.get("name"))),
                    None,
                )
        if node is None:
            return None
    return node


def _evidence(result: dict, *pairs: tuple[str, str]) -> list[dict]:
    """`(field, label)` pairs, kept only where the result actually holds a value."""
    found = []
    for field, label in pairs:
        value = resolve(result, field)
        if value is not None and not isinstance(value, (dict, list)):
            found.append({"field": field, "label": label, "value": value})
    return found


def _statement(
    result: dict, id_: str, section: str, text: str, *pairs: tuple[str, str], tone: str = "plain"
) -> dict:
    return {
        "id": id_,
        "section": section,
        "tone": tone,
        "text": text,
        "evidence": _evidence(result, *pairs),
    }


# --- wording helpers ----------------------------------------------------------------------------


def _label(model: str | None) -> str:
    from app.services.modeling import MODELS

    spec = MODELS.get(model or "")
    return spec.label if spec else str(model)


def _in_a_sentence(label: str) -> str:
    return label[:1].lower() + label[1:] if label[1:2].islower() else label


def _signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value):.3f}"


def _rank(model: str) -> int:
    from app.services.modeling import MODELS

    spec = MODELS.get(model)
    return spec.complexity_rank if spec else 99


def _list(items: list[str]) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# --- the four sections --------------------------------------------------------------------------


def _how_chosen(result: dict) -> list[dict]:
    selection = result.get("selection")
    chosen = result.get("best_model")
    chosen_label = _label(chosen)
    if not selection:
        return [
            _statement(
                result, "how_chosen_legacy", "how_chosen",
                f"This result was saved before models were chosen on training folds: {chosen_label} was "
                f"picked by its held-out score, so that score is the best of several draws on those rows "
                f"and flatters it. Train again to have the choice made on the training rows alone.",
                ("best_model", "chosen model"),
                tone="warning",
            )
        ]

    method = selection.get("method")
    if method == "only_candidate":
        return [
            _statement(
                result, "how_chosen_method", "how_chosen",
                f"Only one model was trained, {chosen_label}, so there was no choice between models to explain.",
                ("selection.method", "method"), ("best_model", "chosen model"),
            )
        ]
    if method == "default_order":
        reason = str(selection.get("chosen_on") or "").strip()
        reason = (reason[:1].upper() + reason[1:]) if reason else "No model could be cross-validated"
        return [
            _statement(
                result, "how_chosen_method", "how_chosen",
                f"{reason.rstrip('.')}. {chosen_label} is the simplest model, kept by default order.",
                ("selection.method", "method"), ("best_model", "chosen model"),
                tone="warning",
            )
        ]

    metric = selection.get("metric")
    metric_name = _METRIC_NAMES.get(metric or "", metric or "the score")
    repeats = int(selection.get("repeats") or 1)
    folds = selection.get("folds")
    repeat_text = f", repeated {repeats} times with different shuffles" if repeats > 1 else ""
    statements = [
        _statement(
            result, "how_chosen_method", "how_chosen",
            f"Every model that trained was cross-validated on the training rows ({folds} folds{repeat_text}) "
            f"and scored by {metric_name}. The held-out rows took no part in the choice.",
            ("selection.method", "method"), ("selection.metric", "scored by"),
            ("selection.folds", "folds"), ("selection.repeats", "repeats"),
        )
    ]

    top = selection.get("highest_scorer")
    chosen_cv = resolve(result, f"results[{chosen}].cv_score")
    if selection.get("rule") == "one_standard_error" and selection.get("rule_note"):
        statements.append(
            _statement(
                result, "how_chosen_rule", "how_chosen", selection["rule_note"],
                ("selection.rule", "rule"), ("selection.standard_error", "one standard error"),
                (f"results[{top}].cv_score", f"{_label(top)} fold score"),
                (f"results[{chosen}].cv_score", f"{chosen_label} fold score"),
            )
        )
    elif chosen_cv is not None:
        statements.append(
            _statement(
                result, "how_chosen_rule", "how_chosen",
                f"{chosen_label} had the highest fold score ({chosen_cv:.3f}).",
                ("selection.rule", "rule"), (f"results[{chosen}].cv_score", f"{chosen_label} fold score"),
            )
        )
    return statements


def _alternatives(result: dict) -> list[dict]:
    selection = result.get("selection") or {}
    if selection.get("method") != "cross_validation":
        return []
    chosen = result.get("best_model")
    chosen_label = _label(chosen)
    chosen_cv = resolve(result, f"results[{chosen}].cv_score")
    top = selection.get("highest_scorer")
    top_cv = resolve(result, f"results[{top}].cv_score")
    standard_error = selection.get("standard_error")
    rival = selection.get("runner_up")
    statements: list[dict] = []

    if rival and selection.get("paired_difference"):
        gap = selection["paired_difference"]
        rival_label = _label(rival)
        pairs = (
            ("selection.runner_up", "closest alternative"),
            ("selection.distinguishable", "folds can tell them apart"),
            ("selection.paired_difference.difference", "average fold gap"),
            ("selection.paired_difference.low", "95% range, low"),
            ("selection.paired_difference.high", "95% range, high"),
        )
        if selection.get("distinguishable") is False and selection.get("distinguishable_note"):
            statements.append(
                _statement(
                    result, "runner_up", "alternatives",
                    f"Too close to call. {selection['distinguishable_note']}", *pairs, tone="warning",
                )
            )
        elif top == chosen:
            statements.append(
                _statement(
                    result, "runner_up", "alternatives",
                    f"{chosen_label} beat {_in_a_sentence(rival_label)}, the closest alternative: it was ahead "
                    f"by {_signed(gap['difference'])} on average across the folds (95% range "
                    f"{_signed(gap['low'])} to {_signed(gap['high'])}), which is more than noise.",
                    *pairs,
                )
            )
        else:
            statements.append(
                _statement(
                    result, "runner_up", "alternatives",
                    f"{rival_label} scored higher by {_signed(gap['difference'])} on average (95% range "
                    f"{_signed(gap['low'])} to {_signed(gap['high'])}) and the folds can tell the two apart; "
                    f"{chosen_label} was kept because it is simpler and within one standard error of the best.",
                    *pairs,
                )
            )

    for entry in result.get("results") or []:
        model = entry.get("model")
        if model in (chosen, rival):
            continue
        name = _label(model)
        if entry.get("failed"):
            statements.append(
                _statement(
                    result, f"alt_{model}", "alternatives",
                    f"{name} could not be trained: {entry['failed']}",
                    (f"results[{model}].failed", "reason"), tone="notable",
                )
            )
            continue
        cv = entry.get("cv_score")
        if cv is None or chosen_cv is None:
            continue
        evidence = (
            (f"results[{model}].cv_score", f"{name} fold score"),
            (f"results[{model}].cv_std", f"{name} fold spread"),
            (f"results[{chosen}].cv_score", f"{chosen_label} fold score"),
        )
        within = (
            standard_error is not None and top_cv is not None and cv >= float(top_cv) - float(standard_error)
        )
        if within:
            why = "it is simpler" if _rank(chosen) < _rank(model) else "it scored higher"
            text = (
                f"{name} scored {cv:.3f} against {chosen_label}'s {chosen_cv:.3f}, within one standard error "
                f"of the best, so the folds cannot rule it out; {chosen_label} was preferred because {why}."
            )
            evidence += (("selection.standard_error", "one standard error"),)
        else:
            text = (
                f"{name} scored {cv:.3f} against {chosen_label}'s {chosen_cv:.3f} "
                f"({_signed(cv - chosen_cv)}), further behind the best than one standard error, so the "
                f"folds favour {chosen_label}."
            )
            evidence += (("selection.standard_error", "one standard error"),)
        statements.append(_statement(result, f"alt_{model}", "alternatives", text, *evidence))
    return statements


def _constraints(result: dict) -> list[dict]:
    statements: list[dict] = []
    split = result.get("split") or {}
    strategy = split.get("strategy")
    chosen_on_folds = (result.get("selection") or {}).get("method") == "cross_validation"
    if strategy == "time":
        statements.append(
            _statement(
                result, "split", "constraints",
                f"The data was split by date, on {split.get('time_column')}. "
                + (
                    "Every fold inside the training rows also trained on earlier rows and validated on later "
                    "ones, so the choice was not made by looking at the future."
                    if chosen_on_folds else ""
                ).strip(),
                ("split.strategy", "split"), ("split.time_column", "date column"),
                ("split.gap_days", "gap in days"),
            )
        )
    elif strategy == "group":
        statements.append(
            _statement(
                result, "split", "constraints",
                f"Every {split.get('group_column')} was kept whole on one side of the split"
                + (", and inside every fold, so the choice was made on entities the model had not seen." if chosen_on_folds else "."),
                ("split.strategy", "split"), ("split.group_column", "grouped by"),
                ("split.train_groups", "groups in training"), ("split.test_groups", "groups in the test set"),
            )
        )
    elif strategy:
        statements.append(
            _statement(
                result, "split", "constraints", "The held-out rows were chosen at random.",
                ("split.strategy", "split"),
            )
        )
    recommended = split.get("recommended") or {}
    if split.get("overridden") and recommended.get("strategy"):
        statements.append(
            _statement(
                result, "split_override", "constraints",
                f"The app recommended splitting {split_phrase(recommended.get('strategy'), recommended.get('column'))}; "
                f"this run split {split_phrase(strategy, split.get('time_column') or split.get('group_column'))} "
                f"instead, so its scores are a different kind of evidence from a run that followed the advice.",
                ("split.recommended.strategy", "recommended"), ("split.strategy", "used"),
                tone="notable",
            )
        )
    if split.get("keep_duplicates_together"):
        statements.append(
            _statement(
                result, "duplicates_together", "constraints",
                "Rows that are exact copies of each other were kept together at the split and in every fold, "
                "so a copy of a training row could not land in a validation fold.",
                ("split.keep_duplicates_together", "copies kept together"),
            )
        )
    if split.get("sealed"):
        statements.append(
            _statement(
                result, "sealed", "constraints",
                f"{int(split.get('final_rows') or 0):,} further rows were sealed away and took no part in the choice.",
                ("split.sealed", "sealed"), ("split.final_rows", "sealed rows"),
            )
        )
    excluded = (result.get("features") or {}).get("excluded") or []
    if excluded:
        statements.append(
            _statement(
                result, "excluded", "constraints",
                f"You withheld {_list(excluded)}, so the models were compared without "
                f"{'it' if len(excluded) == 1 else 'them'}.",
                ("features.excluded.length", "columns withheld"),
                tone="notable",
            )
        )
    derived = (result.get("features") or {}).get("derived") or []
    if derived:
        statements.append(
            _statement(
                result, "derived", "constraints",
                f"{len(derived)} derived input{'s' if len(derived) != 1 else ''} "
                f"(from {_list([str(r.get('column')) for r in derived])}) went to every candidate alike.",
                ("features.derived.length", "derived inputs"),
            )
        )
    point = result.get("operating_point") or {}
    if point.get("objective") not in (None, "default") and point.get("threshold") is not None:
        statements.append(
            _statement(
                result, "operating_point", "constraints",
                f"The cut-off was set to {point['threshold']} so that {point.get('description')}. It was "
                f"chosen after the model was, on the training folds, and did not influence which model won."
                + ("" if point.get("reachable", True) else " No cut-off reaches that on the training folds, so the nearest one was used."),
                ("operating_point.objective", "objective"), ("operating_point.threshold", "cut-off"),
                ("operating_point.reachable", "reachable"),
                tone="plain" if point.get("reachable", True) else "notable",
            )
        )
    calibration = result.get("calibration") or {}
    if calibration.get("method") not in (None, "none"):
        statements.append(
            _statement(
                result, "calibration", "constraints",
                f"The probabilities were calibrated ({calibration['method']}) after the model was chosen; "
                f"that corrects the probabilities and did not change which model won.",
                ("calibration.method", "method"), ("calibration.folds", "folds"),
            )
        )
    return statements


def split_phrase(strategy: str | None, column: str | None) -> str:
    if strategy == "time":
        return f"by date on {column}" if column else "by date"
    if strategy == "group":
        return f"keeping every {column} together" if column else "by group"
    return "at random"


def _improvement(result: dict) -> list[dict]:
    trail = result.get("improvement")
    if not trail:
        return []
    rounds = trail.get("rounds") or []
    kept = [r for r in rounds if r.get("accepted")]
    statements = [
        _statement(
            result, "improvement_summary", "improvement",
            f"This model came from a search of {len(rounds)} rounds, {len(kept)} kept. Every round was judged "
            f"on cross-validated folds of the training rows; the held-out rows were scored once, at the end.",
            ("improvement.rounds.length", "rounds tried"), ("improvement.cv_score", "fold score reached"),
            ("improvement.cv_std", "fold spread"),
        )
    ]
    for index, entry in enumerate(rounds):
        verdict = "kept" if entry.get("accepted") else "rejected"
        outcome = str(entry.get("outcome") or "").strip()
        statements.append(
            _statement(
                result, f"round_{index}", "improvement",
                f"{entry.get('label') or entry.get('name')}: {entry.get('what_it_tried')} -- {verdict}"
                + (f". {outcome}" if outcome else "."),
                (f"improvement.rounds[{index}].score", "fold score"),
                (f"improvement.rounds[{index}].gain", "gain"),
                (f"improvement.rounds[{index}].accepted", "kept"),
            )
        )
    if trail.get("stopped_because") == "goal_out_of_reach":
        statements.append(
            _statement(
                result, "improvement_goal", "improvement",
                f"A target of {trail.get('goal')} was asked for and not reached; the search ran out of things "
                f"worth trying first, which points at the columns rather than the model.",
                ("improvement.goal", "target"), ("improvement.stopped_because", "stopped because"),
                tone="notable",
            )
        )
    return statements


# --- the explanation ----------------------------------------------------------------------------


def explain(result: dict) -> dict:
    """The explanation for a stored result: `{"sections", "statements", "lead_id"}`.

    `lead_id` names the first warning (a choice that is not clearly better than its rival, a model
    picked by default order or on the test set), so the page and the card can lead with it.
    """
    statements = (
        _how_chosen(result) + _alternatives(result) + _constraints(result) + _improvement(result)
    )
    lead = next((s["id"] for s in statements if s["tone"] == "warning"), None)
    return {"sections": SECTIONS, "statements": statements, "lead_id": lead}


def _evidence_text(statement: dict) -> str:
    evidence = "; ".join(f"{e['label']} {e['value']}" for e in statement["evidence"])
    return f" _({evidence})_" if evidence else ""


def as_markdown(explanation: dict) -> list[str]:
    """The same statements for the model card, with the numbers they rest on."""
    statements = explanation.get("statements") or []
    if not statements:
        return []
    lead = explanation.get("lead_id")
    lines = ["", "## Why this model", ""]

    def line(s: dict) -> str:
        return f"- {s['text']}" + _evidence_text(s)

    leading = [s for s in statements if s["id"] == lead]
    if leading:
        lines += [f"**{leading[0]['text']}**" + _evidence_text(leading[0]), ""]
    for key, heading in SECTIONS.items():
        section = [s for s in statements if s["section"] == key and s["id"] != lead]
        if section:
            lines += [f"### {heading}", ""] + [line(s) for s in section] + [""]
    return lines
