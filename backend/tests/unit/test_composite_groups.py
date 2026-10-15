"""A grouped split on more than one column.

Some entities are named by several columns together -- a store within a region, a patient at a
site. The grouped split took one column, so grouping on either alone put the same entity on both sides
and the only workaround was to combine the columns by hand beforehand, which the app could not verify.
A key of several columns now names the group, every column in it is withheld from the inputs, and a
single-column request behaves exactly as it always did.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_export, modeling, tuning
from app.services.model_card import build_model_card

client = TestClient(app)
KEY = "store + region"


def shops(n: int = 1600, seed: int = 0) -> pd.DataFrame:
    """The same store number exists in two regions and means a different shop in each."""
    rng = np.random.default_rng(seed)
    store = rng.integers(0, 45, n)
    region = rng.choice(["north", "south"], n)
    effect = rng.normal(0, 1, (45, 2))
    signal = effect[store, (region == "south").astype(int)]
    return pd.DataFrame(
        {"store": store, "region": region, "x": rng.normal(size=n),
         "y": np.where(signal + rng.normal(0, 0.5, n) > 0.3, "yes", "no")}
    )


def train(df, **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=["logistic_regression"],
        include_deep_diagnostics=False, split_strategy="group", **kwargs,
    )


class TestTheKey:
    def test_one_label_per_combination(self):
        df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 1]})
        key = modeling.composite_group_key(df, ["a", "b"])
        assert key.nunique() == 3 and key.iloc[0] != key.iloc[1] != key.iloc[2]

    def test_the_same_combination_is_the_same_label(self):
        df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 1]})
        key = modeling.composite_group_key(df, ["a", "b"])
        assert key.iloc[0] == key.iloc[1] and key.iloc[0] != key.iloc[2]

    def test_values_that_run_together_do_not_collide(self):
        df = pd.DataFrame({"a": ["a b", "a"], "b": ["c", "b c"]})
        assert modeling.composite_group_key(df, ["a", "b"]).nunique() == 2

    def test_it_is_deterministic_and_row_wise(self):
        df = shops(200)
        once = modeling.composite_group_key(df, ["store", "region"])
        again = modeling.composite_group_key(df.sample(frac=1, random_state=1), ["store", "region"]).sort_index()
        assert once.equals(again)

    def test_the_function_is_self_contained_so_the_export_can_copy_it(self):
        import inspect

        namespace = {"pd": pd}
        exec(inspect.getsource(modeling.composite_group_key), namespace)  # noqa: S102 -- the text under test
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        assert namespace["composite_group_key"](df, ["a", "b"]).equals(modeling.composite_group_key(df, ["a", "b"]))


class TestResolvingTheRequest:
    def test_a_list_wins_over_the_single_column(self):
        assert modeling.resolve_group_columns("customer", ["store", "region"]) == ["store", "region"]

    def test_the_single_column_alone_is_a_list_of_one(self):
        assert modeling.resolve_group_columns("customer", None) == ["customer"]
        assert modeling.resolve_group_columns("customer", []) == ["customer"]

    def test_nothing_is_an_empty_list_and_repeats_are_dropped(self):
        assert modeling.resolve_group_columns(None, None) == []
        assert modeling.resolve_group_columns(None, ["a", "b", "a"]) == ["a", "b"]


class TestPreparingTheFrame:
    def test_a_single_column_is_used_exactly_as_it_is(self):
        df = shops(100)
        frame, name, withheld = modeling.with_group_key(df, ["store"], "y")
        assert frame is df and name == "store" and withheld == ["store"]

    def test_several_make_a_column_named_after_them_and_withhold_every_part(self):
        frame, name, withheld = modeling.with_group_key(shops(100), ["store", "region"], "y")
        assert name == KEY and KEY in frame.columns and withheld == ["store", "region", KEY]

    def test_it_does_not_alter_the_callers_frame(self):
        df = shops(100)
        modeling.with_group_key(df, ["store", "region"], "y")
        assert KEY not in df.columns

    @pytest.mark.parametrize(
        "columns,message",
        [(["store", "nope"], "not in the data"), (["store", "y"], "is the answer"),
         (list("abcdef"), "at most 5"), ([], "needs the column")],
    )
    def test_it_refuses_what_cannot_be_a_key(self, columns, message):
        df = shops(100).assign(**{c: 1 for c in "abcdef"})
        with pytest.raises(modeling.ModelingError, match=message):
            modeling.with_group_key(df, columns, "y")

    def test_a_blank_in_any_key_column_is_refused_naming_it(self):
        df = shops(200)
        df.loc[3, "region"] = None
        with pytest.raises(modeling.ModelingError, match="'region' is blank on some rows"):
            modeling.with_group_key(df, ["store", "region"], "y")

    def test_a_blank_on_a_row_with_no_answer_does_not_matter(self):
        """Those rows are dropped before fitting, so they never need a group."""
        df = shops(200)
        df.loc[3, ["region", "y"]] = None
        assert modeling.with_group_key(df, ["store", "region"], "y")[1] == KEY


class TestTheSplit:
    def test_every_store_and_region_pair_stays_on_one_side(self):
        df = shops()
        result = train(df, group_columns=["store", "region"])
        key = modeling.composite_group_key(df, ["store", "region"])
        X_train, X_test, _, _ = modeling.split_data(
            modeling.with_group_key(df, ["store", "region"], "y")[0], "y", "classification",
            strategy="group", group_column=KEY,
        )
        assert not (set(key.loc[X_train.index]) & set(key.loc[X_test.index]))
        assert result["split"]["train_groups"] + result["split"]["test_groups"] == 90

    def test_grouping_on_one_column_alone_would_have_mixed_the_pairs(self):
        """What the composite key is for: the store number alone treats two shops as one."""
        df = shops()
        by_store = train(df, group_column="store")["split"]
        by_pair = train(df, group_columns=["store", "region"])["split"]
        assert by_store["train_groups"] + by_store["test_groups"] == 45
        assert by_pair["train_groups"] + by_pair["test_groups"] == 90

    def test_every_grouping_column_is_withheld_from_the_inputs(self):
        features = train(shops(), group_columns=["store", "region"])["features"]
        used = features["numeric"] + features["categorical"] + features.get("frequent", [])
        assert used == ["x"]

    def test_the_result_names_the_key_and_its_parts(self):
        split = train(shops(), group_columns=["store", "region"])["split"]
        assert split["group_column"] == KEY and split["group_columns"] == ["store", "region"]

    def test_a_single_column_request_behaves_exactly_as_before(self):
        df = shops()
        old = train(df, group_column="store")
        new = train(df, group_columns=["store"])
        assert old["split"]["group_column"] == new["split"]["group_column"] == "store"
        assert old["results"][0]["metrics"] == new["results"][0]["metrics"]
        assert old["features"] == new["features"]

    def test_the_list_wins_when_both_are_given(self):
        split = train(shops(), group_column="store", group_columns=["store", "region"])["split"]
        assert split["group_column"] == KEY

    def test_only_a_grouped_split_reads_it(self):
        result = modeling.train_candidates(
            shops(), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, group_columns=["store", "region"],
        )
        assert result["split"]["strategy"] == "random" and result["split"]["group_columns"] is None


class TestEveryFoldKeepsThePairWhole:
    def test_model_choice_tuning_and_the_cut_off_group_by_the_key(self, monkeypatch):
        seen = []
        real = tuning.cv_splitter
        monkeypatch.setattr(
            tuning, "cv_splitter", lambda task, y, folds=tuning.CV_FOLDS, groups=None, ordered=False, repeated=False:
            (seen.append(groups), real(task, y, folds, groups, ordered, repeated))[1],
        )
        modeling.train_candidates(
            shops(), target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=True,
            split_strategy="group", group_columns=["store", "region"],
        )
        assert seen and all(groups is not None for groups in seen)
        assert all(g.nunique() > 20 and g.astype(str).str.contains("\x1f").all() for g in seen)

    def test_calibration_uses_it_too(self):
        result = train(shops(2000), group_columns=["store", "region"], calibration="sigmoid")
        assert result["calibration"]["method"] == "sigmoid"

    def test_the_improvement_loop_searches_on_the_key(self):
        from app.services import improvement

        found = improvement.improve(
            shops(), target_column="y", task="classification", model_names=["logistic_regression"],
            budget=4, split_strategy="group", group_columns=["store", "region"],
        )
        assert found["best"] is not None


def session(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


BODY = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "split_strategy": "group"}


class TestOverTheApi:
    def test_the_plan_offers_every_column_for_a_key(self):
        plan = client.get(f"/api/model/{session(shops())}/plan").json()
        assert plan["group_column_options"] == ["store", "region", "x"]

    def test_train_accepts_a_key_and_reports_it(self):
        body = client.post(f"/api/model/{session(shops())}/train", json={**BODY, "group_columns": ["store", "region"]}).json()
        assert body["split"]["group_column"] == KEY and body["split"]["group_columns"] == ["store", "region"]

    def test_a_blank_in_a_key_column_is_refused_with_the_column_named(self):
        df = shops()
        df.loc[5, "region"] = None
        response = client.post(f"/api/model/{session(df)}/train", json={**BODY, "group_columns": ["store", "region"]})
        assert response.status_code >= 400 and "'region' is blank on some rows" in response.text

    def test_the_answer_cannot_be_part_of_the_key(self):
        response = client.post(f"/api/model/{session(shops())}/train", json={**BODY, "group_columns": ["store", "y"]})
        assert response.status_code >= 400 and "is the answer" in response.text

    def test_too_long_a_key_is_a_validation_error(self):
        response = client.post(f"/api/model/{session(shops())}/train", json={**BODY, "group_columns": list("abcdef")})
        assert response.status_code == 422

    def test_the_old_single_column_request_still_works(self):
        body = client.post(f"/api/model/{session(shops())}/train", json={**BODY, "group_column": "store"}).json()
        assert body["split"]["group_column"] == "store"

    def test_a_sealed_holdout_can_be_opened_after_a_composite_run(self):
        sid = session(shops(2400))
        trained = client.post(f"/api/model/{sid}/train", json={**BODY, "group_columns": ["store", "region"], "sealed_holdout": True})
        assert trained.status_code == 200 and trained.json()["split"]["sealed"] is True
        assert client.post(f"/api/model/{sid}/finalise").status_code == 200

    def test_the_improvement_route_holds_the_key(self):
        body = client.post(f"/api/model/{session(shops())}/improve", json={**BODY, "budget": 4, "group_columns": ["store", "region"]}).json()
        assert body["split"]["group_columns"] == ["store", "region"]

    def test_choosing_a_key_is_recorded_against_what_was_recommended(self):
        """Neither column looks like an entity on its own (the store is a number, the region has two
        values), so the app recommends a random split and a composite key is a deliberate choice."""
        sid = session(shops())
        body = client.post(f"/api/model/{sid}/train", json={**BODY, "group_columns": ["store", "region"]}).json()
        assert body["split"]["recommended"]["strategy"] == "random"
        assert body["split"]["overridden"] is True

    def test_the_card_names_the_whole_key(self):
        result = train(shops(), group_columns=["store", "region"])
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert "keeping every store + region on one side" in card


class TestTheExportedScript:
    def run(self, tmp_path, df, result):
        (tmp_path / "data.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        script = tmp_path / "train_model.py"
        script.write_text(model_export.build_training_script(result, file_name="data.csv"), encoding="utf-8")
        done = subprocess.run([sys.executable, str(script), str(tmp_path / "data.csv")],
                              capture_output=True, text=True, cwd=tmp_path, timeout=900)
        assert done.returncode == 0, done.stderr
        return done.stdout

    def prepared(self, df=None, **kwargs):
        result = train(df if df is not None else shops(), group_columns=["store", "region"], **kwargs)
        result["features"]["excluded"] = []
        return result

    def accuracy(self, output):
        match = re.search(r"Accuracy:\s+([0-9.]+)", output)
        assert match, output
        return float(match.group(1))

    def test_the_script_builds_the_key_the_app_built(self):
        script = model_export.build_training_script(self.prepared())
        assert "def composite_group_key" in script and "GROUP_COLUMNS = ['store', 'region']" in script
        assert "'store + region'" in script

    def test_it_reproduces_the_reported_accuracy(self, tmp_path):
        df, result = shops(), self.prepared()
        output = self.run(tmp_path, df, result)
        assert self.accuracy(output) == pytest.approx(result["results"][0]["metrics"]["accuracy"], abs=1e-4)

    def test_it_reproduces_a_sealed_composite_run(self, tmp_path):
        df = shops(2400)
        result = self.prepared(df, sealed_holdout=True)
        assert result["split"]["sealed"] is True
        output = self.run(tmp_path, df, result)
        assert self.accuracy(output) == pytest.approx(result["results"][0]["metrics"]["accuracy"], abs=1e-4)

    def test_a_calibrated_composite_run_reproduces_its_brier_score(self, tmp_path):
        df = shops(2400)
        result = self.prepared(df, calibration="sigmoid")
        assert result["calibration"]["method"] == "sigmoid"
        output = self.run(tmp_path, df, result)
        match = re.search(r"Brier score:\s+([0-9.]+)", output)
        assert match and float(match.group(1)) == pytest.approx(result["results"][0]["metrics"]["brier_score"], abs=1e-4)

    def test_a_single_column_script_is_unchanged(self):
        result = train(shops(), group_column="store")
        result["features"]["excluded"] = []
        script = model_export.build_training_script(result)
        assert "composite_group_key" not in script and "GROUP_COLUMNS" not in script
        assert "groups = labelled['store']" in script
