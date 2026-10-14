"""The test-set ledger: a record of every look at the held-out rows.

A person who retrains, reads the score, changes something and retrains is selecting on the test
set by hand. Nothing automatic can stop that; what can be done is count the looks. The run
history cannot do it -- it keeps twenty runs and is not keyed by how the data was split -- so
these pin a ledger that is never truncated, survives the data changing underneath a model, and
counts per split rule.
"""

from __future__ import annotations

import json
import threading

import pytest

from app.services import evaluation_ledger as ledger
from app.services import model_store, modeling

RANDOM = {"strategy": "random"}
VERSION_1 = {"version": 1, "fingerprint": "a" * 64}
VERSION_2 = {"version": 2, "fingerprint": "b" * 64}


def look(directory, *, version=VERSION_1, split=None, route="train", run_id="r"):
    return ledger.record_look(
        directory, version=version, split=RANDOM if split is None else split, route=route,
        run_id=run_id, metrics={"roc_auc": 0.81},
    )


class TestTheKey:
    def test_the_same_rule_gives_the_same_key(self):
        assert ledger.split_key({"strategy": "random"}) == ledger.split_key({"strategy": "random"})

    def test_an_empty_split_is_the_default_random_rule(self):
        assert ledger.split_key(None) == ledger.split_key({}) == ledger.split_key(RANDOM)

    def test_the_defaults_are_the_apps_own(self):
        explicit = {
            "strategy": "random",
            "requested_test_size": modeling.DEFAULT_TEST_SIZE,
            "seed": modeling.RANDOM_STATE,
        }
        assert ledger.split_key(explicit) == ledger.split_key(RANDOM)

    @pytest.mark.parametrize(
        "changed",
        [
            {"strategy": "time", "time_column": "joined"},
            {"strategy": "group", "group_column": "customer"},
            {"strategy": "random", "requested_test_size": 0.25},
            {"strategy": "random", "seed": 7},
            {"strategy": "random", "final_fraction": 0.15},
            {"strategy": "random", "keep_duplicates_together": True},
        ],
    )
    def test_any_differing_field_gives_a_different_key(self, changed):
        assert ledger.split_key(changed) != ledger.split_key(RANDOM)

    def test_a_time_split_on_another_column_is_a_different_rule(self):
        assert ledger.split_key({"strategy": "time", "time_column": "a"}) != ledger.split_key(
            {"strategy": "time", "time_column": "b"}
        )

    def test_a_column_the_strategy_does_not_use_does_not_matter(self):
        """A leftover group column on a random split holds out the same rows."""
        assert ledger.split_key({"strategy": "random", "group_column": "x"}) == ledger.split_key(
            RANDOM
        )

    def test_the_fraction_a_date_split_ended_up_with_is_not_part_of_the_key(self):
        """A date split moves its cut to a timestamp boundary, so the fraction actually held
        out varies with the data. A key that moved with it would reset the count."""
        a = {"strategy": "time", "time_column": "d", "test_fraction": 0.2, "test_rows": 100}
        b = {"strategy": "time", "time_column": "d", "test_fraction": 0.27, "test_rows": 140}
        assert ledger.split_key(a) == ledger.split_key(b)

    def test_a_trained_result_carries_the_rule_that_made_its_split(self):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.choice(["a", "b"], 200)})
        result = modeling.train_candidates(
            df, target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        assert result["split"]["requested_test_size"] == modeling.DEFAULT_TEST_SIZE
        assert result["split"]["seed"] == modeling.RANDOM_STATE
        assert ledger.split_key(result["split"]) == ledger.split_key(RANDOM)


class TestRecording:
    def test_the_file_is_created_on_the_first_look(self, tmp_path):
        assert not (tmp_path / model_store.LEDGER_FILENAME).exists()
        look(tmp_path)
        assert (tmp_path / model_store.LEDGER_FILENAME).exists()

    def test_each_look_is_appended(self, tmp_path):
        for _ in range(3):
            look(tmp_path)
        assert len(ledger.looks(tmp_path)) == 3

    def test_an_entry_says_what_where_and_when(self, tmp_path):
        entry = look(tmp_path, route="improve", run_id="abc123")
        assert entry["event"] == ledger.LOOK
        assert entry["route"] == "improve" and entry["run_id"] == "abc123"
        assert entry["metrics"] == {"roc_auc": 0.81}
        assert entry["split"] == {"strategy": "random"}
        assert entry["at"]

    def test_the_full_fingerprint_is_stored_not_the_display_prefix(self, tmp_path):
        entry = look(tmp_path)
        assert entry["dataset_fingerprint"] == "a" * 64
        assert entry["dataset_version"] == 1

    def test_a_look_with_no_known_version_is_still_recorded(self, tmp_path):
        entry = look(tmp_path, version=None)
        assert entry["dataset_version"] is None and entry["dataset_fingerprint"] is None
        assert ledger.look_count(tmp_path, None, RANDOM) == 1

    def test_numpy_metrics_survive_the_round_trip(self, tmp_path):
        import numpy as np

        ledger.record_look(
            tmp_path, version=VERSION_1, split=RANDOM, route="train", run_id="r",
            metrics={"roc_auc": np.float64(0.5), "rows": np.int64(60)},
        )
        assert ledger.looks(tmp_path)[0]["metrics"] == {"roc_auc": 0.5, "rows": 60}

    def test_it_is_never_truncated(self, tmp_path):
        """The run history keeps twenty runs; a count derived from it would undercount."""
        for i in range(250):
            look(tmp_path, run_id=str(i))
        assert ledger.look_count(tmp_path, None, RANDOM) == 250
        assert len(ledger.looks(tmp_path)) == 250


class TestCounting:
    def test_a_random_and_a_time_split_have_independent_counts(self, tmp_path):
        look(tmp_path, split={"strategy": "random"})
        look(tmp_path, split={"strategy": "random"})
        look(tmp_path, split={"strategy": "time", "time_column": "d"})
        assert ledger.look_count(tmp_path, None, {"strategy": "random"}) == 2
        assert ledger.look_count(tmp_path, None, {"strategy": "time", "time_column": "d"}) == 1

    def test_a_dataset_version_does_not_reset_the_status_count(self, tmp_path):
        """Cleaning after a look is the commonest way to select on the test set by hand."""
        look(tmp_path, version=VERSION_1)
        look(tmp_path, version=VERSION_2)
        assert ledger.look_count(tmp_path, None, RANDOM) == 2

    def test_a_single_version_can_still_be_asked_about(self, tmp_path):
        look(tmp_path, version=VERSION_1)
        look(tmp_path, version=VERSION_2)
        look(tmp_path, version=VERSION_2)
        assert ledger.look_count(tmp_path, 1, RANDOM) == 1
        assert ledger.look_count(tmp_path, 2, RANDOM) == 2

    def test_evaluations_on_new_data_are_never_counted_against_the_held_out_rows(self, tmp_path):
        look(tmp_path)
        ledger.record(
            tmp_path, event=ledger.NEW_DATA, split=RANDOM, route="apply_answers", run_id="a1",
            version=VERSION_1, metrics={"roc_auc": 0.7},
        )
        assert ledger.look_count(tmp_path, None, RANDOM) == 1
        assert len(ledger.looks(tmp_path)) == 2

    def test_the_sealed_final_holdout_is_its_own_event(self, tmp_path):
        ledger.record(
            tmp_path, event=ledger.FINAL, split=RANDOM, route="finalise", run_id="f",
            version=VERSION_1, metrics={},
        )
        assert ledger.look_count(tmp_path, None, RANDOM) == 0

    def test_looks_under_other_split_rules_are_reported_separately(self, tmp_path):
        look(tmp_path, split={"strategy": "random"})
        look(tmp_path, split={"strategy": "time", "time_column": "d"})
        look(tmp_path, split={"strategy": "time", "time_column": "d"})
        assert ledger.looks_on_other_splits(tmp_path, {"strategy": "random"}) == 2
        assert ledger.looks_on_other_splits(tmp_path, {"strategy": "time", "time_column": "d"}) == 1

    def test_an_empty_session_has_no_looks(self, tmp_path):
        assert ledger.look_count(tmp_path, None, RANDOM) == 0
        assert ledger.looks(tmp_path) == []


class TestItSurvivesTheDataChanging:
    def test_clearing_the_model_keeps_the_ledger(self, tmp_path):
        """`clear` fires on every undo and chat instruction. A ledger it wiped would hand the
        user a fresh 'independent' score after one undo."""
        look(tmp_path)
        (tmp_path / model_store.RESULT_FILENAME).write_text("{}")
        model_store.clear(tmp_path)
        assert not (tmp_path / model_store.RESULT_FILENAME).exists()
        assert ledger.look_count(tmp_path, None, RANDOM) == 1

    def test_the_ledger_is_not_an_artifact_clear_removes(self):
        assert model_store.LEDGER_FILENAME not in model_store.ALL_ARTIFACTS


class TestItNeverFailsTheResultItDescribes:
    def test_an_unwritable_location_does_not_raise(self, tmp_path):
        assert look(tmp_path / "does" / "not" / "exist") is None

    def test_unserialisable_metrics_do_not_raise(self, tmp_path):
        assert (
            ledger.record_look(
                tmp_path, version=VERSION_1, split=RANDOM, route="train", run_id="r",
                metrics={"bad": object()},
            )
            is not None
        )

    def test_an_unreadable_ledger_counts_as_empty_and_does_not_raise(self, tmp_path):
        (tmp_path / model_store.LEDGER_FILENAME).write_text("{ not json")
        assert ledger.look_count(tmp_path, None, RANDOM) == 0

    def test_an_unreadable_ledger_is_set_aside_not_overwritten(self, tmp_path):
        """Starting a new one over it would silently destroy what this file exists to keep."""
        path = tmp_path / model_store.LEDGER_FILENAME
        path.write_text("{ not json")
        look(tmp_path)
        assert (tmp_path / (model_store.LEDGER_FILENAME + ".unreadable")).read_text() == "{ not json"
        assert ledger.look_count(tmp_path, None, RANDOM) == 1

    def test_a_ledger_of_the_wrong_shape_is_ignored(self, tmp_path):
        (tmp_path / model_store.LEDGER_FILENAME).write_text(json.dumps({"not": "a list"}))
        assert ledger.looks(tmp_path) == []

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        look(tmp_path)
        assert not list(tmp_path.glob("*.tmp"))


class TestConcurrentLooks:
    def test_two_tabs_training_at_once_record_two_looks(self, tmp_path):
        """Both revealed a score, so both count -- and neither may overwrite the other."""
        barrier = threading.Barrier(8)

        def worker(i):
            barrier.wait()
            look(tmp_path, run_id=str(i))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert ledger.look_count(tmp_path, None, RANDOM) == 8
        assert {e["run_id"] for e in ledger.looks(tmp_path)} == {str(i) for i in range(8)}


class TestThresholds:
    def test_the_status_thresholds_are_the_decided_ones(self):
        """ADR 0001 section 3: independent on the first look, exhausted from the fourth."""
        assert ledger.INDEPENDENT_LOOKS == 1
        assert ledger.EXHAUSTED_AT_LOOK == 4
