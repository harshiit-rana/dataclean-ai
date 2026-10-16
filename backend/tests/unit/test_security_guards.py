"""Guards added by the threat review (#114, docs/security/threat-model.md). Each test pins a finding that was
demonstrated before it was fixed."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import cleaning_export
from app.services.operations import OperationFailedError, parse_operation

client = TestClient(app)


class TestSessionIds:
    """Demonstrated: with a meta.json in the uploads folder, DELETE /api/session/. removed every session."""

    @pytest.mark.parametrize("session_id", ["%2E", "%2E%2E", "not-a-session", "1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed%20"])
    def test_anything_but_a_server_made_id_is_not_found(self, session_id):
        root = Path(get_settings().upload_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "meta.json").write_text(json.dumps({"file_name": "planted"}))
        assert client.delete(f"/api/session/{session_id}").status_code == 404
        assert client.get(f"/api/session/{session_id}/project").status_code == 404
        assert (root / "meta.json").exists()

    def test_a_request_for_an_unknown_session_leaves_no_folder_behind(self):
        """Demonstrated: five unauthenticated requests left five empty folders in the uploads directory."""
        root = Path(get_settings().upload_dir)
        before = set(os.listdir(root)) if root.exists() else set()
        for n in range(5):
            client.get(f"/api/session/00000000-0000-4000-8000-00000000000{n}/project")
        after = set(os.listdir(root)) if root.exists() else set()
        assert after == before

    def test_a_real_session_still_opens(self):
        body = client.post("/api/upload", files={"file": ("a.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv")}).json()
        assert client.get(f"/api/session/{body['session_id']}/project").status_code == 200


class TestTheOperationLine:
    """The OPERATION line is written by a model that has read the uploaded file."""

    def test_an_escaped_quote_in_a_column_name_survives(self):
        """Demonstrated with Groq: a column whose name held a `"` was cut at it, so the fix named another column."""
        assert parse_operation(r'drop_column(column="say \"hi\" here")') == ("drop_column", {"column": 'say "hi" here'})

    def test_unknown_parameters_are_dropped(self):
        assert parse_operation('drop_column(column="x", inplace=True, __class__="y")') == ("drop_column", {"column": "x"})

    @pytest.mark.parametrize("line", [
        'drop_column(column=__import__("os").getcwd())',
        'drop_column(column="a" + "b")',
        'drop_column("x")',
        'drop_column(**{"column": "x"})',
        'os.system("x")',
    ])
    def test_anything_but_literal_keyword_values_is_refused(self, line):
        assert parse_operation(line) is None

    def test_what_the_old_parser_accepted_still_parses(self):
        assert parse_operation("drop_duplicate_rows") == ("drop_duplicate_rows", {})
        assert parse_operation("drop_column(column=revenue)") == ("drop_column", {"column": "revenue"})
        assert parse_operation('round_numeric(column="x", decimals=2)') == ("round_numeric", {"column": "x", "decimals": "2"})
        assert parse_operation('drop_column(column="x")\nOPERATION: drop_duplicate_rows()') == ("drop_column", {"column": "x"})


def test_the_cleaning_script_never_writes_an_unchecked_case_into_code():
    frame = pd.DataFrame({"x": ["a"]})
    with pytest.raises(OperationFailedError):
        cleaning_export._lines("standardize_case", {"column": "x", "case": "lower(); import os; os.getcwd"},
                                    {}, frame)
