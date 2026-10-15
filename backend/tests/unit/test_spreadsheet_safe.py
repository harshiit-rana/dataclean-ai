"""CSV downloads a spreadsheet cannot run as formulas, while the data the app reads stays exact."""

from __future__ import annotations

import io
import pathlib
import re
import zipfile
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import spreadsheet_safe
from app.services.audit_report import build_markdown_report
from app.services.session_store import session_dir

client = TestClient(app)


def risky_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "note": ["=HYPERLINK(\"http://x\",\"click\")", "+1-2", "@SUM(A1)", "\tcmd", "plain", "-5", "-3.2e1"],
        "amount": [-5.0, 1.0, -2.5, 3.0, 4.0, -1.0, 0.0],
    })


class TestTheRule:
    def test_formula_leading_text_gets_a_quote(self):
        safe, count = spreadsheet_safe.neutralise(risky_frame())
        assert safe["note"].tolist()[:4] == ["'=HYPERLINK(\"http://x\",\"click\")", "'+1-2", "'@SUM(A1)", "'\tcmd"]
        assert count == 4

    def test_plain_text_and_numbers_written_as_text_are_left_alone(self):
        safe, _ = spreadsheet_safe.neutralise(risky_frame())
        assert safe["note"].tolist()[4:] == ["plain", "-5", "-3.2e1"]

    def test_numeric_columns_are_never_touched(self):
        safe, _ = spreadsheet_safe.neutralise(risky_frame())
        assert safe["amount"].tolist() == risky_frame()["amount"].tolist()

    def test_a_stored_file_keeps_every_other_value_as_it_was_written(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text("note,amount\n=1+1,10.0\nok,-5\n", encoding="utf-8")
        assert spreadsheet_safe.csv_bytes(path).decode() == "note,amount\n'=1+1,10.0\nok,-5\n"

    def test_blanks_stay_blank(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text("note,amount\n,1\n=x,\n", encoding="utf-8")
        assert spreadsheet_safe.csv_bytes(path).decode() == "note,amount\n,1\n'=x,\n"


def session_with_cleaned_file() -> tuple[str, pathlib.Path]:
    buffer = io.StringIO()
    risky_frame().to_csv(buffer, index=False)
    sid = client.post("/api/upload", files={"file": ("r.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, sid)
    risky_frame().to_csv(directory / "cleaned.csv", index=False)
    return sid, directory


class TestTheDownloads:
    def test_the_cleaned_file_download_is_escaped_and_the_stored_file_is_not(self):
        sid, directory = session_with_cleaned_file()
        response = client.get(f"/api/export/{sid}/csv")
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]
        downloaded = pd.read_csv(io.StringIO(response.text), dtype=str, keep_default_na=False)
        assert downloaded["note"][0].startswith("'=HYPERLINK")
        assert downloaded["amount"][0] == "-5.0"
        assert pd.read_csv(directory / "cleaned.csv")["note"][0].startswith("=HYPERLINK")

    def test_the_bundle_s_csv_is_escaped(self, monkeypatch):
        class FakeGraph:
            async def aget_state(self, _config):
                return SimpleNamespace(values={"action_log": []}, next=())

        monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
        sid, _ = session_with_cleaned_file()
        bundle = client.get(f"/api/export/{sid}/zip")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            name = next(n for n in archive.namelist() if n.endswith("_cleaned.csv"))
            assert "'=HYPERLINK" in archive.read(name).decode()

    def test_an_applied_file_download_is_escaped(self):
        sid, directory = session_with_cleaned_file()
        folder = directory / "applied" / "0123456789ab"
        folder.mkdir(parents=True)
        risky_frame().to_csv(folder / "predictions.csv", index=False)
        response = client.get(f"/api/apply/{sid}/0123456789ab/predictions")
        assert response.status_code == 200 and "'@SUM(A1)" in response.text
        assert "@SUM(A1)" in (folder / "predictions.csv").read_text() and "'@SUM" not in (folder / "predictions.csv").read_text()


class TestEveryDownloadIsCovered:
    def test_no_csv_is_served_except_through_the_helper(self):
        app_dir = pathlib.Path(__file__).resolve().parents[2] / "app"
        offenders = []
        for path in app_dir.rglob("*.py"):
            if path.name == "spreadsheet_safe.py":
                continue
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r'media_type="text/csv"', text):
                offenders.append(f"{path.name}:{text[: match.start()].count(chr(10)) + 1}")
        assert offenders == []


class TestTheReport:
    def test_the_report_says_how_many_cells_the_download_escapes(self):
        report = build_markdown_report(
            session_id="s", file_name="f.csv", goal_context=None, issues=[], action_log=[],
            executive_summary="x", dispositions={}, formula_cells=4,
        )
        assert "| Cells a spreadsheet would read as formulas | 4 --" in report

    def test_the_cleaning_script_says_why_its_output_is_not_escaped(self):
        from app.services import cleaning_export

        script = cleaning_export.build_cleaning_script(risky_frame(), [])
        assert "prefix such cells with a '" in script
